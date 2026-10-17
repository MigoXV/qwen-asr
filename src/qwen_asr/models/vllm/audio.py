# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import MultiModalConfig
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
except ModuleNotFoundError:
    from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
)
from ..transformers.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.models.common.position_embedding import SinusoidsPositionEmbedding
from .audio_layers import (
    Qwen3ASRAudioEncoderLayer,
)


class Qwen3ASRAudioEncoder(nn.Module):
    """vLLM-native Qwen3-ASR Audio Encoder."""

    def __init__(
        self,
        config: Qwen3ASRAudioEncoderConfig,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

        # Position embedding
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )

        # Convolutional layers for mel-spectrogram processing
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )

        conv_out_dim = config.downsample_hidden_size * (
            (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        )
        self.conv_out = nn.Linear(conv_out_dim, config.d_model, bias=False)

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3ASRAudioEncoderLayer(
                    config,
                    multimodal_config=multimodal_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.encoder_layers)
            ]
        )

        # Output layers
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = _ACTIVATION_REGISTRY[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

        # Get attention backend
        attn_backend_override = (
            multimodal_config.mm_encoder_attn_backend
            if multimodal_config is not None
            else None
        )
        backend_kwargs = {}
        if "attn_backend_override" in inspect.signature(get_vit_attn_backend).parameters:
            backend_kwargs["attn_backend_override"] = attn_backend_override
        self.attn_backend = get_vit_attn_backend(
            head_size=config.d_model // config.encoder_attention_heads,
            dtype=torch.get_default_dtype(),
            **backend_kwargs,
        )

    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> torch.Tensor | None:
        """Compute max_seqlen only for flash attention backends."""
        max_seqlen = None
        if self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        return max_seqlen

    def prepare_attention_inputs(self, cu_seqlens: torch.Tensor):
        """Allow platform encoders to prepare packed-window metadata once."""
        return cu_seqlens, self.compute_attn_mask_seqlen(cu_seqlens)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv2d1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.conv2d1.weight.device

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        aftercnn_lens: torch.Tensor,
    ):
        # Compute chunk information
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        # Split input features into chunks and pad
        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)

        # Compute feature lengths after CNN
        feature_lens_after_cnn = self._get_cnn_output_lengths(chunk_lengths)
        # Vectorized mask creation: avoid creating many small tensors
        max_len_after_cnn = feature_lens_after_cnn.max().item()
        indices = torch.arange(max_len_after_cnn, device=padded_feature.device)
        padded_mask_after_cnn = indices.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(
            1
        )

        # Add channel dimension for conv2d
        padded_feature = padded_feature.unsqueeze(1)

        # Apply convolutional layers (chunk if needed to avoid OOM)
        if padded_feature.size(0) <= self.conv_chunksize:
            # Fast path: no chunking needed
            padded_embed = F.gelu(self.conv2d1(padded_feature))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
        else:
            # Chunked processing to avoid OOM
            padded_embeds = []
            for chunk in padded_feature.split(self.conv_chunksize, dim=0):
                padded_embed = F.gelu(self.conv2d1(chunk))
                padded_embed = F.gelu(self.conv2d2(padded_embed))
                padded_embed = F.gelu(self.conv2d3(padded_embed))
                padded_embeds.append(padded_embed)
            padded_embed = torch.cat(padded_embeds, dim=0)

        # (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )

        # Add positional embedding
        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding

        # Extract valid hidden states and compute cu_seqlens
        hidden_states = padded_embed[padded_mask_after_cnn]

        # Compute cumulative sequence lengths for chunked attention
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (
            self.n_window_infer // (self.n_window * 2)
        )
        # Use tolist() for efficient batch conversion from tensor to Python
        for cnn_len in aftercnn_lens.tolist():
            num_full_chunks = cnn_len // window_aftercnn
            remainder = cnn_len % window_aftercnn
            cu_chunk_lens.extend([window_aftercnn] * num_full_chunks)
            if remainder:
                cu_chunk_lens.append(remainder)
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(
            -1, dtype=torch.int32
        )

        cu_seqlens, max_seqlen = self.prepare_attention_inputs(cu_seqlens)

        return self._forward_transformer(hidden_states, cu_seqlens, max_seqlen)

    def _forward_transformer(self, hidden_states, cu_seqlens, max_seqlen):
        # Platform runners may capture this tensor-only portion independently
        # of the dynamic convolution/chunk preparation above.
        # Apply transformer layers
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
                max_seqlen,
            )

        # Apply output layers
        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)

        return hidden_states

    def _get_cnn_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output lengths after the three conv2d layers."""
        lengths = input_lengths
        for _ in range(3):
            lengths = (lengths - 1) // 2 + 1
        return lengths

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with mapping from HuggingFace format."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("self_attn.qkv.", "self_attn.q_proj.", "q"),
            ("self_attn.qkv.", "self_attn.k_proj.", "k"),
            ("self_attn.qkv.", "self_attn.v_proj.", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict.get(name)
                if param is not None:
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
