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

import inspect

import torch
import torch.nn as nn
from vllm.config import MultiModalConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
try:
    from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
except ModuleNotFoundError:
    # vLLM 0.13 keeps attention modules under vllm.attention.
    from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from ..transformers.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig


class Qwen3ASRAudioAttention(nn.Module):
    """Multi-headed attention for Qwen3-Omni Audio Encoder using MMEncoderAttention."""

    def __init__(
        self,
        config: Qwen3ASRAudioEncoderConfig,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = self.num_heads // tp_size

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scaling = self.head_dim**-0.5

        self.qkv = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,
            bias=True,
            prefix=f"{prefix}.qkv",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=True,
            prefix=f"{prefix}.out_proj",
        )

        attention_kwargs = {}
        if "multimodal_config" in inspect.signature(MMEncoderAttention.__init__).parameters:
            attention_kwargs["multimodal_config"] = multimodal_config
        self.attn = MMEncoderAttention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            **attention_kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_length, _ = hidden_states.size()
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(1, seq_length, -1, self.head_dim)
        k = k.view(1, seq_length, -1, self.head_dim)
        v = v.view(1, seq_length, -1, self.head_dim)

        attn_output = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        attn_output = attn_output.view(seq_length, -1)
        output, _ = self.out_proj(attn_output)
        return output


class Qwen3ASRAudioEncoderLayer(nn.Module):
    """Transformer encoder layer for Qwen3-Omni Audio Encoder."""

    def __init__(
        self,
        config: Qwen3ASRAudioEncoderConfig,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3ASRAudioAttention(
            config, multimodal_config=multimodal_config, prefix=f"{prefix}.self_attn"
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = _ACTIVATION_REGISTRY[config.activation_function]
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.encoder_ffn_dim,
            bias=True,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            self.embed_dim,
            bias=True,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_size)
            cu_seqlens: Cumulative sequence lengths
            max_seqlen: Maximum sequence length in the batch
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        # Clamp for numerical stability with fp16
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        return hidden_states
