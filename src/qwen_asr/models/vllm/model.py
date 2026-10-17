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
import logging
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Literal, cast
import numpy as np
import torch
import torch.nn as nn
from transformers.models.whisper import WhisperFeatureExtractor
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen2_5OmniAudioFeatureInputs
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.model_executor.models.whisper import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config
from ..transformers.configuration_qwen3_asr import Qwen3ASRThinkerConfig
from qwen_asr.models.common.audio_lengths import (
    get_audio_output_lengths as _get_feat_extract_output_lengths,
)
from .audio import (
    Qwen3ASRAudioEncoder,
)
from .processing import (
    Qwen3ASRProcessingInfo,
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRMultiModalProcessor,
)


logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRMultiModalProcessor,
    info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    SupportsTranscription,
):
    supported_languages = ISO639_1_SUPPORTED_LANGS

    # Platform backends override only the encoder, sharing processing, weights,
    # text decoder and streaming inference with the standard vLLM backend.
    audio_encoder_cls = Qwen3ASRAudioEncoder

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config  # needed for torch compile forward context
        thinker_config: Qwen3ASRThinkerConfig = (
            vllm_config.model_config.hf_config.thinker_config
        )
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config

        tower_context = (
            self._mark_tower_model(vllm_config, "audio")
            if hasattr(self, "_mark_tower_model") else nullcontext()
        )
        with tower_context:
            self.audio_tower = self.audio_encoder_cls(
                thinker_config.audio_config,
                multimodal_config=multimodal_config,
                prefix=maybe_prefix(prefix, "audio_tower"),
            )
        self.quant_config = quant_config

        language_context = (
            self._mark_language_model(vllm_config)
            if hasattr(self, "_mark_language_model") else nullcontext()
        )
        with language_context:
            self.language_model = Qwen3ForCausalLM(
                vllm_config=vllm_config.with_hf_config(
                    thinker_config.text_config, architectures=["Qwen3ForCausalLM"]
                ),
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self._embed_accepts_oov = (
            "handle_oov_mm_token" in inspect.signature(self._embed_text_input_ids).parameters
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Qwen2_5OmniAudioFeatureInputs | None:
        input_audio_features = kwargs.pop("input_audio_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        if input_audio_features is None:
            return None

        return Qwen2_5OmniAudioFeatureInputs(
            type="audio_features",
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
            feature_attention_mask=feature_attention_mask,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("input_audio_features")
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        return mm_input_by_modality

    def _process_audio_input(
        self,
        audio_input: Qwen2_5OmniAudioFeatureInputs,
        audio_hashes: list[str] | None = None,
        cached_audio_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]
        log_cuda_memory = logger.isEnabledFor(logging.DEBUG) and input_features.is_cuda
        if log_cuda_memory:
            allocated_before = torch.cuda.memory_allocated(input_features.device)
            reserved_before = torch.cuda.memory_reserved(input_features.device)

        audio_output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)

        audio_features = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_output_lengths,
        )
        outputs = audio_features.split(audio_output_lengths.tolist())
        if log_cuda_memory:
            output_bytes = sum(item.numel() * item.element_size() for item in outputs)
            logger.debug(
                "Audio encoder CUDA memory: allocated %.1f -> %.1f MiB, "
                "reserved %.1f -> %.1f MiB, cached output %.1f MiB",
                allocated_before / 2**20,
                torch.cuda.memory_allocated(input_features.device) / 2**20,
                reserved_before / 2**20,
                torch.cuda.memory_reserved(input_features.device) / 2**20,
                output_bytes / 2**20,
            )
        return outputs

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += tuple(audio_embeddings)
        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        embed_kwargs = (
            {"handle_oov_mm_token": handle_oov_mm_token}
            if self._embed_accepts_oov else {}
        )
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            **embed_kwargs,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["talker.", "code2wav."],
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        return loaded_weights

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        seq_len = len(input_tokens)

        if not mm_features:
            # No audio features, just return linear positions
            llm_positions = (
                torch.arange(seq_len, dtype=torch.long).view(1, -1).expand(3, -1)
            )
            return llm_positions.clone(), 0

        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0

        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset

            # Get audio feature length from mm_feature data
            audio_feature_length = mm_feature.data["audio_feature_lengths"].data
            if isinstance(audio_feature_length, torch.Tensor):
                audio_feature_length = audio_feature_length.item()
            audio_len = _get_feat_extract_output_lengths(
                torch.tensor(audio_feature_length)
            ).item()

            # Text segment before audio (includes audio_start token)
            text_len = offset - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_positions = (
                torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )
            llm_pos_ids_list.append(text_positions)
            st_idx = st_idx + text_len

            # Audio token segment
            audio_positions = (
                torch.arange(audio_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )
            llm_pos_ids_list.append(audio_positions)

            st = offset + audio_len

        # Handle remaining text (includes audio_end and any trailing text)
        if st < seq_len:
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = seq_len - st
            final_text_positions = (
                torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )
            llm_pos_ids_list.append(final_text_positions)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if llm_positions.shape[1] != seq_len:
            raise RuntimeError("Position ids length mismatch with input ids length")

        mrope_position_delta = (llm_positions.max() + 1 - seq_len).item()
        return llm_positions, mrope_position_delta

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            tower_model=["audio_tower."],
        )

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)
        feature_extractor: WhisperFeatureExtractor = processor.feature_extractor
        return SpeechToTextConfig(
            max_audio_clip_s=feature_extractor.chunk_length,
            sample_rate=feature_extractor.sampling_rate,
        )

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        """Get the generation prompt to be used for transcription requests."""
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_placeholder = cls.get_placeholder_str("audio", 0)

        if task_type not in ("transcribe", "translate"):
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                "Supported task types are 'transcribe' and 'translate'."
            )
        full_lang_name_to = cls.supported_languages.get(to_language, to_language)
        if to_language is None:
            prompt = (
                f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt = (
                f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
                f"<|im_start|>assistant\nlanguage {full_lang_name_to}<asr_text>"
            )

        prompt_token_ids = tokenizer.encode(prompt)
        prompt_dict = {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"audio": audio},
        }
        return cast(PromptType, prompt_dict)
