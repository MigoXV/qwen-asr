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

"""兼容导出入口；实现按职责拆分到同目录模块。"""

from .audio_layers import (
    Qwen3ASRAudioAttention,
    Qwen3ASRAudioEncoderLayer,
)
from .audio import (
    Qwen3ASRAudioEncoder,
)
from .processing import (
    Qwen3ASRProcessingInfo,
    Qwen3ASRDummyInputsBuilder,
    _qwen3asr_field_config,
    Qwen3ASRMultiModalDataParser,
    Qwen3ASRMultiModalProcessor,
)
from .model import (
    Qwen3ASRForConditionalGeneration,
)

from ..transformers.configuration_qwen3_asr import Qwen3ASRConfig, Qwen3ASRThinkerConfig, Qwen3ASRAudioEncoderConfig
from ..transformers.processing_qwen3_asr import Qwen3ASRProcessor
from qwen_asr.models.common.audio_lengths import get_audio_output_lengths as _get_feat_extract_output_lengths
from qwen_asr.models.common.position_embedding import SinusoidsPositionEmbedding

__all__ = ['Qwen3ASRAudioAttention', 'Qwen3ASRAudioEncoderLayer', 'Qwen3ASRAudioEncoder', 'Qwen3ASRProcessingInfo', 'Qwen3ASRDummyInputsBuilder', 'Qwen3ASRMultiModalDataParser', 'Qwen3ASRMultiModalProcessor', 'Qwen3ASRForConditionalGeneration']
