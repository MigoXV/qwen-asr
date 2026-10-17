# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

from .modeling.layers import (
    Qwen3ASRTextRMSNorm,
    rotate_half,
    repeat_kv,
    eager_attention_forward,
    apply_rotary_pos_emb,
    Qwen3ASRTextAttention,
    Qwen3ASRTextMLP,
    Qwen3ASRThinkerTextMLP,
    Qwen3ASRThinkerTextRMSNorm,
    Qwen3ASRThinkerTextAttention,
)
from .modeling.base import (
    Qwen3ASRPreTrainedModel,
    Qwen3ASRThinkerCausalLMOutputWithPast,
    Qwen3ASRPreTrainedModelForConditionalGeneration,
)
from .modeling.audio import (
    Qwen3ASRAudioAttention,
    Qwen3ASRAudioEncoderLayer,
    Qwen3ASRAudioEncoder,
)
from .modeling.rotary import (
    Qwen3ASRThinkerTextRotaryEmbedding,
)
from .modeling.text import (
    Qwen3ASRThinkerTextDecoderLayer,
    Qwen3ASRThinkerTextModel,
    Qwen3ASRThinkerTextPreTrainedModel,
)
from .modeling.model import (
    Qwen3ASRThinkerForConditionalGeneration,
    Qwen3ASRForConditionalGeneration,
)

from .configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig, Qwen3ASRConfig, Qwen3ASRThinkerConfig
from qwen_asr.models.common.audio_lengths import get_audio_output_lengths as _get_feat_extract_output_lengths
from qwen_asr.models.common.position_embedding import SinusoidsPositionEmbedding

__all__ = [
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRThinkerTextModel",
    "Qwen3ASRThinkerForConditionalGeneration",
    "Qwen3ASRPreTrainedModel",
    "Qwen3ASRPreTrainedModelForConditionalGeneration",
    "Qwen3ASRThinkerTextPreTrainedModel",
]
