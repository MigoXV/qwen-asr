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

from .config_audio import (
    Qwen3ASRAudioEncoderConfig,
)
from .config_text import (
    Qwen3ASRTextConfig,
)
from .config_model import (
    Qwen3ASRThinkerConfig,
    Qwen3ASRConfig,
)



__all__ = ["Qwen3ASRConfig", "Qwen3ASRThinkerConfig", "Qwen3ASRAudioEncoderConfig"]
