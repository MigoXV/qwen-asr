# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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

__all__ = ["Qwen3ASRConfig", "Qwen3ASRForConditionalGeneration", "Qwen3ASRProcessor"]


def __getattr__(name):
    if name == "Qwen3ASRConfig":
        from .configuration_qwen3_asr import Qwen3ASRConfig
        return Qwen3ASRConfig
    if name == "Qwen3ASRForConditionalGeneration":
        from .modeling.model import Qwen3ASRForConditionalGeneration
        return Qwen3ASRForConditionalGeneration
    if name == "Qwen3ASRProcessor":
        from .processing_qwen3_asr import Qwen3ASRProcessor
        return Qwen3ASRProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
