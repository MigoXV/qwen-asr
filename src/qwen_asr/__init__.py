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
"""
qwen_asr: Qwen3-ASR package.
"""

try:
    from .inferencers.text.asr_output import parse_asr_output
except ImportError:  # pragma: no cover - optional during lightweight tests
    parse_asr_output = None

try:
    from .inferencers.vllm import VLLMInferencer
except ImportError:  # pragma: no cover - optional during lightweight tests
    VLLMInferencer = None

try:
    from .inferencers.transformers import TransformersInferencer
except ImportError:  # pragma: no cover - optional during lightweight tests
    TransformersInferencer = None

__all__ = [
    "TransformersInferencer",
    "VLLMInferencer",
    "parse_asr_output",
]
