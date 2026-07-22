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

from typing import Any

__all__ = [
    "TransformersInferencer",
    "VLLMInferencer",
    "parse_asr_output",
]


def __getattr__(name: str) -> Any:
    """按需导入后端，避免 CLI 启动时加载未选择的推理框架。"""
    if name == "parse_asr_output":
        from .inferencers.text.asr_output import parse_asr_output

        return parse_asr_output
    if name == "VLLMInferencer":
        from .inferencers.vllm import VLLMInferencer

        return VLLMInferencer
    if name == "TransformersInferencer":
        from .inferencers.transformers import TransformersInferencer

        return TransformersInferencer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
