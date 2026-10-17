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
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any, Dict, List, Optional

import numpy as np

from qwen_asr.engines.vllm.engine import VLLMEngine
from qwen_asr.inferencers.language import resolve_language_code

logger = logging.getLogger(__name__)


class VLLMInferencer:
    """将 ASR 音频、语言和上下文转换为 vLLM 多模态请求。"""

    def __init__(self, engine: VLLMEngine):
        self.engine = engine
        self.processor = engine.processor

    @classmethod
    def LLM(cls, model: str, max_new_tokens: Optional[int] = 4096, **kwargs):
        return cls(VLLMEngine.load(model=model, max_new_tokens=max_new_tokens, **kwargs))

    async def aclose(self) -> None:
        await self.engine.aclose()

    @property
    def errored(self) -> bool:
        return self.engine.errored

    async def transcribe_stream(
        self,
        audio: np.ndarray,
        sample_rate: int,
        context: str = "",
        language: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        对单条音频进行流式转写，逐步产出模型生成的文本片段。

        说明：
          - 输入音频仅按单块处理，不再做内部切块。
          - 输出为模型原始生成的增量文本片段；可通过拼接所有片段得到完整 raw 输出。
          - 若希望得到最终 language/text，请在适配层使用文本后处理工具解析。
        """
        force_lang = self._normalize_force_language(language)
        prompt = self._build_text_prompt(context=context, force_language=force_lang)
        inp = {"prompt": prompt, "multi_modal_data": {"audio": [(audio, sample_rate)]}}
        stream = self.engine.generate_stream(inp)
        async with aclosing(stream):
            async for chunk in stream:
                yield chunk

    def _build_messages(self, context: str, audio_payload: Any) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    def _normalize_force_language(self, language: Optional[str]) -> Optional[str]:
        resolved = resolve_language_code(language)
        if language is not None and str(language).strip() != "" and resolved is None:
            logger.warning(
                "Language code '%s' not recognised, falling back to auto-detect.",
                language,
            )
        elif resolved is not None and resolved != language:
            logger.debug("Language resolved: '%s' -> '%s'", language, resolved)
        return resolved

    def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
        """
        构造单次请求使用的字符串 prompt。

        如果提供 force_language，会在生成提示后追加 "language X<asr_text>"，
        要求模型只输出 ASR 文本。
        """
        msgs = self._build_messages(context=context, audio_payload="")
        base = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        if force_language:
            base = base + f"language {force_language}{'<asr_text>'}"
        return base
