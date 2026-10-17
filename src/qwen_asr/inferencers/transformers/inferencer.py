# coding=utf-8
from __future__ import annotations

from contextlib import aclosing
from typing import AsyncIterator, Optional

import numpy as np

from qwen_asr.engines.transformers.engine import TransformersEngine
from qwen_asr.inferencers.language import resolve_language_code


class TransformersInferencer:
    """构造 ASR 提示词与 Processor 输入，编排流式转写。"""

    def __init__(self, engine: TransformersEngine):
        self.engine = engine
        self.processor = engine.processor

    @classmethod
    def LLM(cls, model: str, max_new_tokens: int = 4096, device: str | None = None, **kwargs):
        return cls(TransformersEngine.load(
            model=model, max_new_tokens=max_new_tokens, device=device, **kwargs,
        ))

    async def transcribe_stream(
        self,
        audio: np.ndarray,
        sample_rate: int,
        context: str = "",
        language: Optional[str] = None,
    ) -> AsyncIterator[str]:
        msgs = [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        force_lang = resolve_language_code(language)
        if force_lang:
            prompt += f"language {force_lang}<asr_text>"

        inputs = self.processor(
            text=[prompt],
            audio=[audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        async with aclosing(self.engine.generate_stream(inputs)) as stream:
            async for chunk in stream:
                yield chunk
