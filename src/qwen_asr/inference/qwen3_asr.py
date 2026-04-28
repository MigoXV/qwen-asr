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
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from transformers import AutoConfig, AutoProcessor
from vllm import ModelRegistry, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind

from qwen_asr.core.vllm_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

from .utils import (
    AudioLike,
    filter_async_engine_kwargs,
    normalize_audio_input,
    parse_asr_output,
    resolve_language_code,
)

AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)

ModelRegistry.register_model(
    "Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration
)

logger = logging.getLogger(__name__)


@dataclass
class ASRTranscription:
    language: str
    text: str


class Qwen3ASRModel:
    """
    Unified inference wrapper for Qwen3-ASR with vLLM backend only.

    Notes:
      - Each request uses a context text and exactly one audio.
      - If language is provided, the prompt will force the output to be text-only by appending
        "language {Language}<asr_text>" to the assistant prompt.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        sampling_params: Any,
    ):
        self.model = model
        self.processor = processor
        self.sampling_params = sampling_params

    @classmethod
    def LLM(
        cls,
        model: str,
        max_new_tokens: Optional[int] = 4096,
        **kwargs,
    ) -> "Qwen3ASRModel":
        engine_kwargs = filter_async_engine_kwargs(AsyncEngineArgs, kwargs)
        llm = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(model=model, **engine_kwargs)
        )
        processor = Qwen3ASRProcessor.from_pretrained(model, fix_mistral_regex=True)
        sampling_params = SamplingParams(
            **({"temperature": 0.0, "max_tokens": max_new_tokens})
        )
        return cls(
            model=llm,
            processor=processor,
            sampling_params=sampling_params,
        )

    async def transcribe(
        self,
        audio: AudioLike,
        context: str = "",
        language: Optional[str] = None,
    ) -> ASRTranscription:
        force_lang = self._normalize_force_language(language)
        raw_text_parts: List[str] = []
        async for chunk in self.transcribe_stream(
            audio=audio,
            context=context,
            language=force_lang,
        ):
            raw_text_parts.append(chunk)
        lang, text = parse_asr_output("".join(raw_text_parts), user_language=force_lang)
        return ASRTranscription(language=lang, text=text)

    async def transcribe_stream(
        self,
        audio: AudioLike,
        context: str = "",
        language: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        对单条音频进行流式转写，逐步产出模型生成的文本片段。

        说明：
          - 输入音频仅按单块处理，不再做内部切块。
          - 输出为模型原始生成的增量文本片段；可通过拼接所有片段得到完整 raw 输出。
          - 若希望得到最终 language/text，可将所有片段拼接后使用 parse_asr_output 解析，
            或直接调用 transcribe()。
        """
        wav = normalize_audio_input(audio)
        force_lang = self._normalize_force_language(language)
        prompt = self._build_text_prompt(context=context, force_language=force_lang)
        inp = {"prompt": prompt, "multi_modal_data": {"audio": [wav]}}
        async for chunk in self._stream_generate_single(inp):
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
        Build the string prompt for one request.

        If force_language is provided, "language X<asr_text>" is appended after the generation prompt
        to request text-only output.
        """
        msgs = self._build_messages(context=context, audio_payload="")
        base = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        if force_language:
            base = base + f"language {force_language}{'<asr_text>'}"
        return base

    async def _abort_request(self, request_id: str) -> None:
        abort = getattr(self.model, "abort", None)
        if abort is None:
            return
        try:
            result = abort(request_id)
        except TypeError:
            result = abort(request_id=request_id)
        if inspect.isawaitable(result):
            await result

    async def _stream_generate_single(
        self, inp: Dict[str, Any]
    ) -> AsyncIterator[str]:
        request_id = uuid4().hex
        sampling_params = (
            self.sampling_params.clone()
            if hasattr(self.sampling_params, "clone")
            else self.sampling_params
        )
        sampling_params.output_kind = RequestOutputKind.DELTA
        cumulative_text = ""
        finished = False
        try:
            async for out in self.model.generate(
                inp,
                sampling_params,
                request_id=request_id,
            ):
                if not out.outputs:
                    continue

                cur_text = out.outputs[0].text or ""
                if not cur_text:
                    continue

                if cur_text.startswith(cumulative_text):
                    delta_text = cur_text[len(cumulative_text) :]
                    cumulative_text = cur_text
                else:
                    delta_text = cur_text
                    cumulative_text = cumulative_text + cur_text

                if delta_text:
                    yield delta_text
            finished = True
        except asyncio.CancelledError:
            logger.warning(
                "Streaming inference interrupted, aborting request %s", request_id
            )
            raise
        except Exception:
            logger.warning(
                "Streaming inference failed, aborting request %s", request_id
            )
            raise
        finally:
            if not finished:
                await self._abort_request(request_id)
