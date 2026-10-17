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

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator
from typing import Any, Dict, Optional
from uuid import uuid4

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind

from qwen_asr.models.transformers.processing_qwen3_asr import Qwen3ASRProcessor
from qwen_asr.engines.vllm.utils import filter_async_engine_kwargs

logger = logging.getLogger(__name__)


class VLLMEngine:
    """持有 vLLM 引擎与采样参数，负责生成、取消和关闭。"""

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
    def load(
        cls,
        model: str,
        max_new_tokens: Optional[int] = 4096,
        **kwargs,
    ) -> "VLLMEngine":
        # 只透传 AsyncEngineArgs 支持的 vLLM 参数。
        engine_kwargs = filter_async_engine_kwargs(AsyncEngineArgs, kwargs)
        llm = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(model=model, **engine_kwargs)
        )
        # processor 负责构造 Qwen3-ASR 的多模态输入。
        processor = Qwen3ASRProcessor.from_pretrained(model, fix_mistral_regex=True)
        sampling_params = SamplingParams(
            **({"temperature": 0.0, "max_tokens": max_new_tokens})
        )
        return cls(
            model=llm,
            processor=processor,
            sampling_params=sampling_params,
        )

    async def aclose(self) -> None:
        shutdown = getattr(self.model, "shutdown", None)
        if shutdown is None:
            shutdown = getattr(self.model, "shutdown_background_loop", None)
        if shutdown is not None:
            result = shutdown()
            if inspect.isawaitable(result):
                await result

    @property
    def errored(self) -> bool:
        return bool(getattr(self.model, "errored", False))

    async def _abort_request(self, request_id: str) -> None:
        # 兼容不同 vLLM 版本的 abort 调用签名。
        abort = getattr(self.model, "abort", None)
        if abort is None:
            return
        try:
            result = abort(request_id)
        except TypeError:
            result = abort(request_id=request_id)
        if inspect.isawaitable(result):
            await result

    async def generate_stream(
        self, inp: Dict[str, Any]
    ) -> AsyncIterator[str]:
        # 每个请求使用独立 ID，方便取消时精确 abort。
        request_id = uuid4().hex
        sampling_params = (
            self.sampling_params.clone()
            if hasattr(self.sampling_params, "clone")
            else self.sampling_params
        )
        sampling_params.output_kind = RequestOutputKind.DELTA
        finished = False
        try:
            # DELTA 模式保证每次只返回新增文本，避免在长请求中保留累计副本。
            async for out in self.model.generate(
                inp,
                sampling_params,
                request_id=request_id,
            ):
                if not out.outputs:
                    continue

                delta_text = out.outputs[0].text or ""
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
