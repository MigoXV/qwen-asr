from __future__ import annotations

import itertools
import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple


class FakeSamplingParams:
    def __init__(self, output_kind=None) -> None:
        self.output_kind = output_kind

    def clone(self) -> "FakeSamplingParams":
        return FakeSamplingParams(output_kind=self.output_kind)


@dataclass
class FakeCompletionOutput:
    text: str


class FakeRequestOutput:
    def __init__(self, request_id: str, text: str, finished: bool) -> None:
        self.request_id = request_id
        self.outputs = [FakeCompletionOutput(text=text)]
        self.finished = finished


class FakeLLMEngine:
    def __init__(
        self,
        script_factory: Optional[Callable[[dict], Iterable[Tuple[str, bool]]]] = None,
        step_delay: float = 0.0,
    ) -> None:
        self.script_factory = script_factory or (lambda inp: inp["script"])
        self.step_delay = step_delay
        self.raise_on_generate: Optional[BaseException] = None
        self.aborted_ids: List[str] = []
        self._active: Dict[str, List[Tuple[str, bool]]] = {}
        self._lock = asyncio.Lock()

    async def add_request(self, request_id: str, inp: dict) -> None:
        async with self._lock:
            self._active[request_id] = list(self.script_factory(inp))

    async def abort_request(self, request_id: str) -> None:
        async with self._lock:
            self.aborted_ids.append(str(request_id))
            self._active.pop(str(request_id), None)

    async def generate(self, request_id: str, inp: dict):
        if self.raise_on_generate is not None:
            raise self.raise_on_generate

        await self.add_request(request_id, inp)
        while True:
            if self.step_delay:
                await asyncio.sleep(self.step_delay)

            async with self._lock:
                script = self._active.get(request_id)
                if script is None:
                    return
                if not script:
                    self._active.pop(request_id, None)
                    return
                text, finished = script.pop(0)
                if finished or not script:
                    self._active.pop(request_id, None)

            yield FakeRequestOutput(request_id, text, finished)
            if finished:
                return


class FakeLLM:
    def __init__(
        self,
        script_factory: Optional[Callable[[dict], Iterable[Tuple[str, bool]]]] = None,
        step_delay: float = 0.0,
    ) -> None:
        self.llm_engine = FakeLLMEngine(
            script_factory=script_factory,
            step_delay=step_delay,
        )
        self._counter = itertools.count()

    def _next_request_id(self) -> str:
        return f"engine-{next(self._counter)}"

    async def generate(
        self,
        inp,
        sampling_params,
        request_id=None,
        lora_request=None,
        priority=0,
    ):
        del sampling_params, lora_request, priority
        engine_request_id = str(request_id or self._next_request_id())
        async for output in self.llm_engine.generate(engine_request_id, inp):
            yield output

    async def abort(self, request_id: str):
        await self.llm_engine.abort_request(request_id)


class FakeModel:
    def __init__(self, llm: FakeLLM) -> None:
        self.model = llm
        self.sampling_params = FakeSamplingParams()
        self._counter = itertools.count()
        self.seen_audios = []

    @staticmethod
    def _normalize_force_language(language):
        return language

    @staticmethod
    def _build_text_prompt(context: str, force_language: Optional[str]) -> str:
        suffix = force_language or "auto"
        return f"{context}|{suffix}"

    async def transcribe_stream(
        self,
        audio,
        context: str = "",
        language: Optional[str] = None,
    ):
        request_id = f"request-{next(self._counter)}"
        prompt = self._build_text_prompt(context, self._normalize_force_language(language))
        self.seen_audios.append(audio)
        cumulative_text = ""
        finished = False
        try:
            async for output in self.model.generate(
                {"prompt": prompt, "multi_modal_data": {"audio": [audio]}},
                self.sampling_params.clone(),
                request_id=request_id,
            ):
                current_text = output.outputs[0].text or ""
                if current_text.startswith(cumulative_text):
                    delta_text = current_text[len(cumulative_text) :]
                    cumulative_text = current_text
                else:
                    delta_text = current_text
                    cumulative_text = cumulative_text + current_text
                if delta_text:
                    yield delta_text
            finished = True
        finally:
            if not finished:
                await self.model.abort(request_id)


class FakeAbortError(RuntimeError):
    def __init__(self, code, details: str) -> None:
        super().__init__(f"{code}: {details}")
        self.code = code
        self.details = details


class FakeContext:
    def __init__(self, active: bool = True) -> None:
        self._active = active
        self.aborted = None

    def is_active(self) -> bool:
        return self._active

    def set_active(self, active: bool) -> None:
        self._active = active

    async def abort(self, code, details: str):
        self.aborted = (code, details)
        raise FakeAbortError(code, details)
