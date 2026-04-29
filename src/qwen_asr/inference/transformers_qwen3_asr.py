# coding=utf-8
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import Thread
from typing import Any, AsyncIterator, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    TextIteratorStreamer,
)

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from qwen_asr.inference.utils import AudioLike, normalize_audio_input, parse_asr_output, resolve_language_code

AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)
AutoModelForCausalLM.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)


@dataclass
class ASRTranscription:
    language: str
    text: str


class TransformersQwen3ASRModel:
    def __init__(self, model, processor, max_new_tokens: int = 4096):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    @classmethod
    def LLM(cls, model: str, max_new_tokens: int = 4096, device: str | None = None, **kwargs):
        torch_device = "cpu" if device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch_device == "cuda" else torch.float32
        processor = Qwen3ASRProcessor.from_pretrained(model, fix_mistral_regex=True)
        m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype, trust_remote_code=False, **kwargs).to(torch_device).eval()
        return cls(m, processor, max_new_tokens=max_new_tokens)

    async def transcribe(self, audio: AudioLike, context: str = "", language: Optional[str] = None) -> ASRTranscription:
        chunks = []
        async for c in self.transcribe_stream(audio, context, language):
            chunks.append(c)
        lang, text = parse_asr_output("".join(chunks), user_language=resolve_language_code(language))
        return ASRTranscription(language=lang, text=text)

    async def transcribe_stream(self, audio: AudioLike, context: str = "", language: Optional[str] = None) -> AsyncIterator[str]:
        wav = normalize_audio_input(audio)
        msgs = [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        force_lang = resolve_language_code(language)
        if force_lang:
            prompt += f"language {force_lang}<asr_text>"

        inputs = self.processor(text=[prompt], audio=[wav], return_tensors="pt")
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_error: list[BaseException] = []
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | object] = asyncio.Queue()
        sentinel = object()

        def run_generation() -> None:
            try:
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        streamer=streamer,
                    )
            except BaseException as exc:
                generation_error.append(exc)
                end = getattr(streamer, "end", None)
                if callable(end):
                    end()

        def consume_streamer() -> None:
            try:
                for text in streamer:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
            except BaseException as exc:
                generation_error.append(exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        generation_thread = Thread(target=run_generation, daemon=True)
        streamer_thread = Thread(target=consume_streamer, daemon=True)
        streamer_thread.start()
        generation_thread.start()

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if item:
                yield str(item)

        generation_thread.join()
        streamer_thread.join()

        if generation_error:
            raise RuntimeError("Transformers streaming generation failed") from generation_error[0]

    @staticmethod
    def _extract_generated_sequences(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output

        sequences = getattr(output, "sequences", None)
        if isinstance(sequences, torch.Tensor):
            return sequences

        raise TypeError(
            "Unsupported generate() return type: "
            f"{type(output).__name__}. Expected a tensor or an object with a "
            "'sequences' tensor attribute."
        )
