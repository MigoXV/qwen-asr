# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

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
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        yield text
