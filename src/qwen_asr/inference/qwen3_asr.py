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
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
from transformers import AutoConfig, AutoProcessor
from vllm import LLM as vLLM
from vllm import ModelRegistry, SamplingParams
from vllm.sampling_params import RequestOutputKind

from qwen_asr.core.vllm_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from .utils import (
    MAX_ASR_INPUT_SECONDS,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    AudioChunk,
    AudioLike,
    chunk_list,
    merge_languages,
    normalize_audio_input,
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)

AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)

ModelRegistry.register_model(
    "Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration
)


@dataclass
class ASRTranscription:
    """
    One transcription result.

    Attributes:
        language (str):
            Merged language string for the sample, e.g. "Chinese" or "Chinese,English".
            Empty string if unknown or silent audio.
        text (str):
            Transcribed text.
    """

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
        max_inference_batch_size: int = -1,
    ):
        self.model = model
        self.processor = processor
        self.sampling_params = sampling_params
        self.max_inference_batch_size = int(max_inference_batch_size)

    @classmethod
    def LLM(
        cls,
        model: str,
        max_inference_batch_size: int = -1,
        max_new_tokens: Optional[int] = 4096,
        **kwargs,
    ) -> "Qwen3ASRModel":
        """
        Initialize using vLLM backend.

        Args:
            model:
                Model path/repo for vLLM.
            max_inference_batch_size:
                Batch size limit for inference. -1 means no chunking. Small values can avoid OOM.
            max_new_tokens:
                Maximum number of tokens to generate.
            **kwargs:
                Forwarded to vllm.LLM(...).

        Returns:
            Qwen3ASRModel
        """
        llm = vLLM(model=model, **kwargs)

        processor = Qwen3ASRProcessor.from_pretrained(model, fix_mistral_regex=True)
        sampling_params = SamplingParams(
            **({"temperature": 0.0, "max_tokens": max_new_tokens})
        )

        return cls(
            model=llm,
            processor=processor,
            sampling_params=sampling_params,
            max_inference_batch_size=max_inference_batch_size,
        )

    def get_supported_languages(self) -> List[str]:
        """
        Returns the supported language list.

        Returns:
            List[str]: Canonical language names.
        """
        return list(SUPPORTED_LANGUAGES)

    def transcribe(
        self,
        audio: AudioLike,
        context: str = "",
        language: Optional[str] = None,
    ) -> ASRTranscription:
        """
        对单条音频进行语音识别。

        Args:
            audio:
                音频输入，支持以下格式：
                  - str: 本地文件路径 / URL / base64 数据 URL
                  - (np.ndarray, sr): numpy 数组与采样率的元组
            context:
                上下文字符串（可选）。
            language:
                指定语言（可选）。若提供，必须是支持的语言之一。
                若提供，提示词将强制输出纯转录文本。

        Returns:
            ASRTranscription: 识别结果，包含 language 和 text 字段。

        Raises:
            ValueError:
                - 若指定语言不受支持。
        """
        force_lang = self._normalize_force_language(language)
        raw_text = "".join(
            self.transcribe_stream(audio=audio, context=context, language=force_lang)
        )
        lang, txt = parse_asr_output(raw_text, user_language=force_lang)
        return ASRTranscription(language=lang, text=txt)

    def transcribe_stream(
        self,
        audio: AudioLike,
        context: str = "",
        language: Optional[str] = None,
    ) -> Iterator[str]:
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
        yield from self._stream_generate_single(inp)

    def transcribe_batch(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
    ) -> List[ASRTranscription]:
        """
        对音频进行语音识别（支持批量）。

        Args:
            audio:
                音频输入，支持以下格式：
                  - str: 本地文件路径 / URL / base64 数据 URL
                  - (np.ndarray, sr): numpy 数组与采样率的元组
                  - 以上格式的列表（批量输入）
            context:
                上下文字符串（可选）。若为标量，将广播至整个批次。
            language:
                指定语言（可选）。若提供，必须是支持的语言之一。
                若为标量，将广播至整个批次。
                若提供，提示词将强制输出纯转录文本。
        Returns:
            List[ASRTranscription]: 每条音频对应一个识别结果。

        Raises:
            ValueError:
                - 若指定语言不受支持。
                - 若 context/language 的批次大小与 audio 不匹配。
        """
        # 将音频输入统一归一化为波形列表
        wavs = normalize_audios(audio)
        n = len(wavs)

        # 将 context 统一处理为列表，并广播到批次大小
        ctxs = context if isinstance(context, list) else [context]
        if len(ctxs) == 1 and n > 1:
            ctxs = ctxs * n
        if len(ctxs) != n:
            raise ValueError(f"Batch size mismatch: audio={n}, context={len(ctxs)}")

        # 将 language 统一处理为列表，并广播到批次大小
        langs_in: List[Optional[str]]
        if language is None:
            langs_in = [None] * n
        else:
            langs_in = language if isinstance(language, list) else [language]
            if len(langs_in) == 1 and n > 1:
                langs_in = langs_in * n
            if len(langs_in) != n:
                raise ValueError(
                    f"Batch size mismatch: audio={n}, language={len(langs_in)}"
                )

        # 规范化语言名称并校验合法性
        langs_norm: List[Optional[str]] = []
        for l in langs_in:
            if l is None or str(l).strip() == "":
                langs_norm.append(None)
            else:
                ln = normalize_language_name(str(l))
                validate_language(ln)
                langs_norm.append(ln)

        max_chunk_sec = MAX_ASR_INPUT_SECONDS

        # 将每条音频按最大时长切分为若干块，并记录映射关系
        chunks: List[AudioChunk] = []
        for i, wav in enumerate(wavs):
            parts = split_audio_into_chunks(
                wav=wav,
                sr=SAMPLE_RATE,
                max_chunk_sec=max_chunk_sec,
            )
            for j, (cwav, offset_sec) in enumerate(parts):
                chunks.append(
                    AudioChunk(
                        orig_index=i,
                        chunk_index=j,
                        wav=cwav,
                        sr=SAMPLE_RATE,
                        offset_sec=offset_sec,
                    )
                )

        # 对所有音频块执行 ASR 推理
        chunk_ctx: List[str] = [ctxs[c.orig_index] for c in chunks]
        chunk_lang: List[Optional[str]] = [langs_norm[c.orig_index] for c in chunks]
        chunk_wavs: List[np.ndarray] = [c.wav for c in chunks]
        raw_outputs = self._infer_asr(chunk_ctx, chunk_wavs, chunk_lang)

        # 解析每个块的原始输出，提取语言标签和转录文本
        per_chunk_lang: List[str] = []
        per_chunk_text: List[str] = []
        for out, forced_lang in zip(raw_outputs, chunk_lang):
            lang, txt = parse_asr_output(out, user_language=forced_lang)
            per_chunk_lang.append(lang)
            per_chunk_text.append(txt)

        # 将各块结果按原始音频索引汇总
        out_langs: List[List[str]] = [[] for _ in range(n)]
        out_texts: List[List[str]] = [[] for _ in range(n)]

        for c, lang, txt in zip(chunks, per_chunk_lang, per_chunk_text):
            out_langs[c.orig_index].append(lang)
            out_texts[c.orig_index].append(txt)

        # 合并各块文本与语言标签，生成最终结果
        results: List[ASRTranscription] = []
        for i in range(n):
            merged_text = "".join([t for t in out_texts[i] if t is not None])
            merged_language = merge_languages(out_langs[i])
            results.append(ASRTranscription(language=merged_language, text=merged_text))

        return results

    def _build_messages(self, context: str, audio_payload: Any) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    def _normalize_force_language(self, language: Optional[str]) -> Optional[str]:
        if language is None or str(language).strip() == "":
            return None
        force_lang = normalize_language_name(str(language))
        validate_language(force_lang)
        return force_lang

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

    def _infer_asr(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        """
        Run backend inference for chunk-level items.

        Args:
            contexts: List of context strings.
            wavs: List of mono waveforms (np.ndarray).
            languages: List of forced languages or None.

        Returns:
            List[str]: Raw decoded strings (one per chunk).
        """
        return self._infer_asr_vllm(contexts, wavs, languages)

    def _infer_asr_vllm(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        inputs: List[Dict[str, Any]] = []
        for c, w, fl in zip(contexts, wavs, languages):
            prompt = self._build_text_prompt(context=c, force_language=fl)
            inputs.append({"prompt": prompt, "multi_modal_data": {"audio": [w]}})

        outs: List[str] = []
        for batch in chunk_list(inputs, self.max_inference_batch_size):
            outputs = self.model.generate(
                batch, sampling_params=self.sampling_params, use_tqdm=False
            )
            for o in outputs:
                outs.append(o.outputs[0].text)
        return outs

    def _stream_generate_single(self, inp: Dict[str, Any]) -> Iterator[str]:
        """
        Stream one vLLM request.

        Prefers llm_engine.step() for incremental output. If the backend does not expose
        engine internals, it falls back to one-shot generate() and yields a single chunk.
        """
        add_request = getattr(self.model, "_add_request", None)
        llm_engine = getattr(self.model, "llm_engine", None)

        if add_request is None or llm_engine is None:
            outputs = self.model.generate(
                [inp], sampling_params=self.sampling_params, use_tqdm=False
            )
            text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            if text:
                yield text
            return

        sampling_params = (
            self.sampling_params.clone()
            if hasattr(self.sampling_params, "clone")
            else self.sampling_params
        )
        sampling_params.output_kind = RequestOutputKind.DELTA

        if llm_engine.has_unfinished_requests():
            raise RuntimeError(
                "Streaming transcribe does not support concurrent unfinished requests."
            )

        request_id = add_request(inp, sampling_params, lora_request=None, priority=0)

        cumulative_text = ""
        try:
            while llm_engine.has_unfinished_requests():
                step_outputs = llm_engine.step()
                for out in step_outputs:
                    if not out.outputs:
                        continue

                    cur_text = out.outputs[0].text or ""
                    if not cur_text:
                        continue

                    # DELTA mode should already be incremental; this also keeps compatibility
                    # if the backend returns cumulative text for any reason.
                    if cur_text.startswith(cumulative_text):
                        delta_text = cur_text[len(cumulative_text) :]
                        cumulative_text = cur_text
                    else:
                        delta_text = cur_text
                        cumulative_text = cumulative_text + cur_text

                    if delta_text:
                        yield delta_text
        except Exception:
            try:
                llm_engine.abort_request([request_id], internal=True)
            except Exception:
                pass
            raise
