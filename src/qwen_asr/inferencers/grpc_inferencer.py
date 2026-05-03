from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Tuple

import numpy as np

from qwen_asr.inferencers.base import ASRInferencer
from qwen_asr.inferencers.text.transcript_parser import StreamingTranscriptParser

logger = logging.getLogger(__name__)

GrpcInferenceResult = Tuple[str, str, bool]


class GrpcInferencer:
    """
    gRPC 推理适配器。

    该类只处理普通 Python 数据，不依赖 gRPC context、状态码或 proto 对象。
    """

    def __init__(self, inferencer: ASRInferencer) -> None:
        self.inferencer = inferencer

    async def infer(
        self,
        *,
        audio_bytes: bytes,
        sample_rate: int,
        language_code: str,
        interim_results: bool,
        context: str = "",
    ) -> AsyncIterator[GrpcInferenceResult]:
        """解码音频字节并流式返回转写结果。"""
        audio = self.decode_audio(audio_bytes)
        logger.info(
            "Received audio content: %d bytes (≈%.1fs at %dHz), language=%s",
            len(audio_bytes),
            len(audio) / sample_rate,
            sample_rate,
            language_code,
        )

        transcript_parser = StreamingTranscriptParser(language_code=language_code)
        # 启动推理，流式输出转写文本
        async for chunk in self.inferencer.transcribe_stream(
            audio=audio,
            sample_rate=sample_rate,
            context=context,
            language=language_code,
        ):
            # 将每个增量片段推入转写解析器，获取当前完整转写、增量文本和是否有更新。
            transcript, delta, changed = transcript_parser.push(chunk)
            # 如果启用中间结果且文本有更新，先返回增量片段；最后返回完整转写并标记结束。
            if interim_results and changed:
                yield transcript, delta, False
        # 最后返回完整转写并标记结束
        yield transcript_parser.transcript, "", True

    @staticmethod
    def decode_audio(audio_bytes: bytes) -> np.ndarray:
        """将 PCM16/WAV 音频字节解码为 float32 数组。"""
        if audio_bytes.startswith(b"RIFF"):
            audio_bytes = audio_bytes[44:]
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
        return audio
