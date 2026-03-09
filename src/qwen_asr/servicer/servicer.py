from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import grpc
import numpy as np

from qwen_asr.protos.asr.ux_speech_pb2 import (
    SpeechRecognitionAlternative,
    StreamingRecognizeResponse,
    StreamingRecognizeRequest,
    StreamingRecognitionResult,
    WordInfo,
)
from google.protobuf.duration_pb2 import Duration
from qwen_asr.protos.asr.ux_speech_pb2_grpc import UxSpeechServicer

if TYPE_CHECKING:
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

logger = logging.getLogger(__name__)


class ASRServicer(UxSpeechServicer):
    """
    Treat the streaming RPC as a single-shot offline ASR request.

    The client still uses the streaming proto, but the server only reads the
    first audio chunk after the config message and treats it as the complete
    audio payload. One final response is returned and the RPC ends.
    """

    def __init__(self, model: Qwen3ASRModel) -> None:
        super().__init__()
        self.model = model

    def close(self) -> None:
        return None

    async def StreamingRecognize(
        self,
        request_iterator: AsyncIterator[StreamingRecognizeRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[StreamingRecognizeResponse]:
        """Handle a single StreamingRecognize RPC call as one-shot ASR."""
        config_request = await self._anext_or_none(request_iterator)
        if (
            config_request is None
            or config_request.WhichOneof("streaming_request") != "streaming_config"
        ):
            logger.error("First message must contain streaming_config, aborting RPC.")
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "First message must contain streaming_config.",
            )
            return

        streaming_config = config_request.streaming_config
        interim_results = streaming_config.interim_results
        config = streaming_config.config
        sample_rate = config.sample_rate_hertz
        language_code = config.language_code

        audio_request = await self._anext_or_none(request_iterator)
        if (
            audio_request is None
            or audio_request.WhichOneof("streaming_request") != "audio_content"
        ):
            logger.error("Second message must contain audio_content, aborting RPC.")
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Second message must contain audio_content.",
            )
            return

        extra_request = await self._anext_or_none(request_iterator)
        if extra_request is not None:
            logger.error("Only one audio_content message is supported, aborting RPC.")
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Only a single audio_content message is supported.",
            )
            return

        try:
            audio_bytes = audio_request.audio_content
            if audio_bytes.startswith(b"RIFF"):
                audio_bytes = audio_bytes[44:]
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio /= 32768.0
        except Exception as exc:
            logger.error("Failed to decode audio_content: %s", exc, exc_info=True)
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Failed to decode audio_content: {exc}",
            )
            return

        logger.info(
            "Received audio content: %d bytes (≈%.1fs at %dHz), starting inference, language=%s, interim_results=%s",
            len(audio_bytes),
            len(audio) / sample_rate,
            sample_rate,
            language_code,
            interim_results,
        )

        final_result = ""
        try:
            async for chunk in self.model.transcribe_stream(
                audio=(audio, sample_rate),
                language=language_code,
            ):
                if not self._context_is_active(context):
                    logger.info("Client disconnected; stopping response stream.")
                    break
                final_result += chunk
                if interim_results:
                    yield self._make_response(final_result, is_final=False, word=chunk)
            if self._context_is_active(context):
                yield self._make_response(final_result, is_final=True)
        except asyncio.CancelledError:
            logger.info("StreamingRecognize cancelled by client.")
            return
        except Exception as exc:
            logger.error("Inference failed: %s", exc, exc_info=True)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Inference failed: {exc}",
            )
            return

    @staticmethod
    async def _anext_or_none(
        request_iterator: AsyncIterator[StreamingRecognizeRequest],
    ) -> StreamingRecognizeRequest | None:
        try:
            return await anext(request_iterator)
        except StopAsyncIteration:
            return None

    @staticmethod
    def _context_is_active(context: grpc.aio.ServicerContext) -> bool:
        cancelled = getattr(context, "cancelled", None)
        if callable(cancelled):
            try:
                return not bool(cancelled())
            except Exception:
                logger.debug("Failed to query context.cancelled()", exc_info=True)

        is_active = getattr(context, "is_active", None)
        if callable(is_active):
            try:
                return bool(is_active())
            except Exception:
                logger.debug("Failed to query context.is_active()", exc_info=True)

        return True

    @staticmethod
    def _make_response(
        transcript: str, is_final: bool, word: str = ""
    ) -> StreamingRecognizeResponse:
        """Build a ``StreamingRecognizeResponse`` with one result."""
        return StreamingRecognizeResponse(
            results=[
                StreamingRecognitionResult(
                    alternative=SpeechRecognitionAlternative(
                        transcript=transcript,
                        words=[
                            WordInfo(
                                word=word,
                                start_time=Duration(seconds=0, nanos=0),
                                end_time=Duration(seconds=0, nanos=0),
                            )
                        ],
                    ),
                    is_final=is_final,
                )
            ]
        )
