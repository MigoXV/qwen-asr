# coding=utf-8
"""
gRPC Servicer for Qwen3-ASR: offline single-request inference with streaming response.

Flow:
  1. Collect all audio bytes from the client stream (first message = config, rest = audio).
  2. Auto-detect RIFF/WAVE header; strip it if present, then decode as LINEAR16 PCM.
     Resample to 16 kHz if needed using config.sample_rate_hertz.
  3. Run Qwen3ASRModel.transcribe_stream() for incremental text generation.
  4. Yield StreamingRecognizeResponse messages back to the client.
"""
from __future__ import annotations

import logging
import struct
from typing import Iterator

import grpc
import librosa
import numpy as np

from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
from qwen_asr.inference.utils import SAMPLE_RATE
from qwen_asr.protos.asr.ux_speech_pb2 import (
    SpeechRecognitionAlternative,
    StreamingRecognitionResult,
    StreamingRecognizeResponse,
)
from qwen_asr.protos.asr.ux_speech_pb2_grpc import UxSpeechServicer

logger = logging.getLogger(__name__)


class ASRServicer(UxSpeechServicer):
    """
    Offline single-audio inference with streaming text response.

    The client sends a config message followed by audio chunks.
    After all audio is received, the server runs inference once
    and streams back recognition results as they are generated.
    """

    def __init__(self, model: Qwen3ASRModel) -> None:
        super().__init__()
        self.model = model

    # ------------------------------------------------------------------
    # gRPC handler
    # ------------------------------------------------------------------
    def StreamingRecognize(
        self,
        request_iterator,
        context: grpc.ServicerContext,
    ) -> Iterator[StreamingRecognizeResponse]:
        """Handle a single StreamingRecognize RPC call."""

        # ── Phase 1: collect config + audio from the client stream ────
        language: str | None = None
        sample_rate: int = SAMPLE_RATE
        interim_results: bool = True
        config_received: bool = False
        audio_buf = bytearray()

        for request in request_iterator:
            which = request.WhichOneof("streaming_request")

            if which == "streaming_config":
                if config_received:
                    logger.error("Duplicate streaming_config received, aborting RPC.")
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "streaming_config must only be sent once as the first message.",
                    )
                    return
                config_received = True
                streaming_config = request.streaming_config
                interim_results = streaming_config.interim_results

                cfg = streaming_config.config
                if cfg.sample_rate_hertz > 0:
                    sample_rate = cfg.sample_rate_hertz
                if cfg.language_code:
                    language = cfg.language_code

                logger.info(
                    "Config received: language_code=%s, sample_rate=%d, interim_results=%s",
                    language, sample_rate, interim_results,
                )

            elif which == "audio_content":
                if not config_received:
                    logger.error("Audio received before streaming_config, aborting RPC.")
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "First message must contain streaming_config.",
                    )
                    return
                audio_buf.extend(request.audio_content)

            else:
                # Empty oneof – skip
                continue

        if not config_received:
            logger.error("No streaming_config received in request stream.")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "No streaming_config received.",
            )
            return

        audio_duration_sec = len(audio_buf) / 2 / max(sample_rate, 1)
        logger.info(
            "Audio collected: %d bytes (≈%.1fs at %dHz)",
            len(audio_buf), audio_duration_sec, sample_rate,
        )

        if len(audio_buf) == 0:
            logger.info("Empty audio, returning blank result.")
            # No audio data – return an empty final result
            yield self._make_response("", is_final=True)
            return

        # ── Phase 2: bytes → numpy float32 waveform ──────────────────
        try:
            wav = self._decode_audio_bytes(audio_buf, sample_rate)
        except Exception as exc:
            logger.error("Failed to decode audio: %s", exc, exc_info=True)
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Failed to decode audio: {exc}",
            )
            return

        # ── Phase 3 & 4: stream inference → stream responses ─────────
        try:
            logger.info("Starting inference, language=%s", language)
            text_iter = self.model.transcribe_stream(
                audio=(wav, SAMPLE_RATE),
                language=language,
            )

            if interim_results:
                # Yield each incremental chunk as an interim result,
                # then yield the full accumulated text as final.
                accumulated = ""
                for delta in text_iter:
                    accumulated += delta
                    yield self._make_response(accumulated, is_final=False)
                yield self._make_response(accumulated, is_final=True)
                logger.info("Transcription done: %s", accumulated)
            else:
                # Collect everything, return once.
                full_text = "".join(text_iter)
                yield self._make_response(full_text, is_final=True)
                logger.info("Transcription done: %s", full_text)

        except Exception as exc:
            logger.error("Inference failed: %s", exc, exc_info=True)
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Inference error: {exc}",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_wav_header(buf: bytearray) -> bytearray:
        """
        If *buf* starts with a RIFF/WAVE header, skip past it and return
        only the raw PCM payload.  Otherwise return *buf* unchanged.

        WAV header layout (canonical):
          bytes  0-3   "RIFF"
          bytes  4-7   file size (LE uint32)
          bytes  8-11  "WAVE"
          then a series of chunks: 4-byte id + 4-byte size (LE uint32) + data
          We skip all chunks until we find "data", then return everything after
          that chunk header.
        """
        if not (len(buf) >= 12 and buf[0:4] == b"RIFF" and buf[8:12] == b"WAVE"):
            return buf  # not a WAV file

        offset = 12
        while offset + 8 <= len(buf):
            chunk_id = bytes(buf[offset:offset + 4])
            chunk_size = struct.unpack_from("<I", buf, offset + 4)[0]
            offset += 8
            if chunk_id == b"data":
                logger.debug("WAV header stripped, PCM payload offset=%d", offset)
                return buf[offset:offset + chunk_size]
            offset += chunk_size

        raise ValueError("Malformed WAV: 'data' chunk not found")

    @staticmethod
    def _decode_audio_bytes(
        buf: bytearray,
        config_sample_rate: int,
    ) -> np.ndarray:
        """
        Decode raw audio bytes to a mono float32 waveform at 16 kHz.

        Detection logic:
          - If *buf* starts with a RIFF/WAVE header, strip the header and
            treat the payload as LINEAR16 PCM.
          - Otherwise treat *buf* directly as raw LINEAR16 (signed 16-bit LE PCM).

        In both cases ``config_sample_rate`` (from RecognitionConfig.sample_rate_hertz)
        is used as the sample-rate, and the output is resampled to 16 kHz if needed.
        """
        pcm_buf = ASRServicer._strip_wav_header(buf)
        audio = np.frombuffer(pcm_buf, dtype=np.int16).astype(np.float32) / 32768.0
        logger.debug("PCM samples=%d, config_sr=%d", len(audio), config_sample_rate)

        if config_sample_rate != SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=config_sample_rate, target_sr=SAMPLE_RATE
            ).astype(np.float32)

        return audio

    @staticmethod
    def _make_response(
        transcript: str,
        is_final: bool,
    ) -> StreamingRecognizeResponse:
        """Build a ``StreamingRecognizeResponse`` with one result."""
        return StreamingRecognizeResponse(
            results=[
                StreamingRecognitionResult(
                    alternative=SpeechRecognitionAlternative(
                        transcript=transcript,
                    ),
                    is_final=is_final,
                )
            ]
        )
