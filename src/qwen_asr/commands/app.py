# coding=utf-8
"""
Typer CLI entry-point for the Qwen3-ASR gRPC server.

All parameters can be provided via command-line options **or** environment
variables (prefixed with ``QWEN_ASR_``).

Example:
    QWEN_ASR_MODEL=/models/qwen3-asr QWEN_ASR_PORT=50051 python -m qwen_asr.commands.app serve
"""
from __future__ import annotations

import logging
from concurrent import futures
from typing import Optional

import grpc
import typer

from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
from qwen_asr.protos.asr.ux_speech_pb2_grpc import add_UxSpeechServicer_to_server
from qwen_asr.servicer.servicer import ASRServicer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="qwen-asr",
    help="Qwen3-ASR gRPC inference server.",
    add_completion=False,
)


@app.command()
def serve(
    model: str = typer.Option(
        ...,
        help="Path or HuggingFace repo of the Qwen3-ASR model.",
        envvar="MODEL_PATH",
    ),
    port: int = typer.Option(
        50051,
        help="gRPC listening port.",
        envvar="GRPC_PORT",
    ),
    max_new_tokens: int = typer.Option(
        4096,
        help="Maximum number of tokens to generate per request.",
        envvar="MAX_NEW_TOKENS",
    ),
    gpu_memory_utilization: float = typer.Option(
        0.9,
        help="Fraction of GPU memory to use (0.0 - 1.0).",
        envvar="GPU_MEMORY_UTILIZATION",
    ),
    max_model_len: int = typer.Option(
        4096,
        help="Maximum model context length in tokens.",
        envvar="MAX_MODEL_LEN",
    ),
    max_workers: int = typer.Option(
        10,
        help="gRPC thread pool size.",
        envvar="MAX_WORKERS",
    ),
) -> None:
    """Start the Qwen3-ASR gRPC server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )

    logger.info("Loading model from %s …", model)
    asr_model = Qwen3ASRModel.LLM(
        model=model,
        max_new_tokens=max_new_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    logger.info("Model loaded successfully.")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    add_UxSpeechServicer_to_server(ASRServicer(asr_model), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC server listening on port %d", port)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down …")
        server.stop(grace=5)


if __name__ == "__main__":
    app()
