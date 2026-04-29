# coding=utf-8
"""
Typer CLI entry-point for the Qwen3-ASR gRPC server.

All parameters can be provided via command-line options **or** environment
variables (prefixed with ``QWEN_ASR_``).

Example:
    QWEN_ASR_MODEL=/models/qwen3-asr QWEN_ASR_PORT=50051 python -m qwen_asr.commands.app serve
"""
from __future__ import annotations

import asyncio
import logging

import grpc
import typer

from qwen_asr.commands.utils import (
    BACKEND_TRANSFORMERS,
    BACKEND_VLLM,
    DEVICE_AUTO,
    DEVICE_CPU,
    DeviceSelectionError,
    build_llm_kwargs,
    normalize_backend,
    normalize_device,
)
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
    device: str = typer.Option(
        DEVICE_AUTO,
        help="Inference device mode: auto, cuda, or cpu.",
        envvar="DEVICE",
    ),
    enforce_eager: bool = typer.Option(
        False,
        "--enforce-eager/--no-enforce-eager",
        help="Disable vLLM compile/warmup and run entirely in eager mode.",
        envvar="ENFORCE_EAGER",
    ),
    backend: str = typer.Option(
        BACKEND_VLLM,
        help="Inference backend: vllm or transformers.",
        envvar="BACKEND",
    ),
) -> None:
    """Start the Qwen3-ASR gRPC server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )
    try:
        device = normalize_device(device)
        backend = normalize_backend(backend)
    except ValueError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=2) from exc

    try:
        asyncio.run(
            _serve_async(
                model=model,
                port=port,
                max_new_tokens=max_new_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                device=device,
                enforce_eager=enforce_eager,
                backend=backend,
            )
        )
    except KeyboardInterrupt:
        logger.info("Shutting down ...")
    except DeviceSelectionError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=2) from exc


async def _serve_async(
    model: str,
    port: int,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    device: str,
    enforce_eager: bool,
    backend: str,
) -> None:
    logger.info("Loading model from %s …", model)
    llm_kwargs = build_llm_kwargs(
        max_new_tokens=max_new_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        device=device,
        enforce_eager=enforce_eager,
    )
    try:
        if backend == BACKEND_TRANSFORMERS:
            from qwen_asr.inference.transformers_qwen3_asr import TransformersQwen3ASRModel
            tf_kwargs = {"max_new_tokens": llm_kwargs.get("max_new_tokens", max_new_tokens), "device": llm_kwargs.get("device")}
            asr_model = TransformersQwen3ASRModel.LLM(model=model, **tf_kwargs)
        else:
            from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
            asr_model = Qwen3ASRModel.LLM(model=model, **llm_kwargs)
    except Exception:
        if backend == BACKEND_VLLM and llm_kwargs.get("device") == DEVICE_CPU:
            error_message = (
                "Failed to initialize vLLM in CPU mode. Common causes include "
                "torch.compile or inductor warmup instability, insufficient memory, "
                "or unsupported CPU runtime settings."
            )
            if not llm_kwargs.get("enforce_eager"):
                error_message += (
                    " Retry with ENFORCE_EAGER=1 or --enforce-eager to disable "
                    "vLLM compile and warmup."
                )
            logger.error("%s", error_message)
        raise
    logger.info("Model loaded successfully.")

    server = grpc.aio.server()
    servicer = ASRServicer(asr_model)
    add_UxSpeechServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    logger.info("gRPC server listening on port %d", port)

    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        servicer.close()


if __name__ == "__main__":
    app()
