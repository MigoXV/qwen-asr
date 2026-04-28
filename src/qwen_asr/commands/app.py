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
from typing import Any

import grpc
import typer

from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
from qwen_asr.protos.asr.ux_speech_pb2_grpc import add_UxSpeechServicer_to_server
from qwen_asr.servicer.servicer import ASRServicer

logger = logging.getLogger(__name__)

_DEVICE_AUTO = "auto"
_DEVICE_CUDA = "cuda"
_DEVICE_CPU = "cpu"
_DEVICE_CHOICES = {_DEVICE_AUTO, _DEVICE_CUDA, _DEVICE_CPU}


class DeviceSelectionError(RuntimeError):
    """Raised when the requested inference device cannot be used."""


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
        _DEVICE_AUTO,
        help="Inference device mode: auto, cuda, or cpu.",
        envvar="DEVICE",
    ),
) -> None:
    """Start the Qwen3-ASR gRPC server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )
    try:
        device = _normalize_device(device)
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
) -> None:
    logger.info("Loading model from %s …", model)
    llm_kwargs = _build_llm_kwargs(
        max_new_tokens=max_new_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        device=device,
    )
    try:
        asr_model = Qwen3ASRModel.LLM(model=model, **llm_kwargs)
    except Exception:
        if llm_kwargs.get("device") == _DEVICE_CPU:
            logger.error(
                "Failed to initialize vLLM in CPU mode. CUDA was not detected or "
                "DEVICE=cpu was requested, so the server tried device='cpu'. "
                "This image may contain a CUDA-only vLLM build; use a CPU-enabled "
                "vLLM image, or run the container with --gpus all for GPU inference."
            )
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


def _cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch is not installed; falling back to CPU mode.")
        return False
    except Exception as exc:
        logger.warning("Failed to import PyTorch: %s; falling back to CPU mode.", exc)
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception as exc:
        logger.warning(
            "Failed to check CUDA availability: %s; falling back to CPU mode.", exc
        )
        return False


def _normalize_device(device: str) -> str:
    normalized = (device or _DEVICE_AUTO).strip().lower()
    if normalized not in _DEVICE_CHOICES:
        choices = ", ".join(sorted(_DEVICE_CHOICES))
        raise ValueError(f"Invalid DEVICE value '{device}'. Expected one of: {choices}.")
    return normalized


def _build_llm_kwargs(
    *,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    device: str,
    cuda_available: bool | None = None,
) -> dict[str, Any]:
    device = _normalize_device(device)
    if cuda_available is None:
        cuda_available = _cuda_is_available()

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }

    if device == _DEVICE_CPU or (device == _DEVICE_AUTO and not cuda_available):
        if device == _DEVICE_CPU:
            logger.warning(
                "Starting vLLM with device='cpu'. CPU inference may be slow and "
                "requires a CPU-enabled vLLM build."
            )
        else:
            logger.warning(
                "CUDA is not available; starting vLLM with device='cpu'. CPU inference "
                "may be slow and requires a CPU-enabled vLLM build."
            )
        kwargs["device"] = _DEVICE_CPU
    elif device == _DEVICE_CUDA and not cuda_available:
        raise DeviceSelectionError(
            "DEVICE=cuda was requested, but CUDA is not available. Run the container "
            "with --gpus all or use DEVICE=cpu to try CPU inference."
        )

    return kwargs


if __name__ == "__main__":
    app()
