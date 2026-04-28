# coding=utf-8
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEVICE_AUTO = "auto"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"
DEVICE_CHOICES = {DEVICE_AUTO, DEVICE_CUDA, DEVICE_CPU}


class DeviceSelectionError(RuntimeError):
    """Raised when the requested inference device cannot be used."""


def cuda_is_available() -> bool:
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


def normalize_device(device: str) -> str:
    normalized = (device or DEVICE_AUTO).strip().lower()
    if normalized not in DEVICE_CHOICES:
        choices = ", ".join(sorted(DEVICE_CHOICES))
        raise ValueError(f"Invalid DEVICE value '{device}'. Expected one of: {choices}.")
    return normalized


def build_llm_kwargs(
    *,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    device: str,
    cuda_available: bool | None = None,
) -> dict[str, Any]:
    device = normalize_device(device)
    if cuda_available is None:
        cuda_available = cuda_is_available()

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }

    if device == DEVICE_CPU or (device == DEVICE_AUTO and not cuda_available):
        if device == DEVICE_CPU:
            logger.warning(
                "Starting vLLM with device='cpu'. CPU inference may be slow and "
                "requires a CPU-enabled vLLM build."
            )
        else:
            logger.warning(
                "CUDA is not available; starting vLLM with device='cpu'. CPU inference "
                "may be slow and requires a CPU-enabled vLLM build."
            )
        kwargs["device"] = DEVICE_CPU
    elif device == DEVICE_CUDA and not cuda_available:
        raise DeviceSelectionError(
            "DEVICE=cuda was requested, but CUDA is not available. Run the container "
            "with --gpus all or use DEVICE=cpu to try CPU inference."
        )

    return kwargs
