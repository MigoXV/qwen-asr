# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from qwen_asr.configs.constants import (
    BACKEND_CHOICES,
    BACKEND_TRANSFORMERS,
    BACKEND_VLLM,
    DEVICE_CPU,
)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 4096


@dataclass
class VLLMConfig:
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    enforce_eager: bool = False


@dataclass
class TransformersConfig:
    pass


@dataclass
class AppConfig:
    model: str = MISSING
    backend: str = BACKEND_VLLM
    context: str = ""
    server_port: int = 50051
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    device: str = DEVICE_CPU
    vllm: Optional[VLLMConfig] = None
    transformers: Optional[TransformersConfig] = None

    def __post_init__(self) -> None:
        self.backend = self.normalize_backend(self.backend)

        if self.backend == BACKEND_VLLM and self.vllm is None:
            self.vllm = VLLMConfig()
        elif self.backend == BACKEND_TRANSFORMERS and self.transformers is None:
            self.transformers = TransformersConfig()

    @staticmethod
    def normalize_backend(backend: str) -> str:
        normalized = (backend or "").strip().lower()
        if normalized not in BACKEND_CHOICES:
            choices = ", ".join(sorted(BACKEND_CHOICES))
            raise ValueError(
                f"Invalid BACKEND value '{backend}'. Expected one of: {choices}."
            )
        return normalized
