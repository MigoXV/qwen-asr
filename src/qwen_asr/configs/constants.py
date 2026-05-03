# coding=utf-8
from __future__ import annotations

DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"

BACKEND_VLLM = "vllm"
BACKEND_TRANSFORMERS = "transformers"
BACKEND_CHOICES = {BACKEND_VLLM, BACKEND_TRANSFORMERS}
