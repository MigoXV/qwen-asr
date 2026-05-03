# coding=utf-8
from __future__ import annotations

from qwen_asr.configs import AppConfig


def build_vllm_kwargs(config: AppConfig) -> dict[str, object]:
    vllm_config = config.vllm
    if vllm_config is None:
        raise ValueError("vLLM backend selected but vllm config is missing.")

    kwargs: dict[str, object] = {
        "max_new_tokens": config.generation.max_new_tokens,
        "gpu_memory_utilization": vllm_config.gpu_memory_utilization,
        "max_model_len": vllm_config.max_model_len,
        "device": config.device,
    }
    if vllm_config.enforce_eager:
        kwargs["enforce_eager"] = True

    return kwargs


def build_transformers_kwargs(config: AppConfig) -> dict[str, object]:
    # 组装 Transformers 后端初始化参数，device 按配置原样交给后端处理。
    kwargs: dict[str, object] = {
        "max_new_tokens": config.generation.max_new_tokens,
        "device": config.device,
    }
    return kwargs
