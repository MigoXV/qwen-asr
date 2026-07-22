"""vLLM 子进程可重复加载的 Qwen3-ASR 注册入口。"""

from __future__ import annotations


def register_qwen_asr() -> None:
    """在当前进程注册 Transformers 配置与 vLLM 模型架构。"""
    from transformers import AutoConfig, AutoProcessor
    from vllm import ModelRegistry

    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRConfig,
    )
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import (
        Qwen3ASRProcessor,
    )

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor, exist_ok=True)
    ModelRegistry.register_model(
        "Qwen3ASRForConditionalGeneration",
        "qwen_asr.core.vllm_backend.qwen3_asr:Qwen3ASRForConditionalGeneration",
    )
