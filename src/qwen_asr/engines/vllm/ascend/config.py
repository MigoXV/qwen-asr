"""Host-only sizing for BF16 KV cache; no model or NPU context is created."""


def kv_cache_bytes(model: str, slots: int, max_model_len: int) -> int:
    from qwen_asr.models.transformers.configuration_qwen3_asr import Qwen3ASRConfig

    config = Qwen3ASRConfig.from_pretrained(model).thinker_config.text_config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    # Two tensors (K/V), two bytes (BF16); Ascend uses 128-token blocks.
    tokens_per_slot = ((max_model_len + 127) // 128) * 128
    return slots * tokens_per_slot * 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * 2


def build_engine_config(
    *,
    model: str,
    max_new_tokens: int,
    workers: int,
    memory_utilization: float,
    kv_cache_memory_mib: int | None,
    max_model_len: int,
    max_num_batched_tokens: int,
    enforce_eager: bool,
    graph_mode: str,
    async_scheduling: bool,
    triton_decode: bool = False,
    native_fia_update: bool = False,
    encoder_graph: bool = False,
) -> dict:
    """构造单模型 Ascend 引擎配置，不创建模型或设备上下文。"""
    cache_bytes = (kv_cache_memory_mib * 1024 * 1024 if kv_cache_memory_mib is not None
                   else kv_cache_bytes(model, workers, max_model_len))

    config = {
        "model": model,
        "max_new_tokens": max_new_tokens,
        "gpu_memory_utilization": memory_utilization,
        "kv_cache_memory_bytes": cache_bytes,
        "worker_cls": "qwen_asr.engines.vllm.ascend.worker.SharedNPUWorker",
        "block_size": 128,
        "max_model_len": max_model_len,
        "max_num_seqs": workers,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enforce_eager": enforce_eager,
        "dtype": "bfloat16",
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "limit_mm_per_prompt": {"audio": 1},
        "async_scheduling": async_scheduling,
    }
    if not enforce_eager:
        config["compilation_config"] = {
            "mode": 3,
            "cudagraph_mode": graph_mode,
            "cudagraph_capture_sizes": list(range(1, workers + 1)),
        }
    if triton_decode:
        config["additional_config"] = {"qwen_asr_triton_decode": True}
    if native_fia_update:
        config.setdefault("additional_config", {})["qwen_asr_native_fia_update"] = True
    if encoder_graph:
        config.setdefault("additional_config", {})["qwen_asr_encoder_graph"] = True

    return config
