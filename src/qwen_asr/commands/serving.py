"""按 CLI 参数组装推理后端并启动服务。"""

from __future__ import annotations

import asyncio
import logging

from qwen_asr.commands.options import AscendGraphMode, Device
from qwen_asr.inferencers.pipeline import ASRPipeline
from qwen_asr.servicer.server import run_server

logger = logging.getLogger(__name__)


def create_vllm_inferencer(
    *,
    model: str,
    max_new_tokens: int,
    device: Device,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    enforce_eager: bool,
) -> ASRPipeline:
    """按 vLLM 命令参数创建推理适配器。"""
    from qwen_asr.inferencers.vllm import VLLMInferencer

    logger.info("Loading vLLM model from %s …", model)
    kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
    }
    if device is not Device.CUDA:
        logger.warning(
            "vLLM 后端由 vLLM 自动选择加速设备，--device=%s 不会传给引擎。",
            device.value,
        )
    if enforce_eager:
        kwargs["enforce_eager"] = True

    return ASRPipeline(
        inferencer=VLLMInferencer.LLM(model=model, **kwargs),
    )


def create_transformers_inferencer(
    *,
    model: str,
    max_new_tokens: int,
    device: Device,
) -> ASRPipeline:
    """按 Transformers 命令参数创建推理适配器。"""
    from qwen_asr.inferencers.transformers import TransformersInferencer

    logger.info("Loading Transformers model from %s …", model)
    return ASRPipeline(
        inferencer=TransformersInferencer.LLM(
            model=model,
            max_new_tokens=max_new_tokens,
            device=device.value,
        ),
    )


def start_vllm(
    *,
    model: str,
    context: str,
    server_port: int,
    max_new_tokens: int,
    device: Device,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    enforce_eager: bool,
) -> None:
    inferencer = create_vllm_inferencer(
        model=model,
        max_new_tokens=max_new_tokens,
        device=device,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
    )
    asyncio.run(
        run_server(
            inferencer,
            context=context,
            server_port=server_port,
        )
    )


def start_vllm_ascend(
    *,
    model: str,
    context: str,
    server_port: int,
    max_new_tokens: int,
    workers: int,
    max_concurrent_rpcs: int,
    memory_utilization: float,
    kv_cache_memory_mib: int | None,
    max_model_len: int,
    max_num_batched_tokens: int,
    enforce_eager: bool,
    graph_mode: AscendGraphMode,
    async_scheduling: bool,
    triton_decode: bool = False,
    native_fia_update: bool = False,
    encoder_graph: bool = False,
) -> None:
    from qwen_asr.engines.vllm.process import VLLMEngineClient
    from qwen_asr.engines.vllm.ascend.config import build_engine_config

    config = build_engine_config(
        model=model,
        max_new_tokens=max_new_tokens,
        workers=workers,
        memory_utilization=memory_utilization,
        kv_cache_memory_mib=kv_cache_memory_mib,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
        graph_mode=graph_mode.value,
        async_scheduling=async_scheduling,
        triton_decode=triton_decode,
        native_fia_update=native_fia_update,
        encoder_graph=encoder_graph,
    )

    async def serve():
        engine = VLLMEngineClient(max_inflight=max_concurrent_rpcs, engine_config=config)
        try:
            await engine.start()
            logger.info("Ascend service ready: model owners=1, execution slots=%d, RPC capacity=%d, config=%s", workers, max_concurrent_rpcs, config)
            await run_server(ASRPipeline(engine), context=context, server_port=server_port, max_concurrent_rpcs=max_concurrent_rpcs)
        finally:
            await engine.aclose()

    asyncio.run(serve())

def start_transformers(
    *,
    model: str,
    context: str,
    server_port: int,
    max_new_tokens: int,
    device: Device,
) -> None:
    inferencer = create_transformers_inferencer(
        model=model,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    asyncio.run(
        run_server(
            inferencer,
            context=context,
            server_port=server_port,
        )
    )
