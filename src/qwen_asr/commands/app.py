# coding=utf-8
"""Qwen3-ASR gRPC 服务的 Typer CLI 入口。"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum

import grpc
import typer

from qwen_asr.inferencers.grpc_inferencer import GrpcInferencer
from qwen_asr.protos.asr.ux_speech_pb2_grpc import add_UxSpeechServicer_to_server
from qwen_asr.servicer.servicer import ASRServicer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="qwen-asr",
    help="Qwen3-ASR gRPC 推理服务。",
    add_completion=False,
    no_args_is_help=True,
)


class Device(str, Enum):
    """服务支持的推理设备。"""

    CPU = "cpu"
    CUDA = "cuda"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )


async def run_server(
    inferencer: GrpcInferencer,
    *,
    context: str,
    server_port: int,
) -> None:
    """启动 gRPC 服务并管理其生命周期。"""
    server = grpc.aio.server()
    servicer = ASRServicer(inferencer=inferencer, default_context=context)
    add_UxSpeechServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{server_port}")
    await server.start()
    logger.info("gRPC server listening on port %d", server_port)

    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        servicer.close()


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
) -> GrpcInferencer:
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

    return GrpcInferencer(
        inferencer=VLLMInferencer.LLM(model=model, **kwargs),
    )


def create_transformers_inferencer(
    *,
    model: str,
    max_new_tokens: int,
    device: Device,
) -> GrpcInferencer:
    """按 Transformers 命令参数创建推理适配器。"""
    from qwen_asr.inferencers.transformers import TransformersInferencer

    logger.info("Loading Transformers model from %s …", model)
    return GrpcInferencer(
        inferencer=TransformersInferencer.LLM(
            model=model,
            max_new_tokens=max_new_tokens,
            device=device.value,
        ),
    )


@app.command("vllm")
def serve_vllm(
    model: str = typer.Option(
        ...,
        "--model",
        help="模型目录或 Hugging Face 模型名称。",
        envvar="QWEN_ASR_MODEL",
        show_envvar=True,
    ),
    context: str = typer.Option(
        "",
        "--context",
        help="默认识别上下文；请求中的热词优先。",
        envvar="QWEN_ASR_CONTEXT",
        show_envvar=True,
    ),
    server_port: int = typer.Option(
        50051,
        "--server-port",
        min=1,
        max=65535,
        help="gRPC 服务监听端口。",
        envvar="QWEN_ASR_SERVER_PORT",
        show_envvar=True,
    ),
    max_new_tokens: int = typer.Option(
        256,
        "--max-new-tokens",
        min=1,
        help="单次识别最多生成的 token 数。",
        envvar="QWEN_ASR_MAX_NEW_TOKENS",
        show_envvar=True,
    ),
    device: Device = typer.Option(
        Device.CPU,
        "--device",
        help="推理设备。",
        envvar="QWEN_ASR_DEVICE",
        show_envvar=True,
        case_sensitive=False,
    ),
    gpu_memory_utilization: float = typer.Option(
        0.27,
        "--gpu-memory-utilization",
        min=0.0,
        max=1.0,
        help="vLLM 可使用的 GPU 显存比例。",
        envvar="QWEN_ASR_VLLM_GPU_MEMORY_UTILIZATION",
        show_envvar=True,
    ),
    max_model_len: int = typer.Option(
        1536,
        "--max-model-len",
        min=1,
        help="vLLM 最大模型上下文长度。",
        envvar="QWEN_ASR_VLLM_MAX_MODEL_LEN",
        show_envvar=True,
    ),
    max_num_seqs: int = typer.Option(
        4,
        "--max-num-seqs",
        min=1,
        help="vLLM 同时处理的最大序列数。",
        envvar="QWEN_ASR_VLLM_MAX_NUM_SEQS",
        show_envvar=True,
    ),
    max_num_batched_tokens: int = typer.Option(
        1536,
        "--max-num-batched-tokens",
        min=1,
        help="vLLM 单次调度的最大 token 数。",
        envvar="QWEN_ASR_VLLM_MAX_NUM_BATCHED_TOKENS",
        show_envvar=True,
    ),
    enforce_eager: bool = typer.Option(
        False,
        "--enforce-eager/--no-enforce-eager",
        help="是否强制 vLLM 使用 eager 模式。",
        envvar="QWEN_ASR_VLLM_ENFORCE_EAGER",
        show_envvar=True,
    ),
) -> None:
    """使用 vLLM 后端启动服务。"""
    configure_logging()
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


@app.command("transformers")
def serve_transformers(
    model: str = typer.Option(
        ...,
        "--model",
        help="模型目录或 Hugging Face 模型名称。",
        envvar="QWEN_ASR_MODEL",
        show_envvar=True,
    ),
    context: str = typer.Option(
        "",
        "--context",
        help="默认识别上下文；请求中的热词优先。",
        envvar="QWEN_ASR_CONTEXT",
        show_envvar=True,
    ),
    server_port: int = typer.Option(
        50051,
        "--server-port",
        min=1,
        max=65535,
        help="gRPC 服务监听端口。",
        envvar="QWEN_ASR_SERVER_PORT",
        show_envvar=True,
    ),
    max_new_tokens: int = typer.Option(
        256,
        "--max-new-tokens",
        min=1,
        help="单次识别最多生成的 token 数。",
        envvar="QWEN_ASR_MAX_NEW_TOKENS",
        show_envvar=True,
    ),
    device: Device = typer.Option(
        Device.CPU,
        "--device",
        help="推理设备。",
        envvar="QWEN_ASR_DEVICE",
        show_envvar=True,
        case_sensitive=False,
    ),
) -> None:
    """使用 Transformers 后端启动服务。"""
    configure_logging()
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


if __name__ == "__main__":
    app()
