# coding=utf-8
"""Qwen3-ASR gRPC 服务的 Typer CLI 入口。"""

from __future__ import annotations

import logging

import typer

from qwen_asr.commands import serving
from qwen_asr.commands.options import AscendGraphMode, Device

app = typer.Typer(
    name="qwen-asr",
    help="Qwen3-ASR gRPC 推理服务。",
    add_completion=False,
    no_args_is_help=True,
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
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
    serving.start_vllm(
        model=model,
        context=context,
        server_port=server_port,
        max_new_tokens=max_new_tokens,
        device=device,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
    )


@app.command("vllm-ascend")
def serve_vllm_ascend(
    model: str = typer.Option(..., "--model", envvar="QWEN_ASR_MODEL", help="模型目录。"),
    context: str = typer.Option("", "--context", envvar="QWEN_ASR_CONTEXT"),
    server_port: int = typer.Option(50011, "--server-port", min=1, max=65535, envvar="QWEN_ASR_SERVER_PORT"),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens", min=1, envvar="QWEN_ASR_MAX_NEW_TOKENS"),
    workers: int = typer.Option(4, "--workers", "--max-num-seqs", min=1, max=32, envvar="QWEN_ASR_WORKERS", help="共享单模型的引擎并发上限，对应 vLLM max_num_seqs。"),
    max_concurrent_rpcs: int = typer.Option(32, "--max-concurrent-rpcs", min=1, envvar="QWEN_ASR_MAX_CONCURRENT_RPCS"),
    memory_utilization: float = typer.Option(0.10, "--memory-utilization", min=0.01, max=0.9, envvar="QWEN_ASR_MEMORY_UTILIZATION", help="单引擎 Torch 峰值及 KV Cache 的 HBM 预算比例。"),
    kv_cache_memory_mib: int | None = typer.Option(None, "--kv-cache-memory-mib", min=1, envvar="QWEN_ASR_KV_CACHE_MEMORY_MIB", help="固定 KV Cache MiB；默认按槽位数和上下文长度精确计算。"),
    max_model_len: int = typer.Option(1024, "--max-model-len", min=1, envvar="QWEN_ASR_VLLM_MAX_MODEL_LEN"),
    max_num_batched_tokens: int = typer.Option(4096, "--max-num-batched-tokens", min=1, envvar="QWEN_ASR_VLLM_MAX_NUM_BATCHED_TOKENS"),
    enforce_eager: bool = typer.Option(False, "--enforce-eager/--no-enforce-eager", envvar="QWEN_ASR_VLLM_ENFORCE_EAGER"),
    graph_mode: AscendGraphMode = typer.Option(AscendGraphMode.FULL_DECODE_ONLY, "--graph-mode", envvar="QWEN_ASR_ASCEND_GRAPH_MODE", help="ACL Graph 执行方式；完整解码图或分段图。"),
    async_scheduling: bool = typer.Option(True, "--async-scheduling/--no-async-scheduling", envvar="QWEN_ASR_VLLM_ASYNC_SCHEDULING", help="异步调度，减少调度与设备执行之间的等待。"),
    triton_decode: bool = typer.Option(False, "--triton-decode/--no-triton-decode", envvar="QWEN_ASR_ASCEND_TRITON_DECODE", help="910B 1.7B BF16 分页解码 kernel；要求当前已验证的 0.18/2.9.1/3.2.2 栈、8 槽以内及 1024 上下文。"),
    native_fia_update: bool = typer.Option(False, "--native-fia-update/--no-native-fia-update", envvar="QWEN_ASR_ASCEND_NATIVE_FIA_UPDATE", help="使用新版 torch-npu 的 C++ 融合 FIA 图更新；与 triton-decode 分开选择。"),
    encoder_graph: bool = typer.Option(False, "--encoder-graph/--no-encoder-graph", envvar="QWEN_ASR_ASCEND_ENCODER_GRAPH", help="新栈 v3 attention 音频编码图；要求 Triton 解码及 1024 batched tokens，最多缓存 4 档精确长度。"),
) -> None:
    """单个 Ascend Engine Core 共享模型，限制活跃槽位和外层并发。"""
    configure_logging()
    if workers > max_concurrent_rpcs:
        raise typer.BadParameter("workers 不能超过 max-concurrent-rpcs")
    if triton_decode and (workers > 8 or max_model_len != 1024 or enforce_eager
                          or graph_mode != AscendGraphMode.FULL_DECODE_ONLY):
        raise typer.BadParameter("triton-decode 要求不超过 8 槽、1024 上下文及 FULL_DECODE_ONLY 图模式")
    if native_fia_update and triton_decode:
        raise typer.BadParameter("native-fia-update 与 triton-decode 需要分开选择")
    if encoder_graph and (not triton_decode or max_num_batched_tokens != 1024):
        raise typer.BadParameter("encoder-graph 要求 triton-decode 及 max-num-batched-tokens=1024")
    if native_fia_update and (workers > 8 or max_model_len != 1024 or enforce_eager
                              or graph_mode != AscendGraphMode.FULL_DECODE_ONLY):
        raise typer.BadParameter("native-fia-update 要求不超过 8 槽、1024 上下文及 FULL_DECODE_ONLY 图模式")
    serving.start_vllm_ascend(
        model=model,
        context=context,
        server_port=server_port,
        max_new_tokens=max_new_tokens,
        workers=workers,
        max_concurrent_rpcs=max_concurrent_rpcs,
        memory_utilization=memory_utilization,
        kv_cache_memory_mib=kv_cache_memory_mib,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
        graph_mode=graph_mode,
        async_scheduling=async_scheduling,
        triton_decode=triton_decode,
        native_fia_update=native_fia_update,
        encoder_graph=encoder_graph,
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
    serving.start_transformers(
        model=model,
        context=context,
        server_port=server_port,
        max_new_tokens=max_new_tokens,
        device=device,
    )


if __name__ == "__main__":
    app()
