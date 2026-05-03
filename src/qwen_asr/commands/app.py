# coding=utf-8
"""
Qwen3-ASR gRPC 服务的 Typer CLI 入口。

运行时配置从结构化 YAML 文件加载。

示例：
    python -m qwen_asr.commands.app serve --config config.yaml
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import grpc
import typer
from omegaconf import OmegaConf

from qwen_asr.configs import AppConfig
from qwen_asr.protos.asr.ux_speech_pb2_grpc import add_UxSpeechServicer_to_server
from qwen_asr.servicer.servicer import ASRServicer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="qwen-asr",
    help="Qwen3-ASR gRPC 推理服务。",
    add_completion=False,
)


async def run_server(config: AppConfig) -> None:
    # 初始化 gRPC 服务，servicer 内部负责加载推理器。
    server = grpc.aio.server()
    servicer = ASRServicer(config)
    add_UxSpeechServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{config.server_port}")
    await server.start()
    logger.info("gRPC server listening on port %d", config.server_port)

    try:
        await server.wait_for_termination()
    finally:
        # 退出时给 gRPC 一个短暂宽限期，并释放 servicer 持有的模型资源。
        await server.stop(grace=5)
        servicer.close()


@app.command()
def serve(
    config: Path = typer.Option(
        ...,
        "--config",
        help="Qwen3-ASR YAML 配置文件路径。",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        envvar="QWEN_ASR_CONFIG",
    ),
) -> None:
    """启动 Qwen3-ASR gRPC 服务。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )
    # 先合并默认结构化配置，再转换为应用配置对象。
    schema = OmegaConf.structured(AppConfig)
    loaded = OmegaConf.load(config)
    merged = OmegaConf.merge(schema, loaded)
    app_config = OmegaConf.to_object(merged)

    asyncio.run(run_server(app_config))


if __name__ == "__main__":
    app()
