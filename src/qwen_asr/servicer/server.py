"""gRPC 服务启动、故障监督与优雅关闭。"""

from __future__ import annotations

import asyncio
import logging
import signal

import grpc

from qwen_asr.inferencers.pipeline import ASRPipeline
from qwen_asr.protos.asr.ux_speech_pb2_grpc import add_UxSpeechServicer_to_server
from qwen_asr.servicer.servicer import ASRServicer

logger = logging.getLogger(__name__)


async def run_server(
    inferencer: ASRPipeline,
    *,
    context: str,
    server_port: int,
    max_concurrent_rpcs: int = 32,
) -> None:
    """启动 gRPC 服务并管理其生命周期。"""
    server = grpc.aio.server(
        options=[("grpc.so_reuseport", 0), ("grpc.max_receive_message_length", 4 * 1024 * 1024)],
    )
    servicer = ASRServicer(inferencer=inferencer, default_context=context,
                           max_concurrent_requests=max_concurrent_rpcs)
    add_UxSpeechServicer_to_server(servicer, server)
    if not server.add_insecure_port(f"[::]:{server_port}"):
        raise RuntimeError(f"Cannot bind port {server_port}")
    await server.start()
    logger.info("gRPC server listening on port %d", server_port)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    tasks = [asyncio.create_task(server.wait_for_termination()), asyncio.create_task(stop.wait())]
    backend = inferencer.inferencer
    if hasattr(backend, "wait_failure"):
        tasks.append(asyncio.create_task(backend.wait_failure()))
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        try:
            # Cancelling wait_for_termination before stop also cancels gRPC's
            # internal shutdown future. Drain the server before its waiter.
            await server.stop(grace=5)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            servicer.close()
            if hasattr(backend, "aclose"):
                await backend.aclose()
