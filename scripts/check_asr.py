"""通过真实识别请求判定模型就绪，供 systemd 和手动检查使用。"""

import asyncio
import json
from pathlib import Path

import grpc
import typer

from benchmark_asr import request, variants
from qwen_asr.protos.asr.ux_speech_pb2_grpc import UxSpeechStub

app = typer.Typer(add_completion=False)


@app.command()
def main(
    server: str = typer.Option("127.0.0.1:50011"),
    audio: Path = typer.Option(Path("data-bin/rita.wav")),
    timeout: float = typer.Option(1500, min=1),
    requests: int = typer.Option(4, min=1, max=32, envvar="QWEN_ASR_WORKERS"),
):
    async def check():
        sr, inputs = variants(audio, [])
        async with grpc.aio.insecure_channel(server, options=[
            ("grpc.initial_reconnect_backoff_ms", 200),
            ("grpc.min_reconnect_backoff_ms", 200),
            ("grpc.max_reconnect_backoff_ms", 1000),
        ]) as channel:
            await asyncio.wait_for(channel.channel_ready(), timeout)
            stub = UxSpeechStub(channel)
            results = await asyncio.gather(*(
                request(stub, inputs[0]["pcm"], sr, 180) for _ in range(requests)))
            typer.echo(json.dumps(results, ensure_ascii=False, indent=2))
            if any(result["error"] for result in results):
                raise RuntimeError("Model readiness inference failed")
    asyncio.run(check())


if __name__ == "__main__":
    app()
