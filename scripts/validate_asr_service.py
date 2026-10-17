"""真实服务的容量、取消、deadline 与后续请求隔离验证。"""

import asyncio
import json
from pathlib import Path

import grpc
import typer

from benchmark_asr import request, variants
from qwen_asr.protos.asr.ux_speech_pb2 import (
    RecognitionConfig, StreamingRecognitionConfig, StreamingRecognizeRequest,
)
from qwen_asr.protos.asr.ux_speech_pb2_grpc import UxSpeechStub

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output: Path = typer.Option(...),
    server: str = typer.Option("127.0.0.1:50011"),
    audio: Path = typer.Option(Path("data-bin/rita.wav")),
):
    async def validate():
        sample_rate, inputs = variants(audio, [])
        pcm = inputs[0]["pcm"]
        async with grpc.aio.insecure_channel(server) as channel:
            await asyncio.wait_for(channel.channel_ready(), 30)
            stub = UxSpeechStub(channel)
            gate = asyncio.Event()

            async def messages(hold=False):
                yield StreamingRecognizeRequest(streaming_config=StreamingRecognitionConfig(
                    config=RecognitionConfig(encoding=RecognitionConfig.LINEAR16,
                                             sample_rate_hertz=sample_rate), interim_results=True))
                if hold:
                    await gate.wait()
                yield StreamingRecognizeRequest(audio_content=pcm)

            calls = [stub.StreamingRecognize(messages(hold=True), timeout=30) for _ in range(32)]
            readers = [asyncio.create_task(call.read()) for call in calls]
            try:
                await asyncio.sleep(1)
                # Check the terminal status without sending an audio body to a
                # rejected stream (gRPC can otherwise mask it as a send error).
                # A separate connection also bypasses the client's HTTP/2
                # stream limit, ensuring this RPC actually reaches the server.
                async with grpc.aio.insecure_channel(server) as extra_channel:
                    extra_stub = UxSpeechStub(extra_channel)
                    extra = extra_stub.StreamingRecognize(messages(hold=True), timeout=10)
                    status = await extra.code()
                assert status == grpc.StatusCode.RESOURCE_EXHAUSTED, status
            finally:
                for call in calls:
                    call.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
            await asyncio.sleep(0.5)
            cancelled = stub.StreamingRecognize(messages(), timeout=180)
            first = await cancelled.read()
            assert first.results and first.results[0].alternative.transcript
            cancelled.cancel()
            deadline = stub.StreamingRecognize(messages(), timeout=0.001)
            try:
                await deadline.read()
                raise AssertionError("Deadline should expire")
            except grpc.aio.AioRpcError as exc:
                assert exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED, exc
            result = await request(stub, pcm, sample_rate, 180)
            assert not result["error"], result
            expected = "在同一个话题上纠缠不清的人，是不会讨女孩子喜欢的。"
            assert result["transcript"] == expected, result
            return {"capacity_32_rejects_33": True, "cancel_after_first_text": True,
                    "deadline_exceeded": True, "subsequent_request_matches_expected_transcript": True,
                    "subsequent_request": result}

    result = asyncio.run(validate())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
