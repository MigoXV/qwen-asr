"""对已经启动的 Qwen3-ASR gRPC 服务执行 GPU 显存稳态测试。"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import grpc
import numpy as np

from qwen_asr.protos.asr.ux_speech_pb2 import (
    RecognitionConfig,
    StreamingRecognitionConfig,
    StreamingRecognizeRequest,
)
from qwen_asr.protos.asr.ux_speech_pb2_grpc import UxSpeechStub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM ASR GPU 显存稳态测试")
    parser.add_argument("--audio", type=Path, required=True, help="用于循环填充的 WAV")
    parser.add_argument("--server", default="localhost:50051", help="gRPC 地址")
    parser.add_argument("--gpu-index", type=int, default=0, help="nvidia-smi GPU 编号")
    parser.add_argument("--duration", type=float, default=60.0, help="请求时长（秒）")
    parser.add_argument("--warmup", type=int, default=5, help="预热请求数")
    parser.add_argument("--requests", type=int, default=50, help="完整请求数")
    parser.add_argument("--cancellations", type=int, default=20, help="取消请求数")
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--max-drift-mib", type=float, default=256.0)
    parser.add_argument("--max-slope-mib", type=float, default=2.0)
    parser.add_argument("--max-total-memory-mib", type=float, default=12288.0)
    return parser.parse_args()


def load_pcm(path: Path, duration: float) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("压力测试 WAV 必须是 PCM16")
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    target_samples = round(duration * sample_rate)
    if target_samples <= 0 or samples.size == 0:
        raise ValueError("音频和目标时长必须非空")
    repeats = (target_samples + samples.size - 1) // samples.size
    return np.tile(samples, repeats)[:target_samples], sample_rate


def unique_pcm_bytes(samples: np.ndarray, request_index: int) -> bytes:
    varied = samples.copy()
    position = request_index % varied.size
    value = int(varied[position])
    varied[position] = np.int16(max(-32768, min(32767, value + request_index + 1)))
    return varied.tobytes()


async def request_messages(
    audio_bytes: bytes,
    sample_rate: int,
) -> AsyncIterator[StreamingRecognizeRequest]:
    yield StreamingRecognizeRequest(
        streaming_config=StreamingRecognitionConfig(
            config=RecognitionConfig(
                encoding=RecognitionConfig.LINEAR16,
                sample_rate_hertz=sample_rate,
            ),
            interim_results=True,
        )
    )
    yield StreamingRecognizeRequest(audio_content=audio_bytes)


async def run_request(
    stub: UxSpeechStub,
    audio_bytes: bytes,
    sample_rate: int,
    *,
    cancel_after_first: bool,
) -> float:
    started = time.perf_counter()
    call = stub.StreamingRecognize(request_messages(audio_bytes, sample_rate))
    async for _ in call:
        if cancel_after_first:
            call.cancel()
            break
    return time.perf_counter() - started


def gpu_memory_mib(gpu_index: int) -> float:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip().splitlines()[0])


def linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


async def main() -> int:
    args = parse_args()
    samples, sample_rate = load_pcm(args.audio, args.duration)
    channel = grpc.aio.insecure_channel(args.server)
    stub = UxSpeechStub(channel)
    request_index = 0
    observed_memory: list[float] = []

    try:
        print("phase,index,elapsed_s,gpu_memory_mib")
        for index in range(args.warmup):
            elapsed = await run_request(
                stub,
                unique_pcm_bytes(samples, request_index),
                sample_rate,
                cancel_after_first=False,
            )
            request_index += 1
            await asyncio.sleep(args.settle_seconds)
            memory = gpu_memory_mib(args.gpu_index)
            observed_memory.append(memory)
            print(f"warmup,{index + 1},{elapsed:.3f},{memory:.0f}")

        normal_memory: list[float] = []
        for index in range(args.requests):
            elapsed = await run_request(
                stub,
                unique_pcm_bytes(samples, request_index),
                sample_rate,
                cancel_after_first=False,
            )
            request_index += 1
            await asyncio.sleep(args.settle_seconds)
            memory = gpu_memory_mib(args.gpu_index)
            normal_memory.append(memory)
            observed_memory.append(memory)
            print(f"complete,{index + 1},{elapsed:.3f},{memory:.0f}")

        cancellation_memory: list[float] = []
        for index in range(args.cancellations):
            elapsed = await run_request(
                stub,
                unique_pcm_bytes(samples, request_index),
                sample_rate,
                cancel_after_first=True,
            )
            request_index += 1
            await asyncio.sleep(args.settle_seconds)
            memory = gpu_memory_mib(args.gpu_index)
            cancellation_memory.append(memory)
            observed_memory.append(memory)
            print(f"cancel,{index + 1},{elapsed:.3f},{memory:.0f}")
    finally:
        await channel.close()

    drift = max(0.0, normal_memory[-1] - normal_memory[0]) if normal_memory else 0.0
    slope = max(0.0, linear_slope(normal_memory))
    cancel_drift = (
        max(0.0, cancellation_memory[-1] - cancellation_memory[0])
        if cancellation_memory
        else 0.0
    )
    cancel_slope = max(0.0, linear_slope(cancellation_memory))
    peak_memory = max(observed_memory, default=0.0)
    print(
        f"result,peak_memory_mib={peak_memory:.0f},"
        f"complete_drift_mib={drift:.1f},"
        f"complete_slope_mib_per_request={slope:.3f},"
        f"cancel_drift_mib={cancel_drift:.1f},"
        f"cancel_slope_mib_per_request={cancel_slope:.3f}"
    )
    if (
        peak_memory > args.max_total_memory_mib
        or max(drift, cancel_drift) > args.max_drift_mib
        or max(slope, cancel_slope) > args.max_slope_mib
    ):
        print("FAIL: 总显存超过预算或预热后仍在持续增长")
        return 1
    print("PASS: 显存已进入稳态")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
