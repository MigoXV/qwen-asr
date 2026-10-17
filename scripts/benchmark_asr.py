"""以完整 PCM 上传测量离线 ASR 的吞吐倍速与首包延迟。"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import platform
import re
import subprocess
import time
import wave
from contextlib import suppress
from importlib.metadata import version
from pathlib import Path

import grpc
import numpy as np
import typer

from qwen_asr.protos.asr.ux_speech_pb2 import (
    RecognitionConfig, StreamingRecognitionConfig, StreamingRecognizeRequest,
)
from qwen_asr.protos.asr.ux_speech_pb2_grpc import UxSpeechStub

app = typer.Typer(add_completion=False)


def npu_memory_mib(output: str) -> int:
    # npu-smi combines several visual columns inside one pipe-delimited field.
    values = [int(used) for used, total in re.findall(r"(\d+)\s*/\s*(\d+)\s*\|", output)
              if int(total) > 0]
    if not values:
        raise ValueError("Cannot parse HBM usage from npu-smi")
    return max(values)


def variants(path: Path, durations: list[float]) -> tuple[int, list[dict]]:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("Require mono PCM16 WAV")
        sample_rate = audio.getframerate()
        source = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    if not len(source):
        raise ValueError("Empty audio")
    result = []
    for duration in [len(source) / sample_rate, *durations]:
        if not 0 < duration <= 30:
            raise ValueError("Audio duration must be <= 30 seconds")
        count = round(duration * sample_rate)
        pcm = np.tile(source, (count + len(source) - 1) // len(source))[:count].tobytes()
        if any(item["pcm"] == pcm for item in result):
            continue
        result.append({"name": f"rita-{duration:.3f}s", "duration_s": count / sample_rate,
                       "sha256": hashlib.sha256(pcm).hexdigest(), "pcm": pcm})
    return sample_rate, result


async def request(stub, pcm, sample_rate, timeout):
    async def messages():
        yield StreamingRecognizeRequest(streaming_config=StreamingRecognitionConfig(
            config=RecognitionConfig(encoding=RecognitionConfig.LINEAR16,
                                     sample_rate_hertz=sample_rate), interim_results=True))
        yield StreamingRecognizeRequest(audio_content=pcm)

    started = time.perf_counter()
    first_packet = first_text = None
    transcript = ""
    final = False
    error = None
    try:
        async for response in stub.StreamingRecognize(messages(), timeout=timeout):
            now = time.perf_counter()
            if first_packet is None:
                first_packet = now - started
            for result in response.results:
                transcript = result.alternative.transcript
                if transcript and first_text is None:
                    first_text = now - started
                final = final or result.is_final
        if not final or not transcript.strip():
            error = "Missing final or empty transcript"
    except grpc.aio.AioRpcError as exc:
        error = f"{exc.code().name}: {exc.details()}"
    return {"latency_s": time.perf_counter() - started,
            "first_packet_s": first_packet, "first_text_s": first_text,
            "transcript": transcript, "final": final, "error": error}


async def run_round(stubs, item, sample_rate, count, timeout):
    start = asyncio.Event()
    records = []
    active = peak = 0

    async def client(index, stub):
        nonlocal active, peak
        await start.wait()
        for sequence in range(count):
            active += 1
            peak = max(peak, active)
            try:
                result = await request(stub, item["pcm"], sample_rate, timeout)
                records.append({"client": index, "sequence": sequence, **result})
            finally:
                active -= 1

    tasks = [asyncio.create_task(client(i, stub)) for i, stub in enumerate(stubs)]
    started = time.perf_counter()
    start.set()
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    good = [r for r in records if not r["error"]]
    return {"elapsed_s": elapsed, "peak_client_inflight": peak, "requests": records,
            "throughput_x": len(good) * item["duration_s"] / elapsed,
            "success": len(good), "errors": len(records) - len(good)}


def percentile(values, p):
    return float(np.percentile(values, p)) if values else None


@app.command()
def main(
    workers: int = typer.Option(..., min=1, help="记录单模型引擎并发 max_num_seqs；不修改服务。"),
    output: Path = typer.Option(..., help="原始数据与汇总目录。"),
    audio: Path = typer.Option(Path("data-bin/rita.wav")),
    server: str = typer.Option("127.0.0.1:50011"),
    duration: list[float] = typer.Option([10.0, 20.0, 30.0]),
    clients: list[int] = typer.Option([1, 2, 4, 8, 16, 32]),
    rounds: int = typer.Option(3, min=1),
    requests_per_client: int = typer.Option(2, min=1),
    min_requests: int = typer.Option(8, min=1),
    timeout: float = typer.Option(180, min=1),
    warmup: int = typer.Option(2, min=0),
    memory_limit_mib: int = typer.Option(12288, min=1, help="NPU 整卡 HBM 采样上限，含驱动占用。"),
    execution_mode: str = typer.Option("unspecified", help="实际执行模式标签；须另存服务配置与图捕获/回放证据。"),
):
    """运行闭环并发矩阵，失败请求保留在原始记录中。"""
    if any(not 1 <= c <= 32 for c in clients):
        raise typer.BadParameter("客户端并发必须在 1..32")
    if any((output / name).exists() for name in ("metadata.json", "raw.jsonl", "resources.jsonl")):
        raise typer.BadParameter("输出目录已有测试数据，请使用新目录，避免混合不同轮次或配置。")
    sample_rate, inputs = variants(audio, duration)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"workers": workers, "server": server, "python": platform.python_version(),
                "packages": {p: version(p) for p in ("torch", "torch-npu", "vllm", "vllm-ascend")},
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "workspace_diff": subprocess.check_output(["git", "diff", "--stat"], text=True),
                "audio": str(audio), "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "inputs": [{k: v for k, v in item.items() if k != "pcm"} for item in inputs],
                "rounds": rounds, "clients": clients, "requests_per_client": requests_per_client,
                "min_requests": min_requests, "warmup_rounds": warmup,
                "model_owners": 1, "execution_slots": workers, "dtype": "bfloat16",
                "execution_mode": execution_mode,
                "memory_limit_mib": memory_limit_mib,
                "definition": "throughput_x = successful audio seconds / round makespan; first_packet/text from RPC start; full PCM uploaded immediately"}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    async def run():
        channels = [grpc.aio.insecure_channel(server) for _ in range(max(clients))]
        stubs = [UxSpeechStub(c) for c in channels]
        summaries = []
        measurement = asyncio.current_task()
        memory_failure = None
        async def sample_resources():
            nonlocal memory_failure
            while True:
                sample = {"time_unix": time.time()}
                for command in (["npu-smi", "info"], ["ps", "-eo", "pid,ppid,pcpu,rss,args"]):
                    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE)
                    data, _ = await process.communicate()
                    value = data.decode(errors="replace")
                    if command[0] == "npu-smi":
                        used = npu_memory_mib(value)
                        sample["hbm_used_mib"] = used
                        engine_pids = re.findall(r"\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|\s*VLLM", value)
                        sample["model_pids"] = engine_pids
                        if len(engine_pids) != 1:
                            memory_failure = f"Expected exactly one model process, found {engine_pids}"
                        if used > memory_limit_mib:
                            memory_failure = f"HBM {used} MiB exceeds {memory_limit_mib} MiB"
                    if command[0] == "ps":
                        value = "\n".join(line for line in value.splitlines() if any(
                            key in line for key in ("qwen_asr", "EngineCore", "spawn_main", "benchmark_asr")))
                    sample[command[0]] = value
                with (output / "resources.jsonl").open("a") as f:
                    f.write(json.dumps(sample) + "\n")
                if memory_failure:
                    measurement.cancel()
                    return
                await asyncio.sleep(1)
        monitor = asyncio.create_task(sample_resources())
        try:
            await asyncio.wait_for(asyncio.gather(*(c.channel_ready() for c in channels)), 30)
            for item in inputs:
                # Every worker receives warmup traffic; exclude it from measurement.
                for _ in range(warmup):
                    batch = await run_round(stubs[:workers], item, sample_rate, 1, timeout)
                    with (output / "warmup.jsonl").open("a") as f:
                        f.write(json.dumps({"input": item["name"], **batch}, ensure_ascii=False) + "\n")
                    if batch["errors"]:
                        raise RuntimeError(f"Warmup failed: {batch}")
                for concurrent in clients:
                    case_rounds = []
                    for round_index in range(rounds):
                        count = max(requests_per_client, (min_requests + concurrent - 1) // concurrent)
                        result = await run_round(stubs[:concurrent], item, sample_rate, count, timeout)
                        record = {"workers": workers, "input": item["name"], "clients": concurrent,
                                  "round": round_index + 1, **result}
                        with (output / "raw.jsonl").open("a") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        case_rounds.append(result)
                        typer.echo(f"workers={workers} duration={item['duration_s']:.3f}s clients={concurrent} round={round_index+1} speed={result['throughput_x']:.2f}x errors={result['errors']}")
                    requests = [r for row in case_rounds for r in row["requests"]]
                    row = {"workers": workers, "duration_s": item["duration_s"], "clients": concurrent,
                           "throughput_x_median": float(np.median([r["throughput_x"] for r in case_rounds])),
                           "throughput_x_min": min(r["throughput_x"] for r in case_rounds),
                           "throughput_x_max": max(r["throughput_x"] for r in case_rounds),
                           "success": sum(r["success"] for r in case_rounds),
                           "errors": sum(r["errors"] for r in case_rounds)}
                    for metric in ("first_packet_s", "first_text_s", "latency_s"):
                        values = [r[metric] for r in requests if r[metric] is not None and not r["error"]]
                        for p in (50, 95, 99):
                            row[f"{metric}_p{p}"] = percentile(values, p)
                    summaries.append(row)
                    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
                    with (output / "summary.csv").open("w") as f:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerows(summaries)
            return summaries
        except asyncio.CancelledError:
            if memory_failure:
                raise RuntimeError(memory_failure) from None
            raise
        finally:
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor
            await asyncio.gather(*(c.close() for c in channels))

    summaries = asyncio.run(run())
    if any(row["errors"] for row in summaries):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
