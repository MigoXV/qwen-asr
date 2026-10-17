"""在 Audio Harness 的 Poetry 环境运行会议工作流的协议与并发测试。"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import ssl
import time
import wave

import grpc
import numpy as np
import websockets

from audio_harness.serving.aha_grpc.client import AhaGrpcClient

CERT = Path("/root/repositories/audio-harness/model-bin/realtime-tls/cert.pem")
PRIMARY = "conversation.item.input_audio_transcription.completed"
INTERIM = "x_aha.input_audio_transcription.interim"


def inputs(path, durations):
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16000:
            raise ValueError("Require 16 kHz mono PCM16 WAV")
        pcm = source.readframes(source.getnframes())
    result = []
    for duration in [len(pcm) / 32000, *durations]:
        if not 0 < duration <= 30:
            raise ValueError("Audio must be at most 30 seconds")
        count = round(duration * 16000) * 2
        audio = (pcm * ((count + len(pcm) - 1) // len(pcm)))[:count]
        if any(i["pcm"] == audio for i in result):
            continue
        result.append(dict(duration_s=count / 32000, pcm=audio, sha256=hashlib.sha256(audio).hexdigest()))
    return result


def update():
    return {"type": "session.update", "session": {
        "type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 16000},
            "transcription": {"model": "ai-meeting"},
            "turn_detection": {"type": "server_vad"},
        }},
    }}


async def session(item, speed, protocol="grpc", cancel=False):
    configured = asyncio.Event()
    created = asyncio.Event()
    result = dict(protocol=protocol, send_speed=speed, duration_s=item["duration_s"],
                  events=[], error=None, first_interim_s=None, first_primary_s=None,
                  first_text_s=None, first_packet_s=None, transcript="")
    started = time.perf_counter()
    audio_start = None
    last_event = started
    counts = Counter()

    def consume(event):
        nonlocal last_event
        now = time.perf_counter()
        last_event = now
        kind = event["type"]
        counts[kind] += 1
        result["events"].append({"elapsed_s": now - started, **event})
        if kind == "session.created":
            created.set()
        if kind == "session.updated":
            configured.set()
        if kind == "error" or kind.endswith(".failed"):
            raise RuntimeError(json.dumps(event, ensure_ascii=False))
        if audio_start is None:
            return
        elapsed = now - audio_start
        if result["first_packet_s"] is None:
            result["first_packet_s"] = elapsed
        if kind == PRIMARY:
            text = event.get("transcript", "")
            result["transcript"] += text
            if result["first_primary_s"] is None and text:
                result["first_primary_s"] = elapsed
        if kind == INTERIM and event.get("transcript") and result["first_interim_s"] is None:
            result["first_interim_s"] = elapsed
        if (event.get("delta") or event.get("transcript")) and result["first_text_s"] is None:
            result["first_text_s"] = elapsed

    async def transmit(send, finish):
        nonlocal audio_start
        await asyncio.wait_for(created.wait(), 30)
        await send(update())
        await asyncio.wait_for(configured.wait(), 30)
        audio_start = time.perf_counter()
        chunk = 3200  # 100 ms of PCM; no padding or waveform modification.
        for offset in range(0, len(item["pcm"]), chunk):
            await send({"type": "input_audio_buffer.append", "audio": item["pcm"][offset:offset + chunk]})
            if cancel and offset >= chunk * 24:
                return
            if speed:
                target = audio_start + min((offset + chunk) / 32000, item["duration_s"]) / speed
                await asyncio.sleep(max(0, target - time.perf_counter()))
        result["upload_s"] = time.perf_counter() - audio_start
        await finish()

    async def run_grpc():
        credentials = grpc.ssl_channel_credentials(CERT.read_bytes())
        async with grpc.aio.secure_channel("localhost:10001", credentials) as channel:
            connection = AhaGrpcClient(channel).connect(intent="transcription", timeout=300)
            async with connection:
                async def receive():
                    async for event in connection.events():
                        consume(event)
                receiver = asyncio.create_task(receive())
                sender = asyncio.create_task(transmit(connection.send, connection.finish_input))
                try:
                    if cancel:
                        await sender
                        connection.call.cancel()
                        receiver.cancel()
                        await asyncio.gather(receiver, return_exceptions=True)
                        result["cancelled"] = True
                    else:
                        await asyncio.gather(receiver, sender)
                finally:
                    sender.cancel()
                    receiver.cancel()
                    await asyncio.gather(sender, receiver, return_exceptions=True)

    async def run_ws():
        context = ssl.create_default_context(cafile=str(CERT))
        async with websockets.connect(
            "wss://localhost:10000/v1/realtime?intent=transcription&x_aha=v1",
            ssl=context, max_size=8 * 1024 * 1024,
        ) as socket:
            async def send(event):
                if isinstance(event.get("audio"), bytes):
                    event = {**event, "audio": base64.b64encode(event["audio"]).decode()}
                await socket.send(json.dumps(event))
            async def finish():
                await send({"type": "input_audio_buffer.commit"})
            async def receive():
                async for raw in socket:
                    consume(json.loads(raw))
            receiver = asyncio.create_task(receive())
            sender = asyncio.create_task(transmit(send, finish))
            try:
                await sender
                while result["first_primary_s"] is None or time.perf_counter() - last_event < 2:
                    if receiver.done():
                        await receiver
                        break
                    await asyncio.sleep(0.05)
            finally:
                sender.cancel()
                receiver.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)

    try:
        await asyncio.wait_for(run_grpc() if protocol == "grpc" else run_ws(), 300)
        if not cancel:
            if not result["transcript"].strip():
                raise RuntimeError("Missing final primary transcript")
            for kind in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
                         "x_aha.diarization.local.completed"):
                if not counts[kind]:
                    raise RuntimeError(f"Missing expected event: {kind}")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["event_counts"] = dict(counts)
    result["latency_s"] = time.perf_counter() - (audio_start or started)
    result["connection_elapsed_s"] = time.perf_counter() - started
    return result


async def matrix(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "raw.jsonl").exists():
        raise ValueError("Use a fresh output directory")
    variants = inputs(args.audio, args.duration)
    metadata = vars(args).copy()
    metadata.update(audio=str(args.audio), output=str(output), model="ai-meeting", engine_slots=8,
                    grpc_asr_capacity=32, dtype="bfloat16", model_owners=1,
                    inputs=[{k: v for k, v in i.items() if k != "pcm"} for i in variants],
                    definition="throughput_x = successful input audio seconds / round wall time (includes TLS handshake); first_* from first audio send; 100ms PCM chunks; send_speed=0 means unpaced")
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    rows = []
    with (output / "raw.jsonl").open("w") as raw:
        for item in variants:
            warmup = await session(item, args.speed)
            if warmup["error"]:
                raise RuntimeError(f"Warmup failed: {warmup['error']}")
            for clients in args.clients:
                requests = []
                throughputs = []
                for repeat in range(args.rounds):
                    started = time.perf_counter()
                    records = await asyncio.gather(*(session(item, args.speed) for _ in range(clients)))
                    elapsed = time.perf_counter() - started
                    success = sum(r["error"] is None for r in records)
                    throughputs.append(success * item["duration_s"] / elapsed)
                    raw.write(json.dumps(dict(duration_s=item["duration_s"], clients=clients, round=repeat,
                                              wall_s=elapsed, throughput_x=throughputs[-1], requests=records), ensure_ascii=False) + "\n")
                    raw.flush()
                    requests.extend(records)
                    print(json.dumps(dict(duration=item["duration_s"], clients=clients, round=repeat,
                                          throughput_x=throughputs[-1], errors=clients-success)), flush=True)
                row = dict(duration_s=item["duration_s"], clients=clients,
                           throughput_x_median=float(np.median(throughputs)),
                           success=sum(r["error"] is None for r in requests),
                           errors=sum(r["error"] is not None for r in requests))
                for metric in ("first_packet_s", "first_interim_s", "first_primary_s", "first_text_s", "latency_s"):
                    values = [r[metric] for r in requests if r["error"] is None and r[metric] is not None]
                    for p in (50, 95, 99):
                        row[f"{metric}_p{p}"] = float(np.percentile(values, p)) if values else None
                rows.append(row)
                (output / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    if any(row["errors"] for row in rows):
        raise RuntimeError("Workflow benchmark contains failed sessions; inspect raw.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio", type=Path, default=Path("/root/repositories/qwen-asr/data-bin/rita.wav"))
    parser.add_argument("--duration", type=float, nargs="*", default=[10, 20, 30])
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--speed", type=float, default=0)
    args = parser.parse_args()
    if args.rounds < 1 or args.speed < 0 or any(c < 1 or c > 32 for c in args.clients):
        parser.error("rounds>=1, speed>=0, clients in 1..32 required")
    asyncio.run(matrix(args))


if __name__ == "__main__":
    main()
