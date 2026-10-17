"""TLS 双协议、实时发送、EOF 排空和取消后容量恢复的集成验证。"""

import argparse
import asyncio
import json
from pathlib import Path
import time

from benchmark_ai_meeting import inputs, session
from setup_ai_meeting import api


async def run(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "smoke.json").exists() or (output / "cancellation.json").exists():
        raise ValueError("Use a fresh output directory")
    source = inputs(Path("/root/repositories/qwen-asr/data-bin/rita.wav"), [30])
    before = api("/api/v1/runtime/status")
    records = []
    for protocol, speed in (("grpc", 1), ("ws", 10)):
        for item in source:
            result = await session(item, speed, protocol)
            records.append(result)
            print(json.dumps({"protocol": protocol, "speed": speed, "duration": item["duration_s"],
                              "error": result["error"], "transcript": result["transcript"]}, ensure_ascii=False), flush=True)
            (output / "smoke.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    cancelled = await session(source[-1], 1, cancel=True)
    started = time.perf_counter()
    snapshots = []
    while time.perf_counter() - started < 15:
        status = api("/api/v1/runtime/status")
        snapshots.append(status)
        if status["realtime"]["active"] == before["realtime"]["active"] and status["asr"]["active"] == before["asr"]["active"]:
            break
        await asyncio.sleep(0.2)
    else:
        raise RuntimeError("Cancellation left active sessions or ASR slots")
    recovery_s = time.perf_counter() - started
    followup = await session(source[0], 10)
    result = dict(before=before, cancelled=cancelled, recovery_s=recovery_s,
                  snapshots=snapshots, followup=followup)
    (output / "cancellation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if not cancelled.get("cancelled") or cancelled["error"] or followup["error"] or any(r["error"] for r in records):
        raise RuntimeError("Protocol smoke or cancellation validation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
