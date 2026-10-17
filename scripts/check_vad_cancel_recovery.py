"""取消多个会议会话时，健康会话继续完成且 VAD 容器不重启。"""

import argparse
import asyncio
import json
from pathlib import Path
import subprocess

from benchmark_ai_meeting import inputs, session
from setup_ai_meeting import api


def container():
    data = json.loads(subprocess.check_output(["docker", "inspect", "fireredvad-ncnn-arm64"]))[0]
    return {"id": data["Id"], "restart_count": data["RestartCount"], "started_at": data["State"]["StartedAt"],
            "status": data["State"]["Status"], "image": data["Image"]}


async def run(output):
    if output.exists():
        raise ValueError("Use a new output file")
    audio = inputs(Path("/root/repositories/qwen-asr/data-bin/rita.wav"), [30])[-1]
    before = container()
    baseline = api("/api/v1/runtime/status")
    records = []
    for repeat in range(3):
        results = await asyncio.gather(session(audio, 10), *(session(audio, 1, cancel=True) for _ in range(4)))
        records.append({"round": repeat, "healthy": results[0], "cancelled": results[1:], "container": container()})
        output.write_text(json.dumps(dict(before=before, baseline=baseline, rounds=records), ensure_ascii=False, indent=2))
        if any(r["error"] for r in results) or not all(r.get("cancelled") for r in results[1:]):
            raise RuntimeError("Cancellation interrupted a healthy meeting session")
    for _ in range(75):
        after = api("/api/v1/runtime/status")
        if after["asr"]["active"] == baseline["asr"]["active"] and after["realtime"]["active"] == baseline["realtime"]["active"]:
            break
        await asyncio.sleep(0.2)
    else:
        raise RuntimeError("Cancellation left occupied slots")
    final = container()
    if final != before:
        raise RuntimeError(f"VAD container state changed: {before} -> {final}")
    output.write_text(json.dumps(dict(before=before, baseline=baseline, rounds=records, final=final, runtime_after=after), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
