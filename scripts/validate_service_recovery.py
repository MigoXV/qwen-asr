"""终止本服务的 Engine Core，验证 systemd 回收和重新就绪。"""

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import typer

from benchmark_asr import npu_memory_mib

app = typer.Typer(add_completion=False)


def state(service):
    text = subprocess.check_output([
        "systemctl", "show", service, "-p", "MainPID", "-p", "ActiveState",
        "-p", "SubState", "-p", "NRestarts", "-p", "Result",
    ], text=True)
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


@app.command()
def main(
    output: Path = typer.Option(...),
    service: str = typer.Option("qwen-asr-ascend.service"),
    timeout: float = typer.Option(240, min=1),
    memory_limit_mib: int = typer.Option(12288, min=1),
):
    before = state(service)
    assert before["ActiveState"] == "active", before
    info = subprocess.check_output(["npu-smi", "info"], text=True)
    pids = re.findall(r"\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|\s*VLLM", info)
    assert len(pids) == 1, pids
    target = int(pids[0])
    assert service in Path(f"/proc/{target}/cgroup").read_text(), "Refusing to kill an unrelated process"
    started = time.monotonic()
    os.kill(target, signal.SIGKILL)
    samples = []
    output.parent.mkdir(parents=True, exist_ok=True)
    while time.monotonic() - started < timeout:
        info = subprocess.check_output(["npu-smi", "info"], text=True)
        model_pids = re.findall(r"\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|\s*VLLM", info)
        current = state(service)
        sample = {"elapsed_s": time.monotonic() - started, "hbm_used_mib": npu_memory_mib(info),
                  "model_pids": model_pids, "service": current}
        samples.append(sample)
        with output.with_suffix(".samples.jsonl").open("a") as stream:
            stream.write(json.dumps(sample) + "\n")
        assert len(model_pids) <= 1, sample
        assert sample["hbm_used_mib"] <= memory_limit_mib, sample
        if current["MainPID"] != before["MainPID"] and current["ActiveState"] == "active":
            assert len(model_pids) == 1 and model_pids[0] != str(target), sample
            with output.with_suffix(".probe.log").open("w") as log:
                subprocess.run(["poetry", "run", "python", "scripts/check_asr.py", "--requests", "8", "--timeout", "30"],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            result = {"passed": True, "terminated_engine_pid": target, "before": before, "after": current,
                      "recovered_s": time.monotonic() - started,
                      "hbm_peak_mib": max(item["hbm_used_mib"] for item in samples),
                      "max_model_owners": max(len(item["model_pids"]) for item in samples)}
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
            return
        time.sleep(2)
    raise TimeoutError(f"Service did not recover within {timeout} seconds")


if __name__ == "__main__":
    app()
