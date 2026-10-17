"""汇总 Worker 分段、Python 栈和真实 NPU kernel 时间线，保留统计口径。"""

import collections
import csv
import json
import marshal
import pstats
from pathlib import Path
import statistics

import typer

app = typer.Typer(add_completion=False)


def union_us(intervals):
    total = 0.0
    end = float("-inf")
    for left, right in sorted(intervals):
        total += max(0.0, right - max(left, end))
        end = max(end, right)
    return total


@app.command()
def main(directory: Path = typer.Argument(...), output: Path = typer.Option(...)):
    result = {"directory": str(directory.resolve()), "phases": {}, "python": {}, "traces": []}
    for path in sorted(directory.glob("*.json")):
        if "phase" not in path.stem and not path.stem.startswith("short-"):
            continue
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows or "phase" not in rows[0]:
            continue
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row.get("stage", "unknown"), row["phase"]].append(row)
        result["phases"][path.name] = [
            {"stage": stage, "phase": phase, "calls": len(values),
             "host_total_ms": sum(r["host_ms"] for r in values),
             "host_median_ms": statistics.median(r["host_ms"] for r in values),
             "thread_cpu_total_ms": sum(r["thread_cpu_ms"] for r in values),
             "batches": dict(collections.Counter(r.get("batch") for r in values))}
            for (stage, phase), values in groups.items()
        ]
    for path in sorted(directory.glob("*.pstats")):
        with path.open("rb") as stream:
            if not marshal.load(stream):
                continue  # Event-only measurements have no Python samples.
        stats = pstats.Stats(str(path))
        result["python"][path.name] = [
            {"file": key[0], "line": key[1], "function": key[2],
             "primitive_calls": value[0], "calls": value[1],
             "self_s": value[2], "cumulative_s": value[3]}
            for key, value in sorted(stats.stats.items(), key=lambda item: -item[1][2])[:40]
        ]
    for path in sorted(directory.rglob("trace_view.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            result.setdefault("incomplete_traces", []).append(str(path))
            continue  # A live profiler may still be writing its newest trace.
        events = data.get("traceEvents", data) if isinstance(data, dict) else data
        hardware_pids = {e["pid"] for e in events if e.get("name") == "process_name"
                         and e.get("args", {}).get("name") == "Ascend Hardware"}
        compute_types = {"AI_CORE", "AI_VECTOR_CORE", "AI_CPU", "MIX_AIC", "MIX_AIV"}
        kernels = [e for e in events if e.get("ph") == "X" and (
            e.get("cat") == "kernel" or (e.get("pid") in hardware_pids
            and e.get("args", {}).get("Task Type") in compute_types))]
        intervals = [(float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in kernels]
        busy = union_us(intervals)
        window = max((r for _, r in intervals), default=0) - min((l for l, _ in intervals), default=0)
        grouped = collections.defaultdict(list)
        for event in kernels:
            grouped[event["name"]].append(event["dur"])
        trace = {"path": str(path), "kernel_count": len(kernels),
                 "kernel_window_ms": window / 1000, "kernel_busy_union_ms": busy / 1000,
                 "kernel_idle_in_window_ms": (window - busy) / 1000,
                 "busy_fraction": busy / window if window else None,
                 "categories": dict(collections.Counter(e.get("cat") for e in events)),
                 "top_kernels": sorted([
                     {"name": key, "calls": len(values), "total_us": sum(values),
                      "median_us": statistics.median(values)} for key, values in grouped.items()
                 ], key=lambda r: -r["total_us"])[:30]}
        for name in ("op_statistic.csv", "api_statistic.csv", "step_trace_time.csv"):
            table = path.parent / name
            if table.exists():
                rows = list(csv.DictReader(table.open()))
                time_column = next((key for key in ("Total Time(us)", "Time(us)")
                                    if rows and key in rows[0]), None)
                trace[name] = sorted(rows, key=lambda r: -float(r[time_column]))[:30] if time_column else rows
        result["traces"].append(trace)
    result["definitions"] = {
        "host": "Includes dispatch and waits. Nested phases overlap; do not add them.",
        "kernel": "Union of kernel intervals across streams within first-to-last kernel window; excludes work invisible to profiler.",
        "events": "NPU event spans include stream idle time and are not device busy time.",
        "profile": "All instrumented data is diagnostic; use separate unprofiled RPC rounds for performance claims.",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    typer.echo(f"Wrote {output}: {len(result['traces'])} traces")


if __name__ == "__main__":
    app()
