"""从原始基准文件生成可筛选的独立 HTML 报告及合并 CSV。"""

import csv
import html
import json
from collections import Counter
from pathlib import Path

import typer

from benchmark_asr import npu_memory_mib

app = typer.Typer(add_completion=False)


@app.command()
def main(
    root: Path = typer.Argument(Path("tmp-workspace/2026-10-14")),
    baseline: Path | None = typer.Option(None, help="同口径 eager 基线目录。"),
):
    environment = json.loads((root / "environment.json").read_text())
    rows = []
    resources = {}
    correctness = {}
    alignment = {}
    for slots in (4, 8):
        directory = root / f"workers-{slots}"
        values = json.loads((directory / "summary.json").read_text())
        if len(values) != 24:
            raise ValueError(f"Incomplete matrix for engine concurrency {slots}: {len(values)}/24")
        rows.extend(values)
        samples = [json.loads(line) for line in (directory / "resources.jsonl").read_text().splitlines()]
        peak = max(npu_memory_mib(sample["npu-smi"]) for sample in samples)
        owners = sorted({len(sample["model_pids"]) for sample in samples})
        if peak > environment["memory_limit_mib"] or owners != [1]:
            raise ValueError(f"Memory/model-owner constraint violated: {peak}, {owners}")
        resources[str(slots)] = {"hbm_peak_mib": peak, "model_owner_counts": owners,
                                 "sample_count": len(samples)}
        texts = Counter()
        nonfinal = 0
        errors = Counter()
        for line in (directory / "raw.jsonl").read_text().splitlines():
            record = json.loads(line)
            for request in record["requests"]:
                if request["error"]:
                    errors[request["error"]] += 1
                if not request["final"]:
                    nonfinal += 1
                if record["input"].startswith("rita-5.130"):
                    texts[request["transcript"]] += 1
        correctness[str(slots)] = {"original_transcripts": dict(texts), "missing_final": nonfinal,
                                   "errors": dict(errors)}
        alignment_file = root / f"alignment-workers-{slots}.json"
        if alignment_file.exists():
            alignment[str(slots)] = json.loads(alignment_file.read_text())
    rows.sort(key=lambda row: (row["duration_s"], row["clients"], row["workers"]))
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    totals = {"success": sum(r["success"] for r in rows), "errors": sum(r["errors"] for r in rows),
              "resources": resources, "correctness": correctness, "matrix_cells": len(rows)}
    if "observed_overall_hbm_peak_mib" in environment:
        totals["overall_hbm_peak_mib"] = environment["observed_overall_hbm_peak_mib"]
    if alignment:
        totals["eager_text_alignment"] = alignment
    recovery = json.loads((root / "lifecycle-recovery-8.json").read_text())
    if not recovery["passed"]:
        raise ValueError("Service recovery validation failed")
    totals["lifecycle_recovery"] = recovery
    mixed = {}
    for slots in (4, 8):
        path = root / f"mixed-length-workers-{slots}.json"
        if path.exists():
            check = json.loads(path.read_text())
            if not check["passed"]:
                raise ValueError(f"Mixed-length alignment failed: {slots}")
            mixed[str(slots)] = {"passed": True, "requests": len(check["requests"])}
    if mixed:
        totals["mixed_length_validation"] = mixed
    (root / "result.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2))

    comparison_html = ""
    if baseline is not None:
        old = {(r["workers"], r["duration_s"], r["clients"]): r
               for r in json.loads((baseline / "summary.json").read_text())}
        comparison = []
        for row in rows:
            previous = old[(row["workers"], row["duration_s"], row["clients"])]
            comparison.append({
                "workers": row["workers"], "duration_s": row["duration_s"], "clients": row["clients"],
                "eager_throughput_x": previous["throughput_x_median"],
                "graph_throughput_x": row["throughput_x_median"],
                "speedup": row["throughput_x_median"] / previous["throughput_x_median"],
                "eager_first_packet_p95_s": previous["first_packet_s_p95"],
                "graph_first_packet_p95_s": row["first_packet_s_p95"],
            })
        with (root / "comparison.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        comparison_html = '<h2>与原 eager 基线对比：32 客户端</h2><table><tr><th>引擎并发</th><th>音频秒数</th><th>eager 吞吐</th><th>图模式吞吐</th><th>提升</th><th>eager 首包 P95 秒</th><th>图模式首包 P95 秒</th></tr>'
        for row in comparison:
            if row["clients"] == 32:
                comparison_html += '<tr>' + ''.join(f'<td>{v}</td>' for v in (
                    row['workers'], f"{row['duration_s']:.3f}", f"{row['eager_throughput_x']:.2f}",
                    f"{row['graph_throughput_x']:.2f}", f"{row['speedup']:.2f}×",
                    f"{row['eager_first_packet_p95_s']:.3f}", f"{row['graph_first_packet_p95_s']:.3f}")) + '</tr>'
        comparison_html += '</table><p><a href="comparison.csv">完整 48 格对比 CSV</a>。基线与候选分时测量，提升包含执行配置差异。</p>'

    execution = ("eager" if environment["enforce_eager"] else
                 html.escape(environment.get("graph_mode", "ACL Graph")))
    tuning_html = ""
    if (root / "tuning.json").exists():
        tuning_html = '<details><summary>执行路径筛选与无效探测记录</summary><pre>' + html.escape((root / "tuning.json").read_text()) + '</pre></details>'
    alignment_html = ""
    if alignment:
        unmatched = sum(sum(case["transcripts_not_seen_in_baseline"].values())
                        for cases in alignment.values() for case in cases.values())
        alignment_html = f'<p>与 eager 基线按音频逐字核对，未在对应基线中出现的转写数：{unmatched}。<a href="alignment-workers-4.json">4 路对齐</a> · <a href="alignment-workers-8.json">8 路对齐</a>。</p>'
    if mixed:
        alignment_html += '<p>矩阵之外分别发送 16 条混合时长并发请求，全部与 eager 对应音频的输出逐字一致。<a href="mixed-length-workers-4.json">4 路混合输入</a> · <a href="mixed-length-workers-8.json">8 路混合输入</a>。</p>'
    graph_html = ""
    if not environment["enforce_eager"]:
        graph_html = '<p>完整解码图包含解码 forward 的 attention，音频编码和 prefill 在解码图之外。异步调度开启；按并发 4/8 分别捕获 1～4 / 1～8 档位，复用同一份权重。启动记录实际捕获数并检查非零，真实请求有 ACL Graph 回放日志。<a href="service-workers-4.log">4 路执行日志</a> · <a href="service-workers-8.log">8 路执行日志</a>。</p>'

    table = []
    for row in rows:
        values = [row["workers"], f"{row['duration_s']:.3f}", row["clients"],
                  f"{row['throughput_x_median']:.2f}",
                  f"{row['throughput_x_min']:.2f}–{row['throughput_x_max']:.2f}"]
        for metric in ("first_packet_s", "first_text_s", "latency_s"):
            for p in (50, 95, 99):
                value = row[f"{metric}_p{p}"]
                values.append("—" if value is None else f"{value * 1000:.1f}")
        values.extend([row["success"], row["errors"]])
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
        table.append(f'<tr data-slots="{row["workers"]}" data-duration="{row["duration_s"]:.3f}">{cells}</tr>')
    headers = ["引擎并发", "音频秒数", "客户端并发", "吞吐倍速（3轮中位数）", "吞吐最小–最大"]
    for label in ("首包", "首文本", "完成"):
        headers.extend(f"{label} P{p} ms" for p in (50, 95, 99))
    headers.extend(["成功", "失败"])
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    peak_text = "，".join(f"引擎并发 {slots}：{value['hbm_peak_mib']:,} MiB" for slots, value in resources.items())
    overall_peak_html = ""
    if "observed_overall_hbm_peak_mib" in environment:
        overall_peak_html = f"<p>含启动、预热、混合输入和故障恢复的全程采样峰值：{environment['observed_overall_hbm_peak_mib']:,} MiB；全程最多一个模型进程。</p>"
    durations = sorted({f"{row['duration_s']:.3f}" for row in rows}, key=float)
    options = "".join(f'<option value="{value}">{value} 秒</option>' for value in durations)
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2026-10-14 Qwen3-ASR NPU BF16 测试</title>
<style>
body{{font:15px/1.7 system-ui,sans-serif;color:#20242a;background:#fafafa;margin:32px}}
main{{max-width:1800px;margin:auto}}h1{{font-size:26px}}h2{{font-size:20px;margin-top:32px}}
table{{border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums;background:white}}
td,th{{padding:8px 12px;border-bottom:1px solid #e1e4e8;text-align:right}}
th{{background:#eef1f4;position:sticky;top:0}}tr:nth-child(even){{background:#f6f8fa}}
.scroll{{overflow:auto;max-height:74vh}}pre{{background:#eef1f4;padding:16px;overflow:auto}}
select{{font:inherit;padding:4px 12px;margin:8px 16px 16px 0}}a{{color:#1859a5}}
@media print{{body{{margin:10px}}.scroll{{max-height:none;overflow:visible}}select{{display:none}}}}
</style><main><h1>Qwen3-ASR · NPU + vLLM · BF16</h1>
<p>测试日期：2026-10-14。仅加载一份权重，单个 NPU Engine Core。
vLLM <code>max_num_seqs</code> 分别为 4/8；gRPC 最大并发固定为 32；端口 50011。</p>
<p><strong>{totals['success']:,} 次成功，{totals['errors']} 次失败，48 个配置格，每格 3 轮。</strong></p>
<p>性能矩阵整卡 HBM 采样峰值（含驱动）：{peak_text}。预算 12,288 MiB（12 GiB）。
资源采样全程均为一个模型进程。采样约每 1–1.5 秒一次，峰值为观测采样最大值。</p>
{overall_peak_html}
{comparison_html}
{tuning_html}
<h2>结果</h2><label>引擎并发 <select id="slots"><option value="">全部</option><option>4</option><option>8</option></select></label>
<label>音频长度 <select id="duration"><option value="">全部</option>{options}</select></label>
<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{''.join(table)}</tbody></table></div>
<h2>测量方法</h2><p>输入为 data-bin/rita.wav 原音频（5.1299375 秒），以及循环拼接并截断到
10、20、30 秒的变体；单声道 PCM16，16 kHz。客户端一次上传完整音频，启用增量转写。
首包从发起 RPC 到首个响应；首文本到首个非空转写；完成延迟到完整 RPC 结束，均含上传、排队和推理。</p>
<p>吞吐倍速 = 成功请求的音频总秒数 / 整轮墙钟时间。表中吞吐为三轮中位数，另列轮次最小–最大；
延迟分位数汇总该格全部请求。每种输入先按引擎并发预热两轮，预热单独保存且不计入结果。
每客户端每轮至少一次请求，每轮至少四次请求；低并发每格 12 次请求，分位数为该样本分布。</p>
<p>BF16、{execution}，关闭前缀和多模态处理结果缓存；上下文上限 {environment['max_model_len']} token，
调度上限 {environment['max_num_batched_tokens']} token，最大生成 {environment['max_new_tokens']} token。
4/8 并发的 KV Cache 分别为 448/896 MiB，按模型结构计算。</p>
{graph_html}
<p>本版本在同时关闭前缀和多模态处理缓存时，会用独立 request ID 为音频生成标识，
不跨请求复用 encoder 输出；重复音频也执行每请求的声学编码和文本解码。
<a href="cache-validation.txt">缓存行为核对</a>。</p>
<h2>正确性与服务验证</h2><p>所有成功请求须有非空最终转写；原音频转写及频次见 result.json。
并发 32、超限拒绝、首包后取消、deadline 和取消后的后续识别已通过真实 gRPC 校验。
此测试覆盖指定音频及拼接变体，不包含独立人工标注准确率评估。</p>
{alignment_html}
<p>各并发配置分时运行，旧引擎退出、设备资源回收后才加载下一份模型。
矩阵结束后单独终止 Engine Core，systemd 在 {recovery['recovered_s']:.1f} 秒内重新完成真实识别就绪检查；
恢复过程中最多一个模型进程，HBM 峰值 {recovery['hbm_peak_mib']:,} MiB。
该故障注入在性能矩阵之外执行。<a href="lifecycle-recovery-8.json">恢复验证记录</a>。</p>
<h2>环境与配置</h2><pre>{html.escape(json.dumps(environment, ensure_ascii=False, indent=2))}</pre>
<p>上游 vLLM Ascend 0.13.0rc1 的配套说明列出 CANN 8.3.RC2；本报告记录的是当前
CANN 8.3.RC1 + NNAL/ATB 8.3.RC1 上实际完成的测试。
<a href="https://github.com/vllm-project/vllm-ascend/blob/v0.13.0rc1/docs/source/installation.md">上游安装说明</a>。</p>
<h2>复现与原始数据</h2><pre>{html.escape((root / 'commands.txt').read_text())}</pre>
<p><a href="summary.csv">合并 CSV</a> · <a href="summary.json">合并 JSON</a> · <a href="result.json">汇总与正确性</a> ·
<a href="workers-4/raw.jsonl">4 并发原始请求</a> · <a href="workers-8/raw.jsonl">8 并发原始请求</a> ·
<a href="workers-4/resources.jsonl">4 并发资源</a> · <a href="workers-8/resources.jsonl">8 并发资源</a> ·
<a href="service-config.txt">systemd 配置</a> · <a href="environment.json">环境</a></p>
</main><script>
const slots=document.getElementById('slots'),duration=document.getElementById('duration');
function filterRows(){{document.querySelectorAll('tbody tr').forEach(row=>{{row.hidden=
 (slots.value && row.dataset.slots!==slots.value)||(duration.value && row.dataset.duration!==duration.value);}})}}
document.querySelectorAll('select').forEach(select=>select.addEventListener('change',filterRows));
</script></html>'''
    (root / "report.html").write_text(body)
    typer.echo(f"Report: {root / 'report.html'}")


if __name__ == "__main__":
    app()
