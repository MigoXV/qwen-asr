"""汇总正式融合部署与本机会议工作流测试，生成可离线查看的报告。"""

import argparse
import html
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    ) + "</tbody></table>"


def main(root):
    asr = load(root / "asr-benchmark/summary.json")
    workflow = load(root / "harness-benchmark/summary.json")
    post_fix = load(root / "harness-after-vad-fix/summary.json")
    resources = jsonl(root / "resources.jsonl")
    smoke = load(root / "protocol-tests/smoke.json")
    cancellation = load(root / "protocol-tests/cancellation.json")
    stress = load(root / "vad-fix/cancellation-stress.json")
    vad = load(root / "vad-fix/deployment.json")
    alignment = load(root / "audio-alignment/alignment.json")
    deployment = load(root / "workflow/local-active.json")
    runtime = load(root / "deployment/runtime-final.json")
    preview_rejected = runtime["asr"]["rejected"]
    peak = max(r["hbm_mib"] for r in resources)
    owners = max(len(r["model_pids"]) for r in resources)
    errors = sum(r["errors"] for r in asr + workflow) + sum(bool(r["error"]) for r in smoke)
    if (len(asr) != 4 or len(workflow) != 24 or len(post_fix) != 2 or errors
            or any(r["errors"] for r in post_fix) or owners != 1 or peak > 16384
            or cancellation["cancelled"]["error"] or cancellation["followup"]["error"]
            or stress["final"] != stress["before"]):
        raise RuntimeError("Incomplete or failed acceptance matrix")
    summary = dict(asr=asr, workflow=workflow, workflow_after_vad_fix=post_fix, hbm_peak_mib=peak, max_model_owners=owners,
                   measured_asr_requests=sum(r["success"] for r in asr),
                   measured_workflow_sessions=sum(r["success"] for r in workflow),
                   errors=errors, cancellation_recovery_s=cancellation["recovery_s"],
                   harness_asr_preview_rejected_cumulative=preview_rejected,
                   workflow_revision=deployment["revision"], workflow_id=deployment["workflow_id"])
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    headers_asr = ["音频秒数", "客户端", "吞吐倍速", "首包 P50 / P95 (s)", "完成 P95 (s)", "错误"]
    rows_asr = [[f"{r['duration_s']:.3f}", r["clients"], f"{r['throughput_x_median']:.2f}×",
                 f"{r['first_packet_s_p50']:.3f} / {r['first_packet_s_p95']:.3f}",
                 f"{r['latency_s_p95']:.3f}", r["errors"]] for r in asr]
    headers_workflow = ["音频秒数", "客户端", "吞吐倍速", "临时转写 P50 / P95 (s)", "正式转写 P50 / P95 (s)", "完成 P95 (s)", "错误"]
    rows_workflow = [[f"{r['duration_s']:.3f}", r["clients"], f"{r['throughput_x_median']:.2f}×",
                      f"{r['first_interim_s_p50']:.3f} / {r['first_interim_s_p95']:.3f}",
                      f"{r['first_primary_s_p50']:.3f} / {r['first_primary_s_p95']:.3f}",
                      f"{r['latency_s_p95']:.3f}", r["errors"]] for r in workflow]
    intro = (
        f"正式融合服务与 ai-meeting 已部署并保持运行。测量 {summary['measured_asr_requests']} 个直接 ASR 请求、"
        f"{summary['measured_workflow_sessions']} 个工作流会话，错误 {errors}；HBM 采样峰值 {peak} MiB（{peak/1024:.2f} GiB），"
        f"单模型所有者上限 {owners}。协议冒烟另测 TLS gRPC 与 WebSocket 的原音/30 秒音频。"
    )
    config = (
        "Qwen3-ASR-1.7B；Python 3.10；PyTorch/torch-npu 2.8.0；BF16；一个 Engine Core、并发 8；"
        "ASR gRPC 并发 32、端口 50011；max_model_len=1024、max_num_batched_tokens=1024、KV=896 MiB；"
        "FULL_DECODE_ONLY、enforce_eager=false、捕获 8 档解码图；前缀缓存和多模态结果缓存关闭。"
    )
    graph = (
        "输入 → AGC → 双物理 VAD：预览 VAD → interim-asr；正式 VAD → dummy-diarize → speaker-asr。"
        "两个 ASR 节点共享 localhost:50011，VAD 为 localhost:50022。占位分离统一返回说话人 0；"
        "聚类未实现、不执行，也不连接远端服务。参考远端 ai-meeting 已发布 revision 30；本机工作流已发布，model=ai-meeting。"
    )
    methodology = (
        "直接 ASR 每次立即上传完整 PCM；32 客户端，每客户端连续 2 请求，每种音频三轮。"
        "工作流按 100 ms PCM 分块、不限速发送，客户端 1/2/4/8/16/32，每档三轮、每客户端一会话；EOF 排空至 gRPC 正常结束。"
        "吞吐为成功输入音频总秒数除以整轮墙钟耗时，取三轮中位数；工作流墙钟包含 TLS 建连。"
        "直接 ASR 首包从 RPC 开始计时；工作流延迟从第一块音频发送计时，首包可能为 VAD 事件，"
        "首条临时与正式转写单独统计。工作流包含 AGC、两路 VAD、多次预览 ASR 和正式 ASR，工作量不同于单次完整音频 ASR。"
        "音频为 rita.wav 原音 5.1299375 秒及严格裁切到 10/20/30 秒的循环变体，无测试轮次使用 profiler。"
    )
    methodology += (
        "完整工作流矩阵测于 VAD 取消清理修复前；矩阵正常结束后，取消测试才触发旧 VAD 故障。"
        "修复不改变正常推理路径，另以原音及 30 秒音频各复核一次 32 客户端负载；补充结果单列为一轮，"
        "不混入原三轮中位数。"
    )
    checks = (
        "真实 NPU 单元测试 6 项通过；回归测试运行 43 项（7 项需显式 NPU/VAD 开关而跳过），其余通过。"
        "部署日志确认 28 层融合 RoPE、8 档图捕获和 ACL Graph 回放。"
        f"取消测试后会话与 ASR 活跃计数恢复基线，观察恢复耗时 {cancellation['recovery_s']:.3f} 秒，后续转写成功。"
        "实时 gRPC（1×）与快速 WebSocket（10×）均验证 VAD 边界、正式转写和占位说话人事件。"
        "最初一次启动因模型构造签名不符合 vLLM 要求失败，已修正；工作流第一次发布因 ASR 尚未就绪失败，"
        "服务就绪后重试成功。失败尝试日志独立保存，未混入性能数据。"
    )
    checks += (
        f" Harness ASR limiter 的累计 rejected 为 {preview_rejected}；现有预览分支采用 best-effort，"
        "容量满时跳过预览块（workflows/runtime.py 的 _asr_slot 与 _asr 路径）。正式转写等待可用容量，"
        "本矩阵正式转写全部完成。该计数单独披露，不计为成功会话中的传输错误。"
    )
    checks += (
        " 初次取消集成测试发现已有 VAD 容器的清理故障：原生 cancel 已移除会话，桥接层再次 session_close 导致"
        " unknown stream session 并使容器重启。派生镜像只修复这条重复清理路径，其他原生错误仍显式失败；"
        "VAD 源仓库未改动。修复单元测试 3 项通过，修复后双协议复测通过，另做 3 轮、每轮 4 个取消会话"
        "并行于 1 个健康 30 秒会话的压力测试，12 次取消后 3 个健康会话全部完成，容器未重启。"
    )
    quality = (
        "30 秒工作流正式结果均为 141 字，直接 ASR 均为 142 字，差异是输入在词中截断处的末尾“喜”字。"
        "补充原音/AGC × 自动语言/zh-CN 的四组对照，各三次；原音两组均为 142 字，AGC 两组均为 141 字，"
        "因此在这条音频和配置下可归因于 AGC 波形处理。AGC 前后均为 480000 采样、30 秒；"
        "工作流事件覆盖 0–30000 ms，未截短输入音频；单次输出上限仍为 256 token。此处不宣称两种波形逐字一致。"
    )
    if len(alignment) != 12 or any(r["samples"] != 480000 or not r["final"] for r in alignment):
        raise RuntimeError("Incomplete audio alignment evidence")
    evidence = ["deployment/asr-startup.log", "deployment/unit-tests.log", "deployment/regression-tests.log",
                "deployment/asr.env.before", "deployment/asr.env.after", "deployment/source-sha256.json",
                "workflow/remote-active.json", "workflow/local-active.json", "workflow/placeholder-policy.json",
                "asr-benchmark/raw.jsonl", "harness-benchmark/raw.jsonl", "protocol-tests/smoke.json",
                "protocol-tests/cancellation.json", "resources.jsonl"]
    evidence += ["deployment/runtime-final.json", "deployment/benchmark_ai_meeting.executed.py", "harness-after-vad-fix/summary.json"]
    evidence += ["vad-fix/deployment.json", "vad-fix/failure.log", "vad-fix/client.py", "vad-fix/Dockerfile",
                 "vad-fix/unit-tests.log", "vad-fix/cancellation-stress.json", "audio-alignment/alignment.json"]
    body = "<h1>融合 ASR 与本机 ai-meeting 验收报告</h1>" + "".join(
        f"<p>{html.escape(p)}</p>" for p in (intro, config, graph)
    ) + "<h2>测试口径</h2><p>" + html.escape(methodology) + "</p><h2>直接 ASR</h2>" + table(headers_asr, rows_asr)
    body += "<h2>会议工作流</h2>" + table(headers_workflow, rows_workflow) + "<h2>正确性与恢复</h2><p>" + html.escape(checks) + "</p>"
    rows_post = [[f"{r['duration_s']:.3f}", r["clients"], f"{r['throughput_x_median']:.2f}×", r["success"], r["errors"]] for r in post_fix]
    body += "<h2>VAD 修复后补充复测（每种音频一轮）</h2>" + table(["音频秒数", "客户端", "吞吐倍速", "成功会话", "错误"], rows_post)
    body += "<h2>30 秒文本对齐</h2><p>" + html.escape(quality) + "</p>"
    body += "<p>VAD 派生镜像：<code>" + html.escape(vad["image_after"]) + "</code>；基于 <code>" + html.escape(vad["image_before"]) + "</code>。</p>"
    body += "<h2>原始证据</h2><ul>" + "".join(f'<li><a href="{e}">{e}</a></li>' for e in evidence) + "</ul>"
    (root / "report.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>融合 ASR 与 ai-meeting 验收</title>'
        '<style>body{font-family:system-ui,sans-serif;margin:32px auto;padding:0 24px;max-width:1300px;line-height:1.7;color:#183044}'
        'table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 12px;border:1px solid #dce3e9;text-align:left}'
        'th{background:#edf3f7}tr:nth-child(even){background:#f8fafb}a{color:#17609b}</style>' + body + '</html>'
    )
    def markdown_table(headers, rows):
        return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join(
            "| " + " | ".join(str(v) for v in row) + " |" for row in rows
        )
    (root / "report.md").write_text(
        "# 融合 ASR 与本机 ai-meeting 验收报告\n\n" + "\n\n".join((intro, config, graph))
        + "\n\n## 测试口径\n\n" + methodology + "\n\n## 直接 ASR\n\n" + markdown_table(headers_asr, rows_asr)
        + "\n\n## 会议工作流\n\n" + markdown_table(headers_workflow, rows_workflow)
        + "\n\n## VAD 修复后补充复测（每种音频一轮）\n\n" + markdown_table(["音频秒数", "客户端", "吞吐倍速", "成功会话", "错误"], rows_post)
        + "\n\n## 正确性与恢复\n\n" + checks + "\n\n## 30 秒文本对齐\n\n" + quality
        + "\n\nVAD 派生镜像：`" + vad["image_after"] + "`；基于 `" + vad["image_before"] + "`。\n\n## 原始证据\n\n"
        + "\n".join(f"- [{e}]({e})" for e in evidence) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
