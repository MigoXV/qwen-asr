"""汇总隔离的诊断与消融实验，生成独立 HTML 和 JSON。"""
from __future__ import annotations

import collections
import csv
import html
import json
import statistics
from pathlib import Path

import typer

app=typer.Typer(add_completion=False)


def load_lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def describe(directory, oracle):
    rounds=load_lines(directory/'rounds.jsonl')
    measured=collections.defaultdict(list)
    for row in rounds:
        if row['label'] in ('unprofiled','row_copy_ablation'):
            measured[row['label']].append(row['throughput_x'])
    resources=load_lines(directory/'resources.jsonl')
    result={'runs':{label:{'rounds':len(v),'median_x':statistics.median(v),'min_x':min(v),'max_x':max(v)}
                    for label,v in measured.items()},
            'output_token_counts':dict(collections.Counter(v['output_tokens'] for x in rounds for v in x['requests'])),
            'requests':sum(len(x['requests']) for x in rounds),
            'transcript_mismatches':sum(v['transcript']!=oracle for x in rounds for v in x['requests']),
            'hbm_peak_mib':max(x['hbm_mib'] for x in resources),
            'max_model_owners':max(len(x['model_pids']) for x in resources)}
    stages=collections.defaultdict(list)
    for row in json.loads((directory/'phases.json').read_text()):
        stages[(row['stage'],row['phase'])].append(row)
    result['phases']=[{'stage':s,'phase':p,'count':len(v),'host_total_ms':sum(x['host_ms'] for x in v),
                       'host_median_ms':statistics.median(x['host_ms'] for x in v),
                       'stream_span_total_ms':sum(x['stream_span_ms'] or 0 for x in v)}
                      for (s,p),v in stages.items()]
    copy_rows=[]
    for (stage,phase),v in stages.items():
        if phase!='tensor_copy':continue
        grouped=collections.defaultdict(list)
        for x in v:grouped[(x['source_device'],x['destination_device'],x['source_pinned'],x['bytes'],str(x['shape']))].append(x['host_ms'])
        for key,values in grouped.items():
            copy_rows.append({'stage':stage,'source':key[0],'destination':key[1],'pinned':key[2],'bytes':key[3],
                              'shape':key[4],'count':len(values),'total_ms':sum(values),'median_ms':statistics.median(values)})
    result['copies']=sorted(copy_rows,key=lambda x:-x['total_ms'])
    profiler=next((directory/'trace').glob('*/ASCEND_PROFILER_OUTPUT'))
    result['trace_path']=str(profiler.relative_to(directory))
    op=list(csv.DictReader((profiler/'op_statistic.csv').open()))
    result['kernel_count']=sum(int(x['Count']) for x in op)
    result['top_ops']=sorted(op,key=lambda x:-float(x['Total Time(us)']))[:20]
    result['api']=sorted(csv.DictReader((profiler/'api_statistic.csv').open()),key=lambda x:-float(x['Time(us)']))[:20]
    result['device_stage']=list(csv.DictReader((profiler/'step_trace_time.csv').open()))[0]
    result['microbench']=json.loads((directory/'microbench.json').read_text())
    return result


def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


@app.command()
def main(root:Path=typer.Argument(Path('tmp-workspace/2026-10-14/hotspot-diagnosis'))):
    oracle=load_lines(root/'engine-8/rounds.jsonl')[2]['requests'][0]['transcript']
    names=['engine-4','engine-8','engine-4-linear-rope','engine-8-linear-rope']
    data={name:describe(root/name,oracle) for name in names}
    budgets=load_lines(root/'budget-resources.jsonl')
    overall={'hbm_peak_mib':max(x['hbm_mib'] for x in budgets),
             'max_model_owners':max(len(x['model_pids']) for x in budgets)}
    assert overall['hbm_peak_mib']<=12288 and overall['max_model_owners']==1
    assert all(not x['transcript_mismatches'] for x in data.values())
    summary={'hardware':'Ascend 910B4-1 64 GiB; dual Kunpeng 920, 64 cores, 2.6 GHz',
             'scope':'one model; BF16; PyTorch/torch-npu 2.8.0; CANN 8.3.RC1; vLLM Ascend 0.13.0rc1',
             'audio':'rita looped to 30s, PCM16 mono 16000Hz',
             'client_concurrency':32,'grpc_capacity':32,'cache':'prefix off; multimodal processor off; cross-request encoder reuse off',
             'a100_user_reported_x':{'4':272,'8':388},'overall':overall,'experiments':data}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    run_rows=[]
    for name,x in data.items():
        for label,v in x['runs'].items():
            run_rows.append([name,label,f"{v['median_x']:.2f}",f"{v['min_x']:.2f}–{v['max_x']:.2f}",
                             x['output_token_counts'],x['transcript_mismatches'],f"{x['hbm_peak_mib']/1024:.2f}"])
    variants=table(['实验','阶段','吞吐倍速中位数','三轮范围','每请求输出token数/次数','逐字不一致','显存峰值GiB'],run_rows)
    baseline=data['engine-8'];candidate=data['engine-8-linear-rope']
    speedup=candidate['runs']['unprofiled']['median_x']/baseline['runs']['unprofiled']['median_x']
    device_rows=[]
    for name,x in data.items():
        s=x['device_stage'];stage=float(s['Stage']);compute=float(s['Computing']);free=float(s['Free'])
        device_rows.append([name,x['kernel_count'],f'{stage/1000:.2f}',f'{compute/1000:.2f}',f'{free/1000:.2f}',f'{100*compute/stage:.1f}%'])
    device=table(['实验','设备任务数','设备阶段ms','Computing ms','Free ms','计算覆盖'],device_rows)
    phases=table(['阶段','范围（嵌套，不可相加）','次数','Host总ms','Host中位数ms','NPU stream span总ms'],
                 [[x['stage'],x['phase'],x['count'],f"{x['host_total_ms']:.2f}",f"{x['host_median_ms']:.3f}",f"{x['stream_span_total_ms']:.2f}"] for x in baseline['phases']])
    copies=table(['阶段','方向','固定内存','字节数','shape','次数','Host总ms','Host中位数ms'],
                 [[x['stage'],x['source']+' → '+x['destination'],x['pinned'],x['bytes'],x['shape'],x['count'],f"{x['total_ms']:.2f}",f"{x['median_ms']:.3f}"] for x in data['engine-4']['copies'][:12]])
    operators=table(['算子类型','次数','Device累计ms','单次均值us'],
                    [[x['OP Type'],x['Count'],f"{float(x['Total Time(us)'])/1000:.2f}",x['Avg Time(us)']] for x in baseline['top_ops'][:14]])
    apis=table(['Host API','次数','累计ms（跨线程可重叠）','均值us'],
               [[x['API Name'],x['Count'],f"{float(x['Time(us)'])/1000:.2f}",x['Avg(us)']] for x in baseline['api'][:12]])
    links=''.join(f'<li>{name}: <a href="{name}/rounds.jsonl">每请求输出与逐轮吞吐</a> · <a href="{name}/phases.json">分段原始数据</a> · <a href="{name}/{x["trace_path"]}/trace_view.json">CPU/NPU时间线</a> · <a href="{name}/{x["trace_path"]}/kernel_details.csv">设备算子明细</a></li>' for name,x in data.items())
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-ASR Ascend 热点定位与因果对照</title><style>body{{font:15px/1.7 system-ui,sans-serif;color:#20242a;margin:30px;background:#fafafa}}main{{max-width:1600px;margin:auto}}table{{border-collapse:collapse;background:#fff;font-variant-numeric:tabular-nums;max-width:100%}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}}th{{background:#edf0f3}}h1{{font-size:27px}}h2{{margin-top:32px;font-size:21px}}pre{{white-space:pre-wrap;background:#edf0f3;padding:15px}}.scroll{{overflow:auto}}a{{color:#1859a5}}</style><main>
<h1>Qwen3-ASR · Ascend 热点定位与因果对照</h1>
<p>日期：2026-10-14。单份权重、单 Engine Core、BF16、12 GiB 上限。诊断与对照代码全部位于 scripts；src 未修改。</p>
<h2>结论与证据</h2>
<p><strong>主要瓶颈是当前适配没有走融合 RoPE，产生大量小算子及执行空泡；不是 gRPC 吞吐不足，也没有证据支持芯片计算或 HBM 带宽已经打满。</strong>
在同一 CPU、NPU、权重、缓存策略和输出工作量下，仅将 ASR 等轴位置编码映射到融合 RoPE，8 路吞吐提升 {speedup:.2f} 倍。</p>
<p>ASR 模型的 MRoPE section 为 [24,20,20]；当前 AscendMRotaryEmbedding 快速分支要求 [16,24,24]，因此进入通用实现。
项目的 get_mrope_input_positions 对音频和文本都构造三个完全相同的坐标轴，实验据此使用一维位置和 npu_mrope 的线性分支。
NPU 数值检查覆盖 4/8/405 token，BF16 最大绝对误差不超过 0.03125、余弦相似度大于 0.99999；完整请求输出逐字核对结果见下表。</p>
<p>CPU 参与瓶颈，但更多时间在 CANN/Torch NPU 同步、图参数更新和布局拷贝路径。主 Engine 线程一次稳态 perf 采样约占 0.526 个 CPU，频率 2.58 GHz；没有 CPU 配额限制。
全进程 CPU 采样中约 47.8% 样本为 pthread_rwlock_unlock，栈连至 halSqCqQuery 和事件同步；这是 on-CPU 采样比例，不能当成整轮 wall time 占比。</p>
<h2>30 秒音频、32 客户端：未开启 profiler 的三轮结果</h2><div class="scroll">{variants}</div>
<p>所有实验都关闭前缀、多模态处理及跨请求声学编码缓存，temperature=0，max_new_tokens=256，max_model_len=1024，max_num_batched_tokens=4096，ACL FULL_DECODE_ONLY、异步调度。
每请求实际输出固定为 80 token。没有增加副本，没有更换硬件或 dtype。row_copy_ablation 在同一模型实例中切换二维位置拷贝方式。
A100 的 272×/388× 为用户提供的数据，已确认同为 30 秒、32 客户端；尚无 A100 trace 和完整运行配置，不能把剩余差距精确分摊到 CPU、芯片或软件版本。</p>
<h2>设备时间线</h2><div class="scroll">{device}</div>
<p>详细 profiler 每次采集一组 engine 槽位数的请求（4 或 8），保持真实 batch；32 客户端吞吐另行测量。
Computing/Free 来自 Ascend step_trace_time。额外 NPU event、Python cProfile 和 Torch profiler 会引入开销，不能把采样轮次当作稳态吞吐。
默认 8 路 trace 的约 2.69 秒中，约 0.87 秒计算、1.82 秒 Free；113,470 个设备任务经融合后降至 38,101，减少约 66%。</p>
<h2>Host API 与拷贝热点</h2><div class="scroll">{apis}</div><div class="scroll">{copies}</div>
<p>4 路位置元数据仅 96 字节、来自固定内存，但 [3,4] 的二维非连续 copy_ 中位数约 9.3 ms；其他小元数据大多几十微秒。
这包含等待已有设备工作及布局处理的时间，不能解释为传输 96 字节需要 9 ms，更不能按该数估算 PCIe 带宽。
在融合 RoPE 的同一模型实例中，连续行拷贝仅使吞吐从 154.73× 变为 155.41×，约 0.44%，未见显著收益。
因此，长 copy 的时间主要是同步等待的表现，不能把这段时间直接当成独立可消除的拷贝成本。
8 路长 memcpy API 区间约 1.42 秒，其中约 0.95 秒未与任何设备 kernel 重叠；API 和事件等待跨线程重叠，不能相加。</p>
<h2>分段耗时：默认 8 路，32 请求</h2><div class="scroll">{phases}</div>
<p>audio_encoder 包含在 _execute_mm_encoder，后者包含在 _prepare_inputs；graph_attention_update 包含在 _generate_process_reqs_hidden_states，最终包含在 execute_model。
Host 包含调度和阻塞等待；stream span 也可能包含设备空闲。真正设备忙时依据 kernel 时间线，而非相加这些嵌套范围。</p>
<h2>默认路径的设备算子热点</h2><div class="scroll">{operators}</div>
<p>连续拷贝校准约 1.26 TB/s；较大 BF16 矩阵校准约 153 TFLOPS。小 M=4/8 的矩阵有效算力低于大矩阵，这是小 batch 执行效率问题，不能拿理论峰值直接预测该模型吞吐。
校准复用已加载权重，只分配小型临时张量；不是第二份模型，也不是芯片峰值认证。</p>
<h2>定位后的处理顺序与证据边界</h2><ol>
<li>首先修正 ASR RoPE 的融合路径选择。因果对照已证明其显著收益，不能仅靠调 CPU 线程或增加设备并发代替。</li>
<li>检查不连续位置拷贝的同步路径，以及每 token 逐层更新 attention 图参数的 CPU/Runtime 开销；行拷贝对照只改变相同数据的复制方式。</li>
<li>进一步优化 RMSNorm、FIA attention 和小矩阵的融合与调度。只有两端相同 profiler 数据才能判断剩余差距中硬件能力的份额。</li>
</ol><p>相对用户给出的 A100 吞吐，融合 RoPE 后剩余差距约为 4 路 1.76 倍、8 路 1.49 倍。
当前实验不能将该余量全归给硬件。A100 的 CPU、实际缓存、vLLM 版本和算子 trace 尚未拿到。
本轮为定位与消融实验，没有向生产 src 部署诊断补丁。系统服务恢复结果记录在 final-state.json。</p>
<h2>复现与原始数据</h2><ul>{links}</ul>
<p><a href="summary.json">结构化汇总</a> · <a href="commands.txt">命令与配置</a> · <a href="budget-resources.jsonl">单模型与显存监控</a> · <a href="perf-report-8.txt">原服务 CPU 调用栈</a> · <a href="perf-stat-main-4.log">4 路主线程 CPU 计数</a></p>
<p>方法参考：<a href="https://docs.vllm.ai/projects/ascend/en/v0.13.0/developer_guide/feature_guide/ACL_Graph.html">Ascend ACL Graph</a> · <a href="https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html">官方 profiling 指南</a>。实际使用本机 0.13.0rc1 的 env profiling 接口，以安装源码为准。</p></main></html>'''
    (root/'report.html').write_text(body)
    print(root/'report.html')


if __name__=='__main__':app()
