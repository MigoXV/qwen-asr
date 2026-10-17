"""汇总 1.7B 完整矩阵、隔离诊断与 A100 对照；不触碰生产源码。"""
from __future__ import annotations

from collections import Counter,defaultdict
import csv
import gzip
import html
import json
import re
from pathlib import Path
import statistics

import typer

app=typer.Typer(add_completion=False)


def lines(path):return [json.loads(x) for x in path.read_text().splitlines()]
def median(values):return statistics.median(values) if values else None
def union(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def experiment(path,device):
    raw=lines(path/'rounds.jsonl');measured=[r for r in raw if r['label']=='unprofiled']
    assert len(measured)==3,path
    speeds=[r['throughput_x'] for r in measured]
    requests=[r for row in raw for r in row['requests']]
    result={'throughput_median_x':median(speeds),'throughput_rounds_x':speeds,
        'output_tokens':dict(Counter(r['output_tokens'] for r in requests)),
        'transcripts':dict(Counter(r['transcript'] for r in requests)),
        'token_sequences':dict(Counter(str(r.get('token_ids')) for r in requests)),
        'measured_requests':sum(len(r['requests']) for r in measured),'total_requests':len(requests)}
    result['finish_reasons']=dict(Counter(r.get('finish_reason','not_collected') for r in requests))
    row_copy=[r['throughput_x'] for r in raw if r['label']=='row_copy_ablation']
    if row_copy:result['row_copy_ablation']={'median_x':median(row_copy),'rounds_x':row_copy}
    if (path/'frontend-phases.json').exists():
        frontend=json.loads((path/'frontend-phases.json').read_text())
        result['frontend']={'count':len(frontend),'host_total_ms':sum(x['host_ms'] for x in frontend),
            'host_median_ms':median([x['host_ms'] for x in frontend]),'cpu_total_ms':sum(x['thread_cpu_ms'] for x in frontend)}
    phases=defaultdict(list)
    for row in json.loads((path/'phases.json').read_text()):phases[(row['stage'],row['phase'])].append(row)
    result['phases']=[{'stage':stage,'phase':phase,'count':len(rows),
        'host_total_ms':sum(r['host_ms'] for r in rows),'host_median_ms':median([r['host_ms'] for r in rows]),
        'cpu_total_ms':sum(r.get('thread_cpu_ms',0) for r in rows),'cpu_median_ms':median([r.get('thread_cpu_ms',0) for r in rows]),
        'stream_span_total_ms':sum(r['stream_span_ms'] or 0 for r in rows)} for (stage,phase),rows in phases.items()]
    if device=='npu':
        profiler=next((path/'trace').glob('*/ASCEND_PROFILER_OUTPUT'))
        ops=list(csv.DictReader((profiler/'op_statistic.csv').open()))
        result['kernel_count']=sum(int(x['Count']) for x in ops)
        result['top_ops']=sorted(ops,key=lambda x:-float(x['Total Time(us)']))[:20]
        result['device_stage']=list(csv.DictReader((profiler/'step_trace_time.csv').open()))[0]
        result['trace']=str((profiler/'trace_view.json').relative_to(path))
    else:
        pid=json.loads((path/'worker-info.json').read_text())[0]['pid']
        trace=next((path/'trace').glob(f'*_{pid}.*.pt.trace.json.gz'))
        with gzip.open(trace,'rt') as f:events=json.load(f)['traceEvents']
        kernels=[x for x in events if x.get('cat')=='kernel']
        intervals=[(x['ts'],x['ts']+x['dur']) for x in kernels]
        span=max(b for a,b in intervals)-min(a for a,b in intervals)
        busy=union(intervals)
        result['kernel_count']=len(kernels)
        grouped=defaultdict(list)
        for x in kernels:grouped[x['name']].append(x['dur'])
        result['top_ops']=[{'name':n,'count':len(v),'total_us':sum(v)} for n,v in sorted(grouped.items(),key=lambda x:-sum(x[1]))[:20]]
        result['device_stage']={'Stage':span,'Computing':busy,'Free':span-busy}
        result['trace']=str(trace.relative_to(path))
    result['worker_info']=json.loads((path/'worker-info.json').read_text())
    result['microbench']=json.loads((path/'microbench.json').read_text())
    if (path/'perf-stat.log').exists():
        stat=(path/'perf-stat.log').read_text()
        result['cpu_perf']={key:float(re.search(pattern,stat).group(1)) for key,pattern in {
            'cpus_utilized':r'([0-9.]+) CPUs utilized','frequency_ghz':r'([0-9.]+) GHz',
            'ipc':r'([0-9.]+)  insn per cycle'}.items()}
    if (path/'perf-dso.txt').exists():
        result['perf_dso_cpu_percent']={name.strip():float(pct) for pct,name in re.findall(r'^\s*([0-9.]+)%\s+(.+)$',(path/'perf-dso.txt').read_text(),re.M)}
    return result


@app.command()
def main(root:Path=typer.Argument(Path('tmp-workspace/2026-10-14/reevaluation-1.7b'))):
    all_rows=[];text_sets={};mixed=[];validation=[];matrix_budgets=[]
    for variant in ('baseline','fused'):
        for slots in (4,8):
            directory=root/variant/f'workers-{slots}'
            rows=json.loads((directory/'matrix/summary.json').read_text())
            assert len(rows)==24 and not any(x['errors'] for x in rows)
            all_rows.extend(dict(variant=variant,**x) for x in rows)
            texts=defaultdict(Counter)
            for row in lines(directory/'matrix/raw.jsonl'):
                for req in row['requests']:
                    assert not req['error'] and req['final']
                    texts[row['input']][req['transcript']]+=1
            text_sets[(variant,slots)]=texts
            mixed.append(json.loads((directory/'mixed-length.json').read_text()))
            validation.append(json.loads((directory/'service-validation.json').read_text()))
            matrix_budgets.extend({'hbm_mib':x['hbm_used_mib'],'model_pids':x['model_pids']} for x in lines(directory/'matrix/resources.jsonl'))
    assert all(x['passed'] for x in mixed)
    alignment=[]
    for slots in (4,8):
        baseline=text_sets[('baseline',slots)]
        for name,texts in text_sets[('fused',slots)].items():
            alignment.append({'slots':slots,'input':name,'requests':sum(texts.values()),
                              'unseen_transcripts':{t:n for t,n in texts.items() if t not in baseline[name]}})
    npu={}
    for variant in ('original','linear'):
        for slots in (4,8):
            name=f'engine-{slots}'+('-linear-rope' if variant=='linear' else '')
            npu[f'{slots}-{variant}']=experiment(root/'hotspot-diagnosis'/name,'npu')
    numa=experiment(root/'hotspot-diagnosis/engine-8-linear-rope-numa0','npu')
    npu['8-linear-numa0']=numa
    cuda={}
    for variant in ('original','linear'):
        for slots in (4,8):
            cuda[f'{slots}-{variant}']=experiment(root/'a100/results'/f'engine-{slots}-{variant}','cuda')
    cuda['8-original-frontend']=experiment(root/'a100/results/engine-8-original-frontend','cuda')
    budgets=matrix_budgets+lines(root/'budget.jsonl')+lines(root/'hotspot-diagnosis/budget-resources.jsonl')
    if (root/'all-stages-budget.jsonl').exists():budgets+=lines(root/'all-stages-budget.jsonl')
    max_memory=max(x['hbm_mib'] for x in budgets);max_owners=max(len(x['model_pids']) for x in budgets)
    assert max_memory<=16384 and max_owners==1
    excluded_peak=max(x['hbm_mib'] for x in lines(root/'excluded-attempts-budget.jsonl')+lines(root/'excluded-attempts-global-budget.jsonl'))
    for platform in (npu,cuda):
        for slots in (4,8):
            old=set(platform[f'{slots}-original']['transcripts'])
            platform[f'{slots}-linear']['unseen_transcripts']=sorted(set(platform[f'{slots}-linear']['transcripts'])-old)
    numa['unseen_transcripts']=sorted(set(numa['transcripts'])-set(npu['8-original']['transcripts']))
    recovery=json.loads((root/'lifecycle-recovery-8.json').read_text())
    assert recovery['passed']
    result={'model':'Qwen3-ASR-1.7B','matrix_cells':len(all_rows),
        'measured_requests':sum(x['success'] for x in all_rows),'errors':sum(x['errors'] for x in all_rows),
        'authorized_memory_limit_mib':16384,'peak_npu_hbm_mib':max_memory,'excluded_trial_peak_hbm_mib':excluded_peak,'max_model_owners':max_owners,'matrix':all_rows,
        'alignment':alignment,'mixed_tests':mixed,'service_validation':validation,
        'recovery':recovery,'npu':npu,'cuda':cuda}
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    with (root/'matrix.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    speed_rows=[]
    for slots in (4,8):
        a,b=npu[f'{slots}-original'],npu[f'{slots}-linear']
        c,d=cuda[f'{slots}-original'],cuda[f'{slots}-linear']
        speed_rows.append([slots,f"{a['throughput_median_x']:.2f}",f"{b['throughput_median_x']:.2f}",f"{b['throughput_median_x']/a['throughput_median_x']:.2f}×",
            f"{c['throughput_median_x']:.2f}",f"{d['throughput_median_x']:.2f}",f"{d['throughput_median_x']/c['throughput_median_x']:.2f}×"])
    device_rows=[];phase_rows=[]
    for platform,data in [('NPU',npu),('CUDA',cuda)]:
        for name,x in data.items():
            s=x['device_stage'];stage=float(s['Stage']);busy=float(s['Computing'])
            device_rows.append([platform,name,x['kernel_count'],f'{stage/1000:.2f}',f'{busy/1000:.2f}',f'{float(s["Free"])/1000:.2f}',f'{100*busy/stage:.1f}%'])
            for p in x['phases']:
                if p['stage']=='decode' and p['phase'] in ('execute_model','_prepare_inputs','graph_attention_update','sample_tokens','_sample'):
                    phase_rows.append([platform,name,p['phase'],p['count'],f"{p['host_median_ms']:.3f}",f"{p['cpu_median_ms']:.3f}",f"{p['host_total_ms']:.1f}"])
    main_rows=[]
    for row in all_rows:
        if row['duration_s']==30 and row['clients']==32:
            main_rows.append([row['variant'],row['workers'],f"{row['throughput_x_median']:.2f}",
                f"{row['first_packet_s_p50']:.3f}",f"{row['first_packet_s_p95']:.3f}",f"{row['latency_s_p95']:.3f}"])
    matrix_headers=['路径','Engine','音频秒','客户端','吞吐中位×','吞吐范围×','首包P50秒','首包P95秒','首包P99秒','完成P95秒','成功','错误']
    matrix_rows=[[r['variant'],r['workers'],r['duration_s'],r['clients'],f"{r['throughput_x_median']:.2f}",
        f"{r['throughput_x_min']:.2f}–{r['throughput_x_max']:.2f}",f"{r['first_packet_s_p50']:.3f}",f"{r['first_packet_s_p95']:.3f}",
        f"{r['first_packet_s_p99']:.3f}",f"{r['latency_s_p95']:.3f}",r['success'],r['errors']] for r in all_rows]
    npu8=npu['8-original'];fused8=npu['8-linear']
    reduction=100*(1-fused8['kernel_count']/npu8['kernel_count'])
    mismatch=sum(sum(x['unseen_transcripts'].values()) for x in alignment)
    op_tables=''
    for label,values in [('NPU原路径',npu8),('NPU融合RoPE',fused8)]:
        op_tables+=f'<h3>{label}：8路设备算子</h3>'+table(['算子','次数','累计设备ms'],
            [[x['OP Type'],x['Count'],f"{float(x['Total Time(us)'])/1000:.2f}"] for x in values['top_ops'][:10]])
    numa_speedup=numa['throughput_median_x']/npu['8-linear']['throughput_median_x']
    front_rows=[[platform,name,x['frontend']['count'],f"{x['frontend']['host_total_ms']:.2f}",f"{x['frontend']['host_median_ms']:.3f}",f"{x['frontend']['cpu_total_ms']:.2f}"] for platform,data in [('NPU',npu),('CUDA',cuda)] for name,x in data.items() if 'frontend' in x]
    def phase(data,name):return next(x for x in data['phases'] if x['stage']=='decode' and x['phase']==name)
    update=phase(fused8,'graph_attention_update')
    prep_old=phase(npu8,'_prepare_inputs');prep_new=phase(fused8,'_prepare_inputs')
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen3-ASR 1.7B 同模型重新评估</title>
<style>body{{font:15px/1.75 system-ui;color:#202830;background:#fafafa;margin:30px}}main{{max-width:1600px;margin:auto}}table{{border-collapse:collapse;font-variant-numeric:tabular-nums;background:white}}th,td{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e9edf0}}.scroll{{overflow:auto}}select{{padding:6px;margin:8px}}pre{{white-space:pre-wrap}}a{{color:#1859a5}}</style><main>
<h1>Qwen3-ASR-1.7B · NPU 完整评估与 A100 同模型对照</h1>
<p>2026-10-14。BF16，PyTorch 2.8.0，单份权重、单 Engine Core、gRPC32、端口50011。初始显存预算12GiB，用户随后授权必要时可放宽到16GiB；NPU有效测量及恢复阶段峰值{max_memory/1024:.2f}GiB。
试调曾达到{excluded_peak/1024:.2f}GiB，已由监控自动终止并排除，之后压缩8路预算重新测量。诊断代码全部位于 scripts，src 未改。</p>
<h2>定位结果与处理策略</h2>
<p><strong>多个问题叠加：首先是Ascend RoPE融合路径缺口；修正后，Host单线程的Python/Runtime开销成为明显瓶颈，同时仍有设备算子差距。当前证据不支持把全部差距归给芯片裸算力。</strong></p>
<ol><li>模型数学与输入相同：相同1.7B权重和30秒完整音频；融合路径与A100均生成81个token，原路径8路少量请求为80个token（尾字波动，见输出审核）。等轴RoPE简化在A100上只有约1%的收益，在NPU上有约2.4～2.7倍直接Engine收益，主要是后端实现差异。</li>
<li>算子与图执行：NPU8路设备任务从{npu8['kernel_count']:,}降到{fused8['kernel_count']:,}。输入准备Host中位耗时从{prep_old['host_median_ms']:.2f}ms降到{prep_new['host_median_ms']:.2f}ms，而实际线程CPU时间仅从{prep_old['cpu_median_ms']:.2f}ms变为{prep_new['cpu_median_ms']:.2f}ms；被消掉的主要是等待。应首先修正ASR融合路径，再按FIA、布局转换等设备热点继续处理。</li>
<li>CPU/Runtime：无CPU配额、64核可用。稳态20秒perf中，主线程从{npu8['cpu_perf']['cpus_utilized']:.3f}核升至{fused8['cpu_perf']['cpus_utilized']:.3f}核，频率约2.58GHz；融合后on-CPU样本约37.95%在Python库，另有Torch/NPU/Runtime开销。这不是总wall time比例，也不能仅凭利用率判断CPU型号性能。</li>
<li>逐层attention图参数更新每个decode步Host中位{update['host_median_ms']:.2f}ms、线程CPU中位{update['cpu_median_ms']:.2f}ms，几乎没有设备计算。当前实现对28层分别进行任务更新、FIA描述符构造和事件记录。应优先减少Python/FFI调用、复用描述符和buffer，评估批量/原生更新；单纯增加CPU核数难以解决这条串行路径。前端音频处理另见分段表，可独立评估批量化与并行化。</li>
<li>设备与裸芯片：融合后仍有实际设备忙时与attention等算子差距，需要继续优化实现。回环GEMM/HBM校准会受缓存和算子选择影响，不能据此认证芯片峰值，也不能将剩余跨平台差距全分摊为硬件能力；版本、编译融合和图实现仍不同。</li></ol>
<h2>同硬件因果对照：30秒 / 32客户端 / 直接 Engine</h2>
{table(['Engine','NPU原路径×','NPU融合RoPE×','提升','A100原路径×','A100等轴RoPE×','提升'],speed_rows)}
<p>每项先预热，再运行三轮。未开启profiler的吞吐与下面的设备采样分开测量。
两端使用相同1.7B权重和rita音频，SHA256与远端已有报告一致；max_tokens256、temperature0、上下文1024，关闭前缀缓存、多模态处理缓存及跨请求声学编码缓存。实际输出token数和逐请求文本保留于summary.json及rounds.jsonl。</p>
<h2>外层gRPC完整链路：30秒 / 32客户端</h2>
{table(['路径','Engine','吞吐×','首包P50秒','首包P95秒','完成P95秒'],main_rows)}
<p>原音频、10/20/30秒 × 客户端1/2/4/8/16/32 × Engine4/8 × 原路径/融合路径，共{len(all_rows)}格、{result['measured_requests']}条测量请求、{result['errors']}错误。
首包从客户端发起RPC计时；完整PCM立即上传。吞吐 = 成功音频总秒数 / 整轮makespan。
融合路径相对原路径未出现过的转写共{mismatch}条。另测每实例16条混合时长并发、容量32拒绝第33条、取消、deadline及后续请求；systemd故障恢复通过，耗时{recovery['recovered_s']:.2f}秒。</p>
<p><strong>输出审核：</strong>8路原路径的204条30秒测量中，193条为142字，11条末尾少一个“喜”字、为141字；其余输入与4路原路径无此波动。融合路径全部与各音频的原路径主结果匹配，30秒结果与A100一致。
直接Engine采样也记录了原路径8路的80/81-token差异；详细设备trace这一批两端均为81-token请求。融合没有减少输出量。具体分布见<a href="transcript-variants.json">转写审核</a>，完整30秒示例见<a href="complete-30s-output.json">请求与输出样本</a>。该尾字波动的精度机制尚未单独定位。</p>
<h2>算法、CPU与设备：如何区分</h2>
<p>等轴RoPE实验没有删层、减少生成token、缩短音频或更换权重，改变的是同一位置编码数学运算的实现。
Ascend快速分支只接受[16,24,24]，本模型[24,20,20]落入通用路径；A100默认native路径经Inductor与Triton旋转核编译。
NPU 8路设备任务由{npu8['kernel_count']:,}降至{fused8['kernel_count']:,}，减少{reduction:.1f}%。具体CPU与设备证据如下，不能用单一CPU使用率或芯片理论峰值归因。</p>
{table(['设备','实验','设备kernel/任务数','设备窗口ms','计算覆盖ms','空闲ms','覆盖率'],device_rows)}
<p>NPU为Ascend step_trace_time的Computing/Free；CUDA为真实kernel区间求并集，窗口从首个kernel开始到最后一个结束。统计口径有差异，均来自profiler轮次；用于观察任务数量与空泡，不能据此精确分摊稳态跨平台差距。详细采样每次一组4/8请求，吞吐测量仍为32客户端。</p>
{table(['设备','实验','范围','decode次数','Host中位ms','线程CPU中位ms','Host总ms'],phase_rows)}
<p>线程CPU时间统计当前执行线程真正占用CPU的时间；Host还包含阻塞等待。CPU时间也可能包含Runtime轮询，不等价于Python算法计算。
各阶段存在嵌套，不能直接相加；CUDA execute_model包含sampling，NPU sampling另行调用。HBM/GEMM校准复用已加载权重，不是芯片峰值认证。
两端CPU、驱动、vLLM版本、图实现仍不同：NPU为Kunpeng920/CANN8.3.RC1/vLLM Ascend0.13.0rc1/FULL_DECODE_ONLY/异步调度/token预算4路4096、8路1024/block128；A100为Xeon6326/vLLM0.11/CUDA默认FULL_AND_PIECEWISE/token预算1024/block16。因此不能把全部余量归给裸芯片算力或CPU单核性能。</p>
{op_tables}
<p>部分CANN模板名含FP16字样；单独核对的真实MatMul输入为DT_BF16，Worker参数和配置也是BF16，详见precision-evidence.json。没有执行FP16模型性能矩阵。</p>
<h2>前端音频处理</h2>{table(['设备','实验','请求数','Host总ms','Host中位ms','主线程CPU总ms'],front_rows)}
<p>在独立32请求轮次中，仅包围 InputProcessor.process_inputs，包含真实音频特征、token化及请求构造；不启用cProfile，模型仍在单一Worker运行。单独cProfile文件frontend.pstats用于定位函数，未计入稳态吞吐。
Host总时间可能与Engine执行重叠，不能直接从端到端makespan中相减。</p>
<h2>CPU位置对照</h2><p>同一融合路径、8路Engine、32客户端、30秒音频，将父进程及其新子进程CPU亲和性限制到NPU本地NUMA0的0～31号CPU核。
默认调度吞吐{npu['8-linear']['throughput_median_x']:.2f}×，绑定后{numa['throughput_median_x']:.2f}×，比值{numa_speedup:.3f}。
这是CPU位置和调度的对照，不是更换CPU型号；没有显式迁移已有内存，所有权重及buffer均由新实例加载。实际线程亲和性及原始三轮数据见hotspot-diagnosis/engine-8-linear-rope-numa0。</p>
<p>同一NUMA实例追加连续行copy对照，原方式{numa['throughput_median_x']:.2f}×，逐行复制{numa.get('row_copy_ablation',{}).get('median_x',0):.2f}×。对照后恢复原复制方式，再采集phase与trace。</p>
<h2>完整矩阵</h2><label>路径<select id="path"><option value="">全部</option><option>baseline</option><option>fused</option></select></label><label>Engine<select id="engine"><option value="">全部</option><option>4</option><option>8</option></select></label>
<div id="matrix">{table(matrix_headers,matrix_rows)}</div>
<h2>复现与原始文件</h2><p><a href="summary.json">结构化汇总</a> · <a href="matrix.csv">完整矩阵CSV</a> · <a href="provenance.json">权重SHA256与工作区状态</a> · <a href="budget.jsonl">加载与矩阵显存监控</a> · <a href="hotspot-diagnosis/budget-resources.jsonl">诊断显存监控</a> · <a href="lifecycle-recovery-8.json">故障恢复</a> · <a href="a100/results/state.json">远端隔离与恢复状态</a></p>
<p>远端原仓库 /workspace/opusi/qwen-asr 的文件SHA256和git状态在测试前后核对。实验位于仓库外 /workspace/opusi/qwen-asr-diagnostics-20261014-1.7b，复用原Poetry环境，不安装包或修改site-packages。</p>
<p>异常与排除记录：8路4096-token试验先被旧内部预算限制拦截，调预算后在11938MiB时为长音频保留余量主动终止，完整失败/取消记录见excluded-process-budget-startup与excluded-8x4096-memory-headroom。
2048预算在30秒音频预热时出现12429MiB采样值，超过12288上限141MiB，预算监控自动终止，完整记录见excluded-8x2048-budget-violation、excluded-attempts-budget.jsonl及excluded-attempts-global-budget.jsonl；最终8路两条路径均以1024预算重新测量。
A100第一版补丁只修改forward_cuda，未命中默认forward_native；第二版原位算子触发vLLM0.11 fix_functionalization编译错误，两次均排除，原始记录分别保留在a100/excluded-unselected-path与a100/excluded-functionalization-error。最终候选采用可编译的一维native实现，并记录实际dispatch及变化后的kernel数量，无eager降级。</p>
<p>本轮融合仅通过scripts诊断Worker启用；生产src没有部署实验补丁。最终服务模型与配置见final-state.json。</p>
<script>function filter(){{const p=document.getElementById('path').value,e=document.getElementById('engine').value;document.querySelectorAll('#matrix tbody tr').forEach(r=>r.hidden=!!((p&&r.cells[0].textContent!==p)||(e&&r.cells[1].textContent!==e)))}}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',filter));</script></main></html>'''
    (root/'report.html').write_text(body)
    print(root/'report.html')


if __name__=='__main__':app()
