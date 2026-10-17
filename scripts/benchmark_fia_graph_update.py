"""独立复现 ASR 的逐层 FIA 图更新，分离 Host、队列等待和设备执行。"""
import cProfile
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch_npu
import typer

app = typer.Typer(add_completion=False)


def summary(rows):
    return {key: {'median_us': statistics.median([r[key] for r in rows]),
                  'mean_us': statistics.mean([r[key] for r in rows])}
            for key in rows[0]}


@app.command()
@torch.inference_mode()
def main(output: Path = typer.Option(..., help='独立结果目录，必须不存在'),
         steps: int = typer.Option(100, min=10, help='预热后测量的迭代数'),
         layers: int = typer.Option(28, min=1, help='每轮图更新层数，ASR 1.7B 为 28'),
         native_update: bool = typer.Option(False, help='使用已安装 torch-npu 的 C++ 融合图更新接口'),
         asr_shape: bool = typer.Option(True, help='使用真实 ASR 的 64 页 KV 和 int8 mask')):
    args = SimpleNamespace(output=output, steps=steps, layers=layers)
    available = hasattr(torch_npu._C, '_npu_fused_infer_attention_score_out_graph')
    if native_update and not available:
        raise typer.BadParameter('当前 torch-npu 没有融合图更新接口')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(731)
    torch.set_num_threads(4)
    env_names = ('PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'VIRTUAL_ENV',
                 'ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'ASCEND_RT_VISIBLE_DEVICES',
                 'ASCEND_CACHE_PATH', 'ATB_HOME_PATH', 'TASK_QUEUE_ENABLE',
                 'PER_STREAM_QUEUE_ENABLE', 'OMP_NUM_THREADS', 'CPU_AFFINITY_CONF',
                 'PYTORCH_NPU_ALLOC_CONF')
    info = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
            'env': {name: os.environ.get(name) for name in env_names},
            'affinity': sorted(os.sched_getaffinity(0)),
            'native_update': native_update, 'native_update_available': available,
            'asr_shape': asr_shape,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    lib = ctypes.CDLL(str(Path(torch_npu.__file__).parent/'lib/libtorch_npu.so'))
    try:
        getter = getattr(lib, '_Z23OpApiGetTaskQueueEnablev')
        getter.argtypes = []; getter.restype = ctypes.c_uint32
        info['runtime_task_queue_enable'] = getter()
    except AttributeError:
        info['runtime_task_queue_enable'] = 'getter not exported'
    (args.output/'info.json').write_text(json.dumps(info, indent=2))
    mask = torch.ones((2048, 2048), device='npu', dtype=torch.int8 if asr_shape else torch.bool).triu(1)
    results = []
    for batch in (1, 8):
        for length in (384,):
            query = torch.randn(batch, 16, 128, device='npu', dtype=torch.bfloat16)
            key = torch.randn(64 if asr_shape else batch*8, 128, 1024, device='npu', dtype=torch.bfloat16)
            value = torch.randn_like(key)
            table = torch.arange(batch*8, device='npu', dtype=torch.int32).view(batch, 8)
            params = dict(query=query, key=key, value=value, atten_mask=mask,
                          block_table=table, input_layout='TND', block_size=128,
                          actual_seq_lengths=list(range(1, batch+1)),
                          actual_seq_lengths_kv=[length]*batch,
                          num_key_value_heads=8, num_heads=16, scale=128**-.5, sparse_mode=3)
            reference = torch_npu.npu_fused_infer_attention_score(**params)[0]
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**params)
            outputs = [torch.empty_like(query) for _ in range(args.layers)]
            lses = [torch.empty(1, dtype=query.dtype, device=query.device) for _ in outputs]
            for out, lse in zip(outputs, lses):
                torch_npu.npu_fused_infer_attention_score.out(**params, workspace=workspace, out=[out, lse])
            torch.npu.synchronize()
            update_stream = torch.npu.Stream()
            graph = torch.npu.NPUGraph()
            handles, events = [], []
            with torch.npu.graph(graph):
                stream = torch.npu.current_stream()
                for out, lse in zip(outputs, lses):
                    event = torch.npu.ExternalEvent()
                    event.wait(stream)
                    event.reset(stream)
                    events.append(event)
                    torch.npu.graph_task_group_begin(stream)
                    torch_npu.npu_fused_infer_attention_score.out(
                        **params, workspace=workspace, out=[out, lse])
                    handles.append(torch.npu.graph_task_group_end(stream))

            def step(measure=False):
                begin = time.perf_counter_ns()
                torch.npu.current_stream().synchronize()
                synced = time.perf_counter_ns()
                graph.replay()
                replayed = time.perf_counter_ns()
                totals = {'begin': 0, 'fia': 0, 'end': 0, 'event': 0, 'fused': 0}
                with torch.npu.stream(update_stream):
                    for handle, event, out, lse in zip(handles, events, outputs, lses):
                        t0 = time.perf_counter_ns()
                        if native_update:
                            torch_npu._C._npu_fused_infer_attention_score_out_graph(
                                update_stream, handle, event, **params,
                                workspace=workspace, out=[out, lse])
                            if measure:totals['fused'] += time.perf_counter_ns()-t0
                            continue
                        torch.npu.graph_task_update_begin(update_stream, handle)
                        t1 = time.perf_counter_ns()
                        torch_npu.npu_fused_infer_attention_score.out(
                            **params, workspace=workspace, out=[out, lse])
                        t2 = time.perf_counter_ns()
                        torch.npu.graph_task_update_end(update_stream)
                        t3 = time.perf_counter_ns()
                        event.record(update_stream)
                        t4 = time.perf_counter_ns()
                        if measure:
                            for label, elapsed in zip(totals, (t1-t0, t2-t1, t3-t2, t4-t3)):
                                totals[label] += elapsed
                updated = time.perf_counter_ns()
                torch.npu.synchronize()
                finished = time.perf_counter_ns()
                return {'wall': (finished-begin)/1000, 'pre_sync': (synced-begin)/1000,
                        'replay': (replayed-synced)/1000, 'update': (updated-replayed)/1000,
                        'tail_sync': (finished-updated)/1000,
                        **{k:v/1000 for k,v in totals.items()}}

            for _ in range(10): step()
            torch.testing.assert_close(outputs[-1], reference, atol=.03125, rtol=.02)
            rows = [step(True) for _ in range(args.steps)]
            profile = cProfile.Profile()
            profile.enable()
            for _ in range(20): step()
            profile.disable()
            profile.dump_stats(str(args.output/f'b{batch}.pstats'))
            # A graph without Host updates measures the same FIA device workload.
            static = torch.npu.NPUGraph()
            with torch.npu.graph(static):
                for out, lse in zip(outputs, lses):
                    torch_npu.npu_fused_infer_attention_score.out(
                        **params, workspace=workspace, out=[out, lse])
            for _ in range(10): static.replay()
            torch.npu.synchronize()
            device = []
            for _ in range(5):
                a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                a.record()
                for _ in range(20): static.replay()
                b.record(); b.synchronize()
                device.append(a.elapsed_time(b)*1000/20)
            # 改变长度、页表和 query，验证参数更新确实生效，包括跨页边界。
            dynamic_checks = []
            for actual_length in (127, 129, 383):
                params['actual_seq_lengths_kv'] = [actual_length]*batch
                query.normal_()
                table.copy_(table.flip(1))
                target = torch_npu.npu_fused_infer_attention_score(**params)[0]
                step()
                for actual in outputs:
                    torch.testing.assert_close(actual, target, atol=.03125, rtol=.02)
                # 独立的 CPU FP32 SDPA 参考，避免仅与同一 FIA 实现自比较。
                indices = table[0].long()
                keys = key.view(-1, 128, 8, 128)[indices].flatten(0, 1)[:actual_length]
                values = value.view(-1, 128, 8, 128)[indices].flatten(0, 1)[:actual_length]
                keys = keys.transpose(0, 1).cpu().float().repeat_interleave(2, dim=0)
                values = values.transpose(0, 1).cpu().float().repeat_interleave(2, dim=0)
                oracle = torch.nn.functional.scaled_dot_product_attention(
                    query[0].cpu().float().unsqueeze(1), keys, values).squeeze(1)
                torch.testing.assert_close(outputs[-1][0].cpu().float(), oracle,
                                           atol=.015625, rtol=.02)
                dynamic_checks.append({'length': actual_length, 'layers_checked': len(outputs),
                    'fia_max_abs': (outputs[-1].float()-target.float()).abs().max().item(),
                    'fp32_max_abs': (outputs[-1][0].cpu().float()-oracle).abs().max().item()})
            record = {'batch': batch, 'length': length, 'layers': args.layers,
                      'summary': summary(rows), 'rows': rows,
                      'static_fia_device_us': device,
                      'correctness': 'passed', 'dynamic_checks': dynamic_checks,
                      'layout': {'query': list(query.shape), 'kv': list(key.shape),
                                 'query_format': torch_npu.get_npu_format(query),
                                 'kv_format': torch_npu.get_npu_format(key),
                                 'mask_dtype': str(mask.dtype),
                                 'dtype': str(query.dtype)}}
            results.append(record)
            (args.output/'results.json').write_text(json.dumps(results, indent=2))
            print(json.dumps({k:v for k,v in record.items() if k != 'rows'}), flush=True)
            graph.reset(); static.reset()
            del graph, static, outputs, lses, query, key, value
            torch.npu.synchronize()
    (args.output/'maps.txt').write_text(Path('/proc/self/maps').read_text())


if __name__ == '__main__':
    app()
