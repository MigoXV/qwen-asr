"""单份权重的 ASR 热点诊断；正常服务配置不使用诊断 Worker。"""
from __future__ import annotations

import asyncio
import cProfile
import json
import os
import re
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import grpc
import numpy as np
import psutil
import typer

from benchmark_asr import variants, run_round, npu_memory_mib
from qwen_asr.protos.asr.ux_speech_pb2_grpc import UxSpeechStub

app=typer.Typer(add_completion=False)


async def resources(output, done, memory_limit_mib=12288):
    previous={}
    while not done.is_set():
        proc=await asyncio.create_subprocess_exec('npu-smi','info',stdout=asyncio.subprocess.PIPE)
        raw,_=await proc.communicate()
        info=raw.decode()
        pids=[int(p) for p in re.findall(r'\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|\s*VLLM',info)]
        used=npu_memory_mib(info)
        row={'time':time.time(),'hbm_mib':used,'model_pids':pids,'npu-smi':info,'processes':[]}
        for pid in pids+[os.getpid()]:
            with suppress(psutil.NoSuchProcess):
                p=psutil.Process(pid)
                threads=[]
                for t in p.threads():
                    total=t.user_time+t.system_time
                    old=previous.get((pid,t.id))
                    if old:
                        threads.append({'tid':t.id,'cpu_percent':100*(total-old[0])/(row['time']-old[1])})
                    previous[(pid,t.id)]=(total,row['time'])
                row['processes'].append({'pid':pid,'cpu_num':p.cpu_num(),'cpu_times':p.cpu_times()._asdict(),
                                         'threads_top':sorted(threads,key=lambda t:-t['cpu_percent'])[:12]})
        with (output/'resources.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if used>memory_limit_mib or len(pids)>1:
            if pids: subprocess.run(['kill','-TERM',str(pids[0])])
            raise RuntimeError(f'Memory/model limit violated: {used}, {pids}')
        await asyncio.sleep(0.5)


@app.command()
def grpc_load(output:Path=typer.Option(...), rounds:int=typer.Option(3), clients:int=typer.Option(32), slots:int=typer.Option(8)):
    output.mkdir(parents=True,exist_ok=False)
    async def run():
        sr,items=variants(Path('data-bin/rita.wav'),[30])
        item=items[-1]
        channels=[grpc.aio.insecure_channel('127.0.0.1:50011') for _ in range(clients)]
        stubs=[UxSpeechStub(c) for c in channels]
        done=asyncio.Event();monitor=asyncio.create_task(resources(output,done))
        try:
            for _ in range(2):await run_round(stubs[:slots],item,sr,1,180)
            for i in range(rounds):
                result=await run_round(stubs,item,sr,1,180)
                with (output/'raw.jsonl').open('a') as f:f.write(json.dumps({'round':i+1,**result},ensure_ascii=False)+'\n')
                assert not result['errors'],result
                print(f'grpc round={i+1} speed={result["throughput_x"]:.2f}x',flush=True)
        finally:
            done.set();await monitor
            await asyncio.gather(*(c.close() for c in channels))
    asyncio.run(run())


@app.command()
def engine(output:Path=typer.Option(...),slots:int=typer.Option(8),trace:bool=typer.Option(True),cpu_set:str=typer.Option(''),row_copy:bool=typer.Option(False),model:Path=typer.Option(Path('model-bin/Qwen/Qwen3-ASR-0.6B')),cpu_perf:bool=typer.Option(False),memory_limit_mib:int=typer.Option(12288),duration:float=typer.Option(30.0),clients:list[int]=typer.Option([1,8,32]),rounds:int=typer.Option(3),npugraph_ex:bool=typer.Option(True),microbench:bool=typer.Option(False),short_audio:bool=typer.Option(True),paged_decode:bool=typer.Option(False),measure_only:bool=typer.Option(False),affinity_ablation:bool=typer.Option(False),production_triton:bool=typer.Option(False),validate_lengths:bool=typer.Option(False),evaluation_audio:Path|None=typer.Option(None),native_fia_ablation:bool=typer.Option(False),production_native_fia:bool=typer.Option(False),production_encoder_graph:bool=typer.Option(False)):
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    info=subprocess.check_output(['npu-smi','info'],text=True)
    if re.search(r'\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|\s*VLLM',info):
        raise RuntimeError('Stop the existing service first: only one set of weights may be loaded')
    if cpu_set:
        os.sched_setaffinity(0,{int(x) for x in cpu_set.split(',')})
    os.environ['VLLM_TORCH_PROFILER_DIR']=str(output/'trace')
    os.environ['VLLM_TORCH_PROFILER_WITH_STACK']='0'
    os.environ['VLLM_TORCH_PROFILER_RECORD_SHAPES']='1'
    os.environ['VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY']='0'
    script_path=str(Path(__file__).resolve().parent)
    os.environ['PYTHONPATH']=script_path+os.pathsep+os.environ.get('PYTHONPATH','')
    async def run():
        from qwen_asr.inferencers.vllm.inferencer import VLLMInferencer
        from qwen_asr.engines.vllm.ascend.config import kv_cache_bytes
        from qwen_asr.inferencers.text.asr_output import parse_asr_output
        sr,items=variants(Path('data-bin/rita.wav'),[10,20,30] if validate_lengths else ([duration] if duration else []));item=items[-1]
        audio=np.frombuffer(item['pcm'],dtype='<i2').astype(np.float32)/32768
        model_path=str(model.resolve())
        config=dict(model=model_path,max_new_tokens=256,dtype='bfloat16',max_model_len=1024,
                    max_num_batched_tokens=4096 if slots==4 else 1024,max_num_seqs=slots,block_size=128,
                    gpu_memory_utilization=.11,kv_cache_memory_bytes=kv_cache_bytes(model_path,slots,1024),
                    enable_prefix_caching=False,mm_processor_cache_gb=0,async_scheduling=True,
                    limit_mm_per_prompt={'audio':1},enforce_eager=False,
                    worker_cls='ascend_diagnostic_worker.DiagnosticNPUWorker',
                    compilation_config={'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':list(range(1,slots+1))},
                    additional_config={'ascend_compilation_config':{'enable_npugraph_ex':npugraph_ex}},
                    profiler_config={'profiler':'torch','torch_profiler_dir':str(output/'trace'),'ignore_frontend':True,'torch_profiler_with_stack':False})
        if __import__('importlib.metadata',fromlist=['version']).version('vllm').startswith('0.13.'):
            config.pop('additional_config')
        if production_triton:
            config['additional_config']['qwen_asr_triton_decode'] = True
        if production_encoder_graph:
            config['additional_config']['qwen_asr_encoder_graph'] = True
        if production_native_fia:
            config.setdefault('additional_config',{})['qwen_asr_native_fia_update'] = True
        if paged_decode:
            config['additional_config']['pa_shape_list'] = list(range(1, slots + 1))
        (output/'config.json').write_text(json.dumps(config,indent=2))
        done=asyncio.Event();monitor=asyncio.create_task(resources(output,done,memory_limit_mib))
        adapter=None
        try:
            adapter=VLLMInferencer.LLM(**config)
            llm=adapter.engine.model
            worker_info=await llm.collective_rpc('diagnostic_control',args=('info',))
            (output/'worker-info.json').write_text(json.dumps(worker_info,indent=2))
            prompt=adapter._build_text_prompt('',None)
            async def request():
                start=time.perf_counter();first=None;tokens=[];text='';prompt_tokens=None;finish_reason=None;stop_reason=None
                params=adapter.engine.sampling_params.clone()
                from vllm.sampling_params import RequestOutputKind
                params.output_kind=RequestOutputKind.DELTA
                async for result in llm.generate({'prompt':prompt,'multi_modal_data':{'audio':[(audio,sr)]}},params,request_id=uuid4().hex):
                    prompt_tokens=len(result.prompt_token_ids or [])
                    for out in result.outputs:
                        if out.text and first is None:first=time.perf_counter()-start
                        text+=out.text;tokens.extend(out.token_ids)
                        finish_reason=out.finish_reason or finish_reason
                        stop_reason=out.stop_reason if out.stop_reason is not None else stop_reason
                return {'latency_s':time.perf_counter()-start,'first_packet_s':first,
                        'prompt_tokens':prompt_tokens,'output_tokens':len(tokens),'token_ids':tokens,
                        'finish_reason':finish_reason,'stop_reason':stop_reason,'transcript':parse_asr_output(text)[1]}
            async def batch(label,n):
                start=time.perf_counter();results=await asyncio.gather(*(request() for _ in range(n)))
                elapsed=time.perf_counter()-start
                record={'label':label,'input':item['name'],'clients':n,'duration_s':item['duration_s'],'elapsed_s':elapsed,'throughput_x':n*item['duration_s']/elapsed,'requests':results}
                with (output/'rounds.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                print(f'{label} slots={slots} clients={n} speed={record["throughput_x"]:.2f}x tokens={[x["output_tokens"] for x in results[:2]]}',flush=True)
                return record
            for _ in range(2):await batch('warmup',slots)
            for n in clients:
                for _ in range(rounds):await batch('unprofiled',n)
            if native_fia_ablation:
                await llm.collective_rpc('diagnostic_control',args=('native_fia_update',str(output/'native-fia-update-source.py')),kwargs={'events':True})
                try:
                    for _ in range(2):await batch('native_fia_warmup',slots)
                    for n in clients:
                        for _ in range(rounds):await batch('native_fia',n)
                    native_info=await llm.collective_rpc('diagnostic_control',args=('info',))
                    (output/'native-worker-info.json').write_text(json.dumps(native_info,indent=2))
                    for n in clients:
                        await llm.collective_rpc('diagnostic_control',args=('start',),kwargs={'python':True})
                        await batch('native_fia_python_profile',n)
                        await llm.collective_rpc('diagnostic_control',args=('stop',str(output/f'native-python-phases-c{n}.json')))
                finally:
                    await llm.collective_rpc('diagnostic_control',args=('native_fia_update',),kwargs={'events':False})
                for n in clients:
                    for _ in range(rounds):await batch('native_fia_restored',n)
            if affinity_ablation:
                await llm.collective_rpc('diagnostic_control',args=('local_affinity',),kwargs={'events':True})
                try:
                    for _ in range(2):await batch('local_affinity_warmup',slots)
                    for n in clients:
                        for _ in range(rounds):await batch('local_affinity',n)
                    local_info=await llm.collective_rpc('diagnostic_control',args=('info',))
                    (output/'local-worker-info.json').write_text(json.dumps(local_info,indent=2))
                finally:
                    await llm.collective_rpc('diagnostic_control',args=('local_affinity',),kwargs={'events':False})
                for n in clients:
                    for _ in range(rounds):await batch('affinity_restored',n)
            if measure_only:
                if microbench:
                    micro=await llm.collective_rpc('diagnostic_control',args=('microbench',))
                    (output/'microbench.json').write_text(json.dumps(micro,indent=2))
                if short_audio and item is not items[0]:
                    item=items[0]
                    audio=np.frombuffer(item['pcm'],dtype='<i2').astype(np.float32)/32768
                    for _ in range(2):await batch('short_warmup',slots)
                    for n in clients:
                        for _ in range(rounds):await batch('short_unprofiled',n)
                final_info=await llm.collective_rpc('diagnostic_control',args=('info',))
                (output/'final-worker-info.json').write_text(json.dumps(final_info,indent=2))
                return
            if cpu_perf:
                pid=str(worker_info[0]['pid'])
                with (output/'perf-record.log').open('w') as log, (output/'perf-stat.log').open('w') as statlog:
                    perf=await asyncio.create_subprocess_exec('perf','record','-q','-F','199','-g','--call-graph','dwarf,4096',
                        '-t',pid,'-o',str(output/'cpu.data'),'--','sleep','20',stdout=log,stderr=log)
                    stat=await asyncio.create_subprocess_exec('perf','stat','-t',pid,'-e',
                        'task-clock,cycles,instructions,context-switches,cpu-migrations,page-faults','--','sleep','20',stdout=statlog,stderr=statlog)
                    while perf.returncode is None:await batch('cpu_perf',32)
                    await perf.wait();await stat.wait()
                with (output/'perf-report.txt').open('w') as stream:
                    subprocess.run(['perf','report','-i',str(output/'cpu.data'),'--stdio','--no-children','--percent-limit','1'],stdout=stream,stderr=subprocess.STDOUT,check=True)
            processor=llm.input_processor
            original_process=processor.process_inputs
            frontend_rows=[]
            def timed_inputs(*args,**kwargs):
                started=time.perf_counter_ns();cpu=time.thread_time_ns();all_cpu=time.process_time_ns()
                try:return original_process(*args,**kwargs)
                finally:frontend_rows.append({'host_ms':(time.perf_counter_ns()-started)/1e6,
                    'thread_cpu_ms':(time.thread_time_ns()-cpu)/1e6,'process_cpu_ms':(time.process_time_ns()-all_cpu)/1e6})
            processor.process_inputs=timed_inputs
            try:await batch('frontend_timing',32)
            finally:processor.process_inputs=original_process
            (output/'frontend-phases.json').write_text(json.dumps(frontend_rows,indent=2))
            # 同一次 CPU 位置实验复测布局 copy，避免再加载一遍模型。
            auto_row_copy=bool(cpu_set) and os.environ.get('ASR_DIAG_LINEAR_ROPE')=='1'
            if row_copy or auto_row_copy:
                await llm.collective_rpc('diagnostic_control',args=('row_copy',),kwargs={'events':True})
                for _ in range(3):await batch('row_copy_ablation',32)
                await llm.collective_rpc('diagnostic_control',args=('row_copy',),kwargs={'events':False})
            for n in clients:
                await llm.collective_rpc('diagnostic_control',args=('start',),kwargs={'events':True})
                await batch('phase_events',n)
                await llm.collective_rpc('diagnostic_control',args=('stop',str(output/f'phases-c{n}.json')))
                await llm.collective_rpc('diagnostic_control',args=('start',),kwargs={'python':True})
                frontend_profile=cProfile.Profile();frontend_profile.enable()
                try:await batch('python_profile',n)
                finally:
                    frontend_profile.disable();frontend_profile.dump_stats(str(output/f'frontend-c{n}.pstats'))
                await llm.collective_rpc('diagnostic_control',args=('stop',str(output/f'python-phases-c{n}.json')))
            if trace:
                for n in clients:
                    await llm.collective_rpc('diagnostic_control',args=('start',))
                    await llm.start_profile()
                    await batch('torch_trace',n)
                    await llm.stop_profile()
                    await llm.collective_rpc('diagnostic_control',args=('stop',str(output/f'trace-phases-c{n}.json')))
            if microbench:
                micro=await llm.collective_rpc('diagnostic_control',args=('microbench',))
                (output/'microbench.json').write_text(json.dumps(micro,indent=2))
            await batch('post_profile_correctness',slots)
            if short_audio and item is not items[0]:
                item = items[0]
                audio = np.frombuffer(item['pcm'],dtype='<i2').astype(np.float32)/32768
                for _ in range(2):await batch('short_warmup',slots)
                for n in clients:
                    for _ in range(rounds):await batch('short_unprofiled',n)
                    await llm.collective_rpc('diagnostic_control',args=('start',),kwargs={'python':True})
                    await batch('short_python_profile',n)
                    await llm.collective_rpc('diagnostic_control',args=('stop',str(output/f'short-python-c{n}.json')))
                    if trace:
                        await llm.collective_rpc('diagnostic_control',args=('start',))
                        await llm.start_profile()
                        await batch('short_torch_trace',n)
                        await llm.stop_profile()
                        await llm.collective_rpc('diagnostic_control',args=('stop',str(output/f'short-trace-c{n}.json')))
            if validate_lengths:
                for item in items[1:-1]:
                    audio=np.frombuffer(item['pcm'],dtype='<i2').astype(np.float32)/32768
                    for n in clients:await batch('validate_length_tokens',n)
            if evaluation_audio:
                import soundfile as sf
                for name in ('asr_zh.wav','asr_en.wav'):
                    waveform,sr=sf.read(evaluation_audio/name,dtype='float32')
                    pcm=(np.clip(waveform,-1,32767/32768)*32768).astype('<i2').tobytes()
                    audio=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
                    item={'name':name,'duration_s':len(audio)/sr}
                    for n in clients:await batch('validate_reference_tokens',n)
            final_info=await llm.collective_rpc('diagnostic_control',args=('info',))
            (output/'final-worker-info.json').write_text(json.dumps(final_info,indent=2))
        finally:
            if adapter:await adapter.aclose()
            done.set();await monitor
    asyncio.run(run())


if __name__=='__main__':app()
