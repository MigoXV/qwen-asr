"""在 src 外串行运行单模型 BF16 图模式基准和诊断。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import typer

app = typer.Typer(add_completion=False)
PROJECT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def engine_config(model, slots, fused):
    from qwen_asr.core.vllm_ascend_backend.config import kv_cache_bytes
    config = dict(model=str(model.resolve()), max_new_tokens=256, dtype='bfloat16',
        max_model_len=1024, max_num_batched_tokens=4096 if slots==4 else 1024, max_num_seqs=slots,
        block_size=128, gpu_memory_utilization=.11,
        kv_cache_memory_bytes=kv_cache_bytes(str(model), slots, 1024),
        enable_prefix_caching=False, mm_processor_cache_gb=0, async_scheduling=True,
        limit_mm_per_prompt={'audio':1}, enforce_eager=False,
        worker_cls=('ascend_diagnostic_worker.DiagnosticNPUWorker' if fused else
                    'qwen_asr.core.vllm_ascend_backend.worker.SharedNPUWorker'),
        compilation_config={'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY',
                            'cudagraph_capture_sizes':list(range(1,slots+1))})
    return config


@app.command()
def serve(model:Path=typer.Option(...), slots:int=typer.Option(...),
          output:Path=typer.Option(...), fused:bool=typer.Option(False)):
    from qwen_asr.inferencers.vllm.pool import VLLMEngineClient
    from qwen_asr.inferencers.grpc_inferencer import GrpcInferencer
    from qwen_asr.commands.app import run_server
    if fused:
        os.environ['ASR_DIAG_LINEAR_ROPE']='1'
    config=engine_config(model,slots,fused)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'config.json',config)
    async def main():
        pool=VLLMEngineClient(max_inflight=32,engine_config=config)
        try:
            await pool.start(timeout=1200)
            await run_server(GrpcInferencer(pool),context='',server_port=50011,max_concurrent_rpcs=32)
        finally:
            await pool.aclose()
    asyncio.run(main())


def stop_group(process):
    if process is None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=45)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()


async def mixed_validation(directory, slots):
    import grpc
    from benchmark_asr import request,variants
    from qwen_asr.protos.asr.ux_speech_pb2_grpc import UxSpeechStub
    expected={}
    for line in (directory/'matrix'/'raw.jsonl').read_text().splitlines():
        row=json.loads(line)
        for req in row['requests']:
            if not req['error']:expected.setdefault(row['input'],set()).add(req['transcript'])
    sr,items=variants(PROJECT/'data-bin/rita.wav',[10,20,30])
    async with grpc.aio.insecure_channel('127.0.0.1:50011') as channel:
        stub=UxSpeechStub(channel)
        work=items*4
        results=await asyncio.gather(*(request(stub,x['pcm'],sr,180) for x in work))
    rows=[dict(input=item['name'],matches_matrix=result['transcript'] in expected[item['name']],**result)
          for item,result in zip(work,results)]
    passed=all(x['matches_matrix'] and not x['error'] for x in rows)
    write_json(directory/'mixed-length.json',{'passed':passed,'requests':rows})
    assert passed,rows


@app.command()
def matrix(model:Path=typer.Option(...),output:Path=typer.Option(...),resume:bool=typer.Option(False),memory_limit_mib:int=typer.Option(16384)):
    """四个独立实例串行测试；结束后恢复原 systemd 服务。"""
    output=output.resolve();output.mkdir(parents=True,exist_ok=resume)
    model=model.resolve()
    env=os.environ.copy()
    env.update(PYTHONPATH=str(PROJECT/'scripts')+os.pathsep+env.get('PYTHONPATH',''),
               VLLM_WORKER_MULTIPROC_METHOD='spawn',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    if resume:
        original=json.loads((output/'provenance.json').read_text())['original_service']
    else:
        original=subprocess.check_output(['systemctl','is-active','qwen-asr-ascend.service'],text=True).strip()
        write_json(output/'provenance.json',{'model':str(model),'source_status':subprocess.check_output(['git','status','--short'],text=True),
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'model_files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in model.glob('*') if p.is_file() and p.suffix in ('.json','.safetensors')},
        'original_service':original,'port':50011,'memory_limit_mib':12288,'dtype':'bfloat16',
        'model_owners':1,'enforce_eager':False,'grpc_capacity':32})
    subprocess.run(['systemctl','stop','qwen-asr-ascend.service'],check=True)
    active={'process':None};done=threading.Event();failures=[]
    def monitor():
        from benchmark_asr import npu_memory_mib
        with (output/'budget.jsonl').open('a') as file:
            while not done.is_set():
                raw=subprocess.check_output(['npu-smi','info'],text=True)
                used=npu_memory_mib(raw)
                pids=re.findall(r'\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|\s*VLLM',raw)
                file.write(json.dumps({'time':time.time(),'hbm_mib':used,'model_pids':pids})+'\n');file.flush()
                if used>memory_limit_mib or len(pids)>1:
                    failures.append(f'Budget violated: {used} MiB, owners={pids}')
                    if active['process']:
                        try:os.killpg(active['process'].pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                    return
                done.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def command(args,log,timeout=3600):
        with log.open('w') as file:
            child=subprocess.Popen([sys.executable,*args],cwd=PROJECT,env=env,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+timeout
            try:
                while child.poll() is None:
                    if failures or (active['process'] is not None and active['process'].poll() is not None):
                        raise RuntimeError(f'Service exited while running {args}; see service.log')
                    if time.monotonic()>deadline:raise TimeoutError(args)
                    time.sleep(.5)
            finally:
                stop_group(child)
        if child.returncode:raise RuntimeError(f'Command failed: {args}; see {log}')
        if failures:raise RuntimeError(failures)
    completed=False
    try:
        for fused in (False,True):
            for slots in (4,8):
                directory=output/('fused' if fused else 'baseline')/f'workers-{slots}'
                if resume and (directory/'service-validation.json').exists() and (directory/'mixed-length.json').exists():
                    print(f'SKIP already complete fused={fused} slots={slots}',flush=True)
                    continue
                directory.mkdir(parents=True)
                args=[sys.executable,str(Path(__file__).resolve()),'serve','--model',str(model),'--slots',str(slots),'--output',str(directory)]
                if fused:args+=['--fused']
                with (directory/'service.log').open('w') as log:
                    process=subprocess.Popen(args,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active['process']=process
                    try:
                        command(['scripts/check_asr.py','--timeout','1200','--requests',str(slots)],directory/'readiness.json',timeout=1400)
                        print(f'READY fused={fused} slots={slots}',flush=True)
                        command(['scripts/benchmark_asr.py','--workers',str(slots),'--output',str(directory/'matrix'),
                                 '--memory-limit-mib',str(memory_limit_mib),
                                 '--requests-per-client','1','--min-requests','4','--execution-mode',
                                 'bf16-full-decode-fused-rope' if fused else 'bf16-full-decode-original-rope'],directory/'benchmark.log')
                        asyncio.run(mixed_validation(directory,slots))
                        command(['scripts/validate_asr_service.py','--output',str(directory/'service-validation.json')],directory/'service-validation.log')
                        print(f'COMPLETE fused={fused} slots={slots}',flush=True)
                    finally:
                        stop_group(process);active['process']=None
                    raw=subprocess.check_output(['npu-smi','info'],text=True)
                    (directory/'device-after.txt').write_text(raw)
                    if re.search(r'\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|\s*VLLM',raw):
                        raise RuntimeError('Model owner remained after shutdown')
        completed=True
    finally:
        stop_group(active['process'])
        done.set();thread.join(timeout=10)
        write_json(output/'state.json',{'completed':completed,'budget_failures':failures,
                    'restored_original_service':original=='active'})
        if original=='active':subprocess.run(['systemctl','start','qwen-asr-ascend.service'],check=True)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    app()
