"""远端串行实验驱动；原仓库只读，结束后恢复原服务。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path('/workspace/opusi/qwen-asr-diagnostics-20261014-1.7b')
MAIN=Path('/workspace/opusi/qwen-asr')
SOURCE=ROOT/'source'


def owners():
    text=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return text.strip().splitlines() if text.strip() else []


def kill_group(p):
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    p.wait()


def main():
    saved=json.loads((ROOT/'original-processes.json').read_text())
    supervisor=next(x for x in saved if any('supervise_a100.py' in a for a in x['args']))
    restored_pid=ROOT/'restored-service/supervisor.pid'
    if restored_pid.exists():supervisor['pid']=int(restored_pid.read_text())
    os.kill(supervisor['pid'],signal.SIGTERM)
    for _ in range(70):
        if not owners():break
        time.sleep(1)
    assert not owners(),owners()
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(SOURCE/'scripts')+':'+str(SOURCE/'src'))
    active={'process':None};done=threading.Event();failures=[]
    def monitor():
        with (ROOT/'results'/'budget.jsonl').open('a') as file:
            while not done.is_set():
                raw=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
                memory,util,temp,power=[float(x.strip()) for x in raw.split(',')]
                procs=owners();row=dict(time=time.time(),memory_mib=memory,util_percent=util,temperature=temp,power=power,owners=procs)
                file.write(json.dumps(row)+'\n');file.flush()
                if memory>12288 or len(procs)>1:
                    failures.append(row)
                    if active['process']:
                        try:os.killpg(active['process'].pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                    return
                done.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    completed=False
    try:
        for variant in os.environ.get('ASR_CUDA_VARIANTS','original,linear').split(','):
            for slots in [int(x) for x in os.environ.get('ASR_CUDA_SLOTS','4,8').split(',')]:
                name=f'engine-{slots}-{variant}'+os.environ.get('ASR_CUDA_OUTPUT_SUFFIX','')
                out=ROOT/'results'/name
                with (ROOT/'results'/f'{name}.log').open('w') as log:
                    proc=subprocess.Popen([sys.executable,str(SOURCE/'scripts/diagnose_cuda_asr.py'),'--output',str(out),
                        '--slots',str(slots),'--variant',variant],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active['process']=proc
                    try:code=proc.wait(timeout=1500)
                    finally:kill_group(proc);active['process']=None
                if code or failures:raise RuntimeError(f'Failed {out}: {code}, budget failures {failures}')
                assert not owners(),owners()
                print(f'COMPLETE {variant} slots={slots}',flush=True)
        completed=True
    finally:
        if active['process']:kill_group(active['process'])
        done.set();thread.join(timeout=10)
        restore_env=os.environ.copy();restore_env.update(supervisor['environment'])
        restore_env.update(PYTHONDONTWRITEBYTECODE='1',QWEN_ASR_BENCHMARK_DIR=str(ROOT/'restored-service'))
        with (ROOT/'restore.log').open('a') as log:
            restored=subprocess.Popen(supervisor['args'],cwd=supervisor['cwd'],env=restore_env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        hashes=json.loads((ROOT/'source-hashes.json').read_text())
        changed=[name for name,h in hashes.items() if not (MAIN/name).exists() or hashlib.sha256((MAIN/name).read_bytes()).hexdigest()!=h]
        status=subprocess.check_output(['git','status','--short'],cwd=MAIN,text=True)
        state={'completed':completed,'budget_failures':failures,'restored_supervisor_pid':restored.pid,
            'main_changed_files':changed,'main_status_unchanged':status==(ROOT/'original-status.txt').read_text()}
        (ROOT/'results'/'state.json').write_text(json.dumps(state,indent=2))
        assert not changed and state['main_status_unchanged'],state


if __name__=='__main__':main()
