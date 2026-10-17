"""串行采集 1.7B NPU 诊断，覆盖加载阶段的显存和权重副本监控。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import typer

from benchmark_asr import npu_memory_mib
from reevaluate_asr import stop_group

app=typer.Typer(add_completion=False)


@app.command()
def main(output:Path=typer.Option(...),model:Path=typer.Option(...),memory_limit_mib:int=typer.Option(16384)):
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    subprocess.run(['systemctl','stop','qwen-asr-ascend.service'],check=True)
    active={'process':None};done=threading.Event();failures=[]
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
               VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONUNBUFFERED='1')
    def monitor():
        with (output/'budget-resources.jsonl').open('a') as file:
            while not done.is_set():
                raw=subprocess.check_output(['npu-smi','info'],text=True)
                used=npu_memory_mib(raw);pids=re.findall(r'\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|\s*VLLM',raw)
                row=dict(time=time.time(),hbm_mib=used,model_pids=pids)
                file.write(json.dumps(row)+'\n');file.flush()
                if used>memory_limit_mib or len(pids)>1:
                    failures.append(row)
                    if active['process']:
                        try:os.killpg(active['process'].pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                    return
                done.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();completed=False
    try:
        cases=[(False,4,''),(False,8,''),(True,4,''),(True,8,''),(True,8,','.join(str(x) for x in range(32)))]
        for fused,slots,cpu_set in cases:
            directory=output/(f'engine-{slots}{"-linear-rope" if fused else ""}'+('-numa0' if cpu_set else ''))
            run_env=env.copy()
            if fused:run_env['ASR_DIAG_LINEAR_ROPE']='1'
            with directory.with_suffix('.log').open('w') as log:
                args=[sys.executable,'scripts/diagnose_asr.py','engine','--model',str(model.resolve()),
                      '--slots',str(slots),'--output',str(directory),'--memory-limit-mib',str(memory_limit_mib)]
                if slots==8 and not cpu_set:args+=['--cpu-perf']
                if cpu_set:args+=['--cpu-set',cpu_set]
                process=subprocess.Popen(args,env=run_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active['process']=process
                try:code=process.wait(timeout=1800)
                finally:stop_group(process);active['process']=None
            if code or failures:raise RuntimeError(f'{directory} failed, code={code}, budget={failures}')
            raw=subprocess.check_output(['npu-smi','info'],text=True)
            if re.search(r'\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|\s*VLLM',raw):raise RuntimeError('Model owner remained')
            print('COMPLETE',directory.name,flush=True)
        completed=True
    finally:
        stop_group(active['process']);done.set();thread.join(timeout=10)
        (output/'state.json').write_text(json.dumps({'completed':completed,'budget_failures':failures},indent=2))
        subprocess.run(['systemctl','start','qwen-asr-ascend.service'],check=True)


if __name__=='__main__':app()
