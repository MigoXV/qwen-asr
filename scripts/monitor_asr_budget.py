"""跨加载、压测、诊断和服务恢复阶段记录单模型显存硬预算。"""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import typer

from benchmark_asr import npu_memory_mib

app=typer.Typer(add_completion=False)


@app.command()
def main(output:Path=typer.Option(...),stop_file:Path=typer.Option(...),memory_limit_mib:int=typer.Option(12288)):
    with output.open('a') as file:
        while not stop_file.exists():
            raw=subprocess.check_output(['npu-smi','info'],text=True)
            pids=re.findall(r'\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|\s*VLLM',raw)
            row=dict(time=time.time(),hbm_mib=npu_memory_mib(raw),model_pids=pids)
            file.write(json.dumps(row)+'\n');file.flush()
            if row['hbm_mib']>memory_limit_mib or len(pids)>1:
                for pid in pids:
                    try:os.kill(int(pid),signal.SIGTERM)
                    except ProcessLookupError:pass
                raise RuntimeError(f'Budget exceeded: {row}')
            time.sleep(.5)


if __name__=='__main__':app()
