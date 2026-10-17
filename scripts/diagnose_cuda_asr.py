"""在原始仓库的隔离副本中测量 A100 单模型，复用原 Poetry 环境。"""
from __future__ import annotations

import asyncio
import cProfile
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import numpy as np
import typer

app=typer.Typer(add_completion=False)


@app.command()
def main(output:Path=typer.Option(...),slots:int=typer.Option(...),
         variant:str=typer.Option('original'),model:Path=typer.Option(Path('/workspace/model-bin/Qwen/Qwen3-ASR-1.7B')),
         trace:bool=typer.Option(True)):
    assert variant in ('original','linear')
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    os.environ.update(ASR_CUDA_ROPE=variant,VLLM_TORCH_PROFILER_DIR=str(output/'trace'),
        VLLM_CACHE_ROOT=str(output/'compile-cache-root'),
        VLLM_TORCH_PROFILER_WITH_STACK='0',VLLM_TORCH_PROFILER_RECORD_SHAPES='1',
        VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY='0')
    async def run():
        from qwen_asr.inferencers.vllm.inferencer import VLLMInferencer
        from qwen_asr.inferencers.text.asr_output import parse_asr_output
        from vllm.sampling_params import RequestOutputKind
        import wave,math
        audio_path=Path('/workspace/opusi/qwen-asr/data-bin/2026-10-14/rita.wav')
        with wave.open(str(audio_path)) as wav:
            sr=wav.getframerate();pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2')
        audio=np.tile(pcm,math.ceil(sr*30/len(pcm)))[:sr*30].astype(np.float32)/32768
        config=dict(model=str(model),max_new_tokens=256,dtype='bfloat16',max_model_len=1024,
            max_num_batched_tokens=1024,max_num_seqs=slots,block_size=16,gpu_memory_utilization=.2,
            kv_cache_memory_bytes=slots*112*2**20,enable_prefix_caching=False,mm_processor_cache_gb=0,
            enforce_eager=False,tensor_parallel_size=1,pipeline_parallel_size=1,
            worker_cls='cuda_diagnostic_worker.DiagnosticCUDAWorker')
        (output/'config.json').write_text(json.dumps(config,indent=2))
        adapter=None
        try:
            adapter=VLLMInferencer.LLM(**config);llm=adapter.model
            (output/'worker-info.json').write_text(json.dumps(await llm.collective_rpc('diagnostic_control',args=('info',)),indent=2))
            prompt=adapter._build_text_prompt('',None)
            async def request():
                start=time.perf_counter();first=None;tokens=[];text='';prompt_tokens=None
                params=adapter.sampling_params.clone();params.output_kind=RequestOutputKind.DELTA
                async for result in llm.generate({'prompt':prompt,'multi_modal_data':{'audio':[(audio,sr)]}},params,request_id=uuid4().hex):
                    prompt_tokens=len(result.prompt_token_ids or [])
                    for out in result.outputs:
                        if out.text and first is None:first=time.perf_counter()-start
                        text+=out.text;tokens.extend(out.token_ids)
                return {'latency_s':time.perf_counter()-start,'first_packet_s':first,
                        'prompt_tokens':prompt_tokens,'output_tokens':len(tokens),'token_ids':tokens,'transcript':parse_asr_output(text)[1]}
            async def batch(label,count):
                start=time.perf_counter();rows=await asyncio.gather(*(request() for _ in range(count)))
                elapsed=time.perf_counter()-start
                record={'label':label,'slots':slots,'clients':count,'elapsed_s':elapsed,'throughput_x':count*30/elapsed,'requests':rows}
                with (output/'rounds.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                print(f'{label} slots={slots} variant={variant} throughput={record["throughput_x"]:.3f}x tokens={[x["output_tokens"] for x in rows[:2]]}',flush=True)
            for _ in range(2):await batch('warmup',slots)
            for _ in range(3):await batch('unprofiled',32)
            processor=getattr(llm,'input_processor',None) or llm.processor
            original_process=processor.process_inputs;frontend_rows=[]
            def timed_inputs(*args,**kwargs):
                started=time.perf_counter_ns();cpu=time.thread_time_ns();all_cpu=time.process_time_ns()
                try:return original_process(*args,**kwargs)
                finally:frontend_rows.append({'host_ms':(time.perf_counter_ns()-started)/1e6,
                    'thread_cpu_ms':(time.thread_time_ns()-cpu)/1e6,'process_cpu_ms':(time.process_time_ns()-all_cpu)/1e6})
            processor.process_inputs=timed_inputs
            try:await batch('frontend_timing',32)
            finally:processor.process_inputs=original_process
            (output/'frontend-phases.json').write_text(json.dumps(frontend_rows,indent=2))
            await llm.collective_rpc('diagnostic_control',args=('start',),kwargs={'events':True})
            await batch('phase_events',32)
            await llm.collective_rpc('diagnostic_control',args=('stop',str(output/'phases.json')))
            await llm.collective_rpc('diagnostic_control',args=('start',),kwargs={'python':True})
            frontend_profile=cProfile.Profile();frontend_profile.enable()
            try:await batch('python_profile',slots)
            finally:
                frontend_profile.disable();frontend_profile.dump_stats(str(output/'frontend.pstats'))
            await llm.collective_rpc('diagnostic_control',args=('stop',str(output/'python-phases.json')))
            if trace:
                await llm.collective_rpc('diagnostic_control',args=('start',))
                await llm.start_profile();await batch('torch_trace',slots);await llm.stop_profile()
                await llm.collective_rpc('diagnostic_control',args=('stop',str(output/'trace-phases.json')))
            micro=await llm.collective_rpc('diagnostic_control',args=('microbench',))
            (output/'microbench.json').write_text(json.dumps(micro,indent=2))
            await batch('post_profile_correctness',slots)
        finally:
            if adapter:adapter.model.shutdown()
    asyncio.run(run())


if __name__=='__main__':app()
