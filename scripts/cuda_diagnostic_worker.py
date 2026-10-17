"""CUDA 上的隔离诊断 Worker；仅通过测试脚本选择。"""
from __future__ import annotations

import cProfile
import functools
import json
import os
from pathlib import Path
import time

import torch
from vllm.v1.worker.gpu_worker import Worker


class DiagnosticCUDAWorker(Worker):
    def load_model(self):
        variant=os.environ.get('ASR_CUDA_ROPE','original')
        from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
        reference_native=MRotaryEmbedding.forward_native
        if variant=='linear':
            from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding,RotaryEmbedding
            def linear(module,positions,query,key=None,offsets=None):
                module._match_cos_sin_cache_dtype(query)
                return RotaryEmbedding.forward_native(module,positions[0] if positions.ndim==2 else positions,query,key)
            MRotaryEmbedding.forward_cuda=linear
            MRotaryEmbedding.forward_native=linear
        super().load_model()
        self.numerics=[]
        if variant=='linear':
            from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
            model=self.model_runner.model
            base=model.runnable if hasattr(model,'runnable') else model
            rope=base.language_model.model.layers[0].self_attn.rotary_emb
            for count in (4,8,405):
                positions=torch.arange(count,device='cuda').view(1,-1).expand(3,-1).contiguous()
                q=torch.randn(count,16*128,device='cuda',dtype=torch.bfloat16)
                k=torch.randn(count,8*128,device='cuda',dtype=torch.bfloat16)
                reference=reference_native(rope,positions,q.clone(),k.clone())
                actual=rope.forward_cuda(positions,q.clone(),k.clone())
                errors=[(a.float()-b.float()).abs().max().item() for a,b in zip(reference,actual)]
                cosine=[torch.nn.functional.cosine_similarity(a.float().flatten(),b.float().flatten(),dim=0).item() for a,b in zip(reference,actual)]
                self.numerics.append({'tokens':count,'max_abs':errors,'cosine':cosine})
                assert max(errors)<=.0625 and min(cosine)>.9999,self.numerics

    def compile_or_warm_up_model(self):
        super().compile_or_warm_up_model()
        self.diag_enabled=False;self.diag_events=False;self.diag_python=False
        self.diag_rows=[];self.diag_step={};self.diag_profile=cProfile.Profile()
        runner=self.model_runner
        for name in ('_update_states','_prepare_inputs','_execute_mm_encoder','_sample','_bookkeeping_sync'):
            if hasattr(runner,name):self._wrap(runner,name,name)
        self._wrap(runner.model.audio_tower,'forward','audio_encoder')
        self._wrap(runner.model,'compute_logits','compute_logits')
        self._wrap(torch.cuda,'synchronize','device_synchronize')

    def _wrap(self,obj,name,label):
        original=getattr(obj,name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            if not self.diag_enabled:return original(*args,**kwargs)
            started=time.perf_counter_ns();cpu=time.thread_time_ns();events=None
            if self.diag_events and label!='device_synchronize':
                events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[0].record()
            with torch.profiler.record_function('asr::'+label):
                try:return original(*args,**kwargs)
                finally:
                    if events:events[1].record()
                    self.diag_rows.append(dict(phase=label,host_ms=(time.perf_counter_ns()-started)/1e6,
                        thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,events=events,**self.diag_step))
        setattr(obj,name,measured)

    def execute_model(self,scheduler_output):
        if not getattr(self,'diag_enabled',False):return super().execute_model(scheduler_output)
        counts=list(scheduler_output.num_scheduled_tokens.values())
        self.diag_step={'batch':len(counts),'tokens':sum(counts),
                        'stage':'decode' if counts and max(counts)==1 else 'prefill'}
        started=time.perf_counter_ns();cpu=time.thread_time_ns()
        if self.diag_python:self.diag_profile.enable()
        try:
            with torch.profiler.record_function('asr::execute_'+self.diag_step['stage']):
                return super().execute_model(scheduler_output)
        finally:
            if self.diag_python:self.diag_profile.disable()
            self.diag_rows.append(dict(phase='execute_model',host_ms=(time.perf_counter_ns()-started)/1e6,
                thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,events=None,**self.diag_step))

    def diagnostic_control(self,action,path='',events=False,python=False):
        if action=='info':
            config=self.vllm_config
            model=self.model_runner.model
            base=model.runnable if hasattr(model,'runnable') else model
            rope=base.language_model.model.layers[0].self_attn.rotary_emb
            return {'pid':os.getpid(),'numerics':self.numerics,
                'rope_class':type(rope).__name__,'rope_dispatch':rope._forward_method.__qualname__,
                'cpu_affinity':sorted(os.sched_getaffinity(0)),
                'graph_mode':str(config.compilation_config.cudagraph_mode),
                'capture_sizes':config.compilation_config.cudagraph_capture_sizes,
                'allocated_mib':torch.cuda.memory_allocated()/2**20,'reserved_mib':torch.cuda.memory_reserved()/2**20}
        if action=='start':
            self.diag_rows=[];self.diag_events=events;self.diag_python=python
            self.diag_profile=cProfile.Profile();self.diag_enabled=True
            return True
        if action=='stop':
            self.diag_enabled=False;torch.cuda.synchronize()
            for row in self.diag_rows:
                pair=row.pop('events');row['stream_span_ms']=pair[0].elapsed_time(pair[1]) if pair else None
            target=Path(path);target.write_text(json.dumps(self.diag_rows))
            self.diag_profile.dump_stats(str(target.with_suffix('.pstats')))
            return len(self.diag_rows)
        if action=='microbench':return self.microbench()
        raise ValueError(action)

    @torch.inference_mode()
    def microbench(self):
        rows=[]
        selected=[(n,p) for n,p in self.model_runner.model.named_parameters() if p.ndim==2 and
                  any(t in n for t in ('layers.0.self_attn.qkv_proj.weight','layers.0.mlp.gate_up_proj.weight'))]
        for name,weight in selected:
            for batch in (4,8,256):
                x=torch.randn(batch,weight.shape[1],device=weight.device,dtype=weight.dtype)
                for _ in range(5):torch.nn.functional.linear(x,weight)
                torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):torch.nn.functional.linear(x,weight)
                end.record();end.synchronize();ms=start.elapsed_time(end)/50
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(100):graph_output=torch.nn.functional.linear(x,weight)
                start.record()
                for _ in range(10):graph.replay()
                end.record();end.synchronize();graph_ms=start.elapsed_time(end)/1000
                rows.append(dict(name=name,weight_shape=list(weight.shape),batch=batch,ms=ms,
                    graph_ms=graph_ms,graph_tflops=2*batch*weight.numel()/graph_ms/1e9))
        a=torch.empty(32*1024*1024,device='cuda',dtype=torch.bfloat16);b=torch.empty_like(a)
        for _ in range(5):b.copy_(a)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):b.copy_(a)
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):graph.replay()
        end.record();end.synchronize();ms=start.elapsed_time(end)/100
        return {'matmul':rows,'hbm_copy_ms':ms,'hbm_read_write_gbps':2*a.numel()*a.element_size()/ms/1e6}
