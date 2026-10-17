"""Script-only diagnostic worker; never selected by the production service.

Host durations include dispatch/wait. NPU event spans include stream idle gaps;
only the profiler's kernel timeline is used to calculate device busy time.
"""

from __future__ import annotations

import cProfile
import functools
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch_npu

from qwen_asr.engines.vllm.ascend.worker import SharedNPUWorker


class DiagnosticNPUWorker(SharedNPUWorker):
    def load_model(self):
        if os.environ.get('ASR_DIAG_LINEAR_ROPE')=='1':
            from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding
            def linear_rope(module, positions, query, key):
                # ASR get_mrope_input_positions constructs three identical axes.
                # This experiment is restricted to this ASR model, never VL input.
                if positions.ndim==2:
                    positions=positions[0]
                cache=module.cos_sin_cache
                if cache.dtype!=query.dtype or cache.device!=query.device:
                    cache=cache.to(device=query.device,dtype=query.dtype)
                    module.cos_sin_cache=cache
                return torch_npu.npu_mrope(positions.contiguous(),query.contiguous(),key.contiguous(),
                    cache.contiguous(),module.head_size,mrope_section=[0,0,0],rotary_mode='half')
            AscendMRotaryEmbedding.forward_oot=linear_rope
        super().load_model()
        if os.environ.get('ASR_DIAG_TRITON_PA') == '1':
            from ascend_triton_attention import install
            info = install(self.model_runner,
                           block=int(os.environ.get('ASR_DIAG_TRITON_BLOCK', '128')),
                           grouped=os.environ.get('ASR_DIAG_TRITON_GQA') == '1')
            print('ASR_TRITON_DECODE '+json.dumps(info), flush=True)
        if os.environ.get('ASR_DIAG_LINEAR_ROPE')=='1':
            from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
            model=self.model_runner.model
            base=model.runnable if hasattr(model,'runnable') else model
            rope=base.language_model.model.layers[0].self_attn.rotary_emb
            checks=[]
            for count in (4,8,405):
                positions=torch.arange(count,device='npu').view(1,-1).expand(3,-1).contiguous()
                q=torch.randn(count,16*128,device='npu',dtype=torch.bfloat16)
                k=torch.randn(count,8*128,device='npu',dtype=torch.bfloat16)
                reference=MRotaryEmbedding.forward_native(rope,positions,q.clone(),k.clone())
                actual=rope.forward_oot(positions,q.clone(),k.clone())
                errors=[(a.float()-b.float()).abs().max().item() for a,b in zip(reference,actual)]
                cosine=[torch.nn.functional.cosine_similarity(a.float().flatten(),b.float().flatten(),dim=0).item() for a,b in zip(reference,actual)]
                checks.append({'tokens':count,'max_abs':errors,'cosine':cosine})
                if max(errors)>.0625 or min(cosine)<.9999:
                    raise RuntimeError(f'Linear rope numerical mismatch: {checks}')
            print('LINEAR_ROPE_NUMERICAL_CHECK '+json.dumps(checks),flush=True)
            self.diag_numerical_checks=checks

    def compile_or_warm_up_model(self):
        super().compile_or_warm_up_model()
        self.diag_enabled = False
        self.diag_events = False
        self.diag_python = False
        self.diag_row_copy = False
        self.diag_rows = []
        self.diag_step = {}
        self.diag_profile = cProfile.Profile()
        runner = self.model_runner
        for name in ("_update_states", "_prepare_inputs", "_execute_mm_encoder",
                     "_preprocess", "_sample", "_bookkeeping_sync",
                     "_update_full_graph_params_if_needed", "_model_forward", "_generate_process_reqs_hidden_states"):
            if hasattr(runner, name):
                self._wrap(runner, name, name)
        self._wrap(runner.model.audio_tower, "forward", "audio_encoder")
        self._wrap(runner.model, "compute_logits", "compute_logits")
        import vllm_ascend.worker.model_runner_v1 as runner_module
        if hasattr(runner_module, "update_attn_params"):
            self._wrap(runner_module, "update_attn_params", "graph_attention_update")
        if hasattr(runner_module, "update_full_graph_params"):
            self._wrap(runner_module, "update_full_graph_params", "graph_attention_update")
        # Isolate the synchronization immediately before graph replay.
        self._wrap(torch.npu, "synchronize", "npu_synchronize")
        original_copy = torch.Tensor.copy_
        worker = self
        def measured_copy(destination, source, *args, **kwargs):
            if worker.diag_row_copy and source.device.type=='cpu' and destination.device.type=='npu' and destination.ndim==2 and destination.shape[0]==3 and not destination.is_contiguous():
                for axis in range(3):
                    original_copy(destination[axis],source[axis],*args,**kwargs)
                return destination
            if not worker.diag_enabled:
                return original_copy(destination, source, *args, **kwargs)
            started=time.perf_counter_ns()
            cpu_started=time.thread_time_ns()
            try:
                return original_copy(destination, source, *args, **kwargs)
            finally:
                worker.diag_rows.append({"phase":"tensor_copy", "host_ms":(time.perf_counter_ns()-started)/1e6,
                    "thread_cpu_ms":(time.thread_time_ns()-cpu_started)/1e6,
                    "start_ns":started,"events":None,"bytes":destination.numel()*destination.element_size(),
                    "source_device":str(source.device),"destination_device":str(destination.device),
                    "source_pinned":source.is_pinned() if source.device.type=='cpu' else None,
                    "shape":list(destination.shape),**worker.diag_step})
        torch.Tensor.copy_=measured_copy

    def _wrap(self, obj, name, label):
        original = getattr(obj, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            if not self.diag_enabled:
                return original(*args, **kwargs)
            start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            events = None
            if self.diag_events and label != "npu_synchronize":
                events = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                events[0].record()
            with torch.profiler.record_function("asr::" + label):
                try:
                    return original(*args, **kwargs)
                finally:
                    if events:
                        events[1].record()
                    self.diag_rows.append({"phase": label, "host_ms": (time.perf_counter_ns()-start)/1e6,
                                           "thread_cpu_ms": (time.thread_time_ns()-cpu_start)/1e6,
                                           "start_ns": start, "events": events, **self.diag_step})
        setattr(obj, name, measured)

    def execute_model(self, scheduler_output):
        if not getattr(self, "diag_enabled", False):
            return super().execute_model(scheduler_output)
        counts = list(scheduler_output.num_scheduled_tokens.values())
        self.diag_step = {"batch": len(counts), "tokens": sum(counts),
                          "stage": "decode" if counts and max(counts)==1 else "prefill"}
        started = time.perf_counter_ns()
        cpu_started = time.thread_time_ns()
        if self.diag_python:
            self.diag_profile.enable()
        try:
            with torch.profiler.record_function("asr::execute_" + self.diag_step["stage"]):
                return super().execute_model(scheduler_output)
        finally:
            if self.diag_python:
                self.diag_profile.disable()
            self.diag_rows.append({"phase": "execute_model", "host_ms": (time.perf_counter_ns()-started)/1e6,
                                   "thread_cpu_ms": (time.thread_time_ns()-cpu_started)/1e6,
                                   "start_ns": started, "events": None, **self.diag_step})

    def sample_tokens(self, grammar_output):
        if not getattr(self, "diag_enabled", False):
            return super().sample_tokens(grammar_output)
        started = time.perf_counter_ns()
        cpu_started = time.thread_time_ns()
        if self.diag_python:
            self.diag_profile.enable()
        try:
            with torch.profiler.record_function("asr::sample_tokens"):
                return super().sample_tokens(grammar_output)
        finally:
            if self.diag_python:
                self.diag_profile.disable()
            self.diag_rows.append({"phase": "sample_tokens", "host_ms": (time.perf_counter_ns()-started)/1e6,
                                   "thread_cpu_ms": (time.thread_time_ns()-cpu_started)/1e6,
                                   "start_ns": started, "events": None, **self.diag_step})

    def diagnostic_control(self, action, path="", events=False, python=False):
        if action == 'native_fia_update':
            import inspect
            import textwrap
            from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
            implementation = AscendAttentionBackend.get_impl_cls()
            if events:
                if not hasattr(torch_npu._C,'_npu_fused_infer_attention_score_out_graph'):
                    raise RuntimeError('Native fused graph update is unavailable')
                original = implementation.update_graph_params
                source = textwrap.dedent(inspect.getsource(original))
                source = source.removeprefix('@staticmethod\n')
                offset = source.rindex('torch.npu.graph_task_update_begin(update_stream, handle)')
                prefix, tail = source[:offset], source[offset:]
                tail = tail.replace('torch.npu.graph_task_update_begin(update_stream, handle)', '', 1)
                tail = tail.replace('torch_npu.npu_fused_infer_attention_score.out(',
                    'torch_npu._C._npu_fused_infer_attention_score_out_graph(\n'
                    '                    update_stream, handle, event,', 1)
                tail = tail.replace('torch.npu.graph_task_update_end(update_stream)', '', 1)
                tail = tail.replace('event.record(update_stream)', '', 1)
                source = prefix+tail
                scope = {}
                exec(compile(source,'<native-fia-update>','exec'),original.__globals__,scope)
                self.diag_saved_native_fia = original
                implementation.update_graph_params = staticmethod(scope['update_graph_params'])
                if path:Path(path).write_text(source)
            else:
                implementation.update_graph_params = staticmethod(self.diag_saved_native_fia)
            self.diag_native_fia_active = events
            return {'native_fia_update': events}
        if action == "local_affinity":
            if events:
                self.diag_saved_affinity = {}
                for task in Path('/proc/self/task').iterdir():
                    tid = int(task.name)
                    try:
                        current = os.sched_getaffinity(tid)
                        self.diag_saved_affinity[tid] = current
                        # This diagnostic is specific to the measured physical
                        # NPU4 topology (NUMA0 CPUs 0..31). Keep dedicated ACL
                        # and release thread roles on distinct local CPUs.
                        local = {next(iter(current)) % 32} if len(current) == 1 else set(range(30))
                        os.sched_setaffinity(tid, local)
                    except ProcessLookupError:
                        pass
            else:
                for tid, affinity in getattr(self, 'diag_saved_affinity', {}).items():
                    try:
                        os.sched_setaffinity(tid, affinity)
                    except ProcessLookupError:
                        pass
            return {'local_affinity': events}
        if action=='row_copy':
            self.diag_row_copy=events
            return {'row_copy':events}
        if action == "info":
            from vllm_ascend.compilation.acl_graph import get_graph_params
            graph_params = get_graph_params()
            examples = {}
            for size, parameters in graph_params.attn_params.items():
                if not parameters:continue
                fields=[]
                for value in parameters[0]:
                    if isinstance(value,torch.Tensor):
                        fields.append({'type':type(value).__name__,'shape':list(value.shape),
                            'stride':list(value.stride()),'dtype':str(value.dtype),
                            'format':torch_npu.get_npu_format(value),
                            'contiguous':value.is_contiguous()})
                    elif isinstance(value,(int,float,str,list,tuple)):
                        fields.append(value)
                    else:fields.append({'type':str(type(value))})
                workspace=graph_params.workspaces.get(size)
                examples[str(size)]={'fields':fields,'workspace_bytes':
                    workspace.numel()*workspace.element_size() if isinstance(workspace,torch.Tensor) else None}
            model = self.model_runner.model
            weights = [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype),
                        "format": torch_npu.get_npu_format(p)}
                       for n,p in model.named_parameters() if p.ndim == 2][:12]
            affinities=Counter()
            for task in Path('/proc/self/task').iterdir():
                try:affinities[str(sorted(os.sched_getaffinity(int(task.name))))]+=1
                except ProcessLookupError:pass
            return {"pid": os.getpid(), "weights": weights,"graph_parameter_examples":examples,
                    "native_fia_update":getattr(self,"diag_native_fia_active",False),
                    "production_native_fia_update":getattr(self.model_runner,"asr_native_fia_update",None),
                    "worker_class": type(self).__module__ + "." + type(self).__qualname__,
                    "runner_class": type(self.model_runner).__module__ + "." + type(self.model_runner).__qualname__,
                    "kv_cache_memory_bytes": self.cache_config.kv_cache_memory_bytes,
                    "use_aclgraph": self.model_runner.use_aclgraph,
                    "graph_mode": str(self.vllm_config.compilation_config.cudagraph_mode),
                    "triton_decode": getattr(self.model_runner, 'asr_triton_decode', None),
                    "encoder_graph": getattr(getattr(self.model_runner.model.audio_tower, "encoder_graph", None), "stats", None),
                    "libraries": [line.strip() for line in Path('/proc/self/maps').read_text().splitlines()
                                  if any(name in line for name in ('libascendcl', 'libopapi', 'libtorch_npu', 'libhccl'))],
                    "numerics":getattr(self,'diag_numerical_checks',[]),
                    "cpu_affinity":sorted(os.sched_getaffinity(0)),
                    "thread_affinity_counts":dict(affinities),
                    "allocated_mib": torch.npu.memory_allocated()/2**20,
                    "reserved_mib": torch.npu.memory_reserved()/2**20}
        if action == "start":
            self.diag_rows = []
            self.diag_events = events
            self.diag_python = python
            self.diag_profile = cProfile.Profile()
            self.diag_enabled = True
            return {"started": True}
        if action == "stop":
            self.diag_enabled = False
            torch.npu.synchronize()
            for row in self.diag_rows:
                pair = row.pop("events")
                row["stream_span_ms"] = pair[0].elapsed_time(pair[1]) if pair else None
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(self.diag_rows))
            self.diag_profile.dump_stats(str(target.with_suffix(".pstats")))
            count = len(self.diag_rows)
            self.diag_rows = []
            return {"records": count, "path": str(target)}
        if action == "microbench":
            return self._microbench()
        raise ValueError(action)

    @torch.inference_mode()
    def _microbench(self):
        """Same loaded weights, no second model. Device-only calibration."""
        model = self.model_runner.model
        selected = [(n,p) for n,p in model.named_parameters()
                    if p.ndim==2 and any(x in n for x in ("layers.0.self_attn.qkv_proj.weight",
                                                          "layers.0.mlp.gate_up_proj.weight"))]
        results = []
        for name, weight in selected:
            for batch in (4,8,256):
                x = torch.randn((batch,weight.shape[1]),device=weight.device,dtype=weight.dtype)
                for _ in range(5):
                    torch.nn.functional.linear(x,weight)
                torch.npu.synchronize()
                start,end = torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    torch.nn.functional.linear(x,weight)
                end.record();end.synchronize()
                ms = start.elapsed_time(end)/50
                graph = torch.npu.NPUGraph()
                graph_output=torch.empty((batch,weight.shape[0]),device=weight.device,dtype=weight.dtype)
                transposed_weight=weight.t()
                with torch.npu.graph(graph):
                    for _ in range(100):
                        torch.mm(x,transposed_weight,out=graph_output)
                start.record()
                for _ in range(10):graph.replay()
                end.record();end.synchronize()
                graph_ms=start.elapsed_time(end)/1000
                results.append({"name":name,"weight_shape":list(weight.shape),"batch":batch,
                                "stream_ms":ms,"effective_tflops":2*batch*weight.numel()/ms/1e9,
                                "graph_ms":graph_ms,"graph_tflops":2*batch*weight.numel()/graph_ms/1e9,
                                "weight_read_gbs":weight.numel()*weight.element_size()/ms/1e6})
        layout_results = []
        for name, weight in selected:
            for batch in (1, 2, 4, 8, 256):
                x = torch.randn((batch, weight.shape[1]), device=weight.device, dtype=weight.dtype)
                reference = torch.nn.functional.linear(x, weight)
                for label, candidate in (("ND", weight), ("NZ", torch_npu.npu_format_cast(weight, 29))):
                    actual = torch.nn.functional.linear(x, candidate)
                    error = (actual.float() - reference.float()).abs()
                    max_abs, mean_abs = error.max().item(), error.mean().item()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        for _ in range(32):
                            graph_output = torch.nn.functional.linear(x, candidate)
                    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    for _ in range(3):graph.replay()
                    torch.npu.synchronize()
                    start.record()
                    for _ in range(20):graph.replay()
                    end.record();end.synchronize()
                    layout_results.append({"name": name, "shape": list(weight.shape), "batch": batch,
                        "layout": label, "format": torch_npu.get_npu_format(candidate),
                        "graph_ms": start.elapsed_time(end)/640, "max_abs": max_abs,
                        "mean_abs": mean_abs, "aligned": bool(torch.allclose(actual, reference, atol=.0625, rtol=.05)),
                        "graph_aligned": bool(torch.allclose(graph_output, reference, atol=.0625, rtol=.05))})
        # Two 64 MiB buffers, copying 128 MiB of HBM traffic per iteration.
        source = torch.empty(32*1024*1024,device="npu",dtype=torch.bfloat16)
        destination = torch.empty_like(source)
        for _ in range(5): destination.copy_(source)
        torch.npu.synchronize()
        start,end = torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(100): destination.copy_(source)
        end.record();end.synchronize()
        ms=start.elapsed_time(end)/100
        return {"linear":results,"layouts":layout_results,"copy_64mib_ms":ms,"copy_read_write_gbs":2*source.numel()*2/ms/1e6,
                "note":"Event spans of batched enqueues; this is stack calibration, not a silicon peak specification."}
