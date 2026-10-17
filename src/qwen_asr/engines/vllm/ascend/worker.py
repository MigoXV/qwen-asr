"""Compact, explicit KV cache budgeting for the single NPU Engine Core.

Honor upstream vLLM's kv_cache_memory_bytes option, which vLLM Ascend
0.13.0rc1 otherwise ignores, while still running encoder profiling. This
avoids reserving a device-sized cache for short audio requests.
"""

import gc
import logging

import torch
import torch_npu
from vllm_ascend.worker.worker import NPUWorker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SharedNPUWorker(NPUWorker):
    def load_model(self):
        enabled = self.vllm_config.additional_config.get("qwen_asr_triton_decode", False)
        native_fia = self.vllm_config.additional_config.get("qwen_asr_native_fia_update", False)
        encoder_graph = self.vllm_config.additional_config.get("qwen_asr_encoder_graph", False)
        if encoder_graph and (not enabled
                or self.vllm_config.scheduler_config.max_num_batched_tokens != 1024
                or self.vllm_config.lora_config is not None):
            raise ValueError("Encoder graph requires Triton decode, 1024 batched tokens and no LoRA")
        if enabled and native_fia:
            raise ValueError("Choose either Triton decode or native FIA update")
        if enabled:
            from .triton_decode import validate_config
            validate_config(self.vllm_config)
        if native_fia:
            from .fia_update import validate_config
            validate_config(self.vllm_config)
        super().load_model()
        if enabled:
            from .triton_decode import install
            install(self.model_runner)
        elif native_fia:
            from .fia_update import install
            install(self.model_runner)
        else:
            logger.info("ASR decode attention: upstream FIA/PA (triton_decode=false)")

    def compile_or_warm_up_model(self):
        warmup_duration = super().compile_or_warm_up_model()
        from vllm.compilation.counter import compilation_counter

        mode = self.vllm_config.compilation_config.cudagraph_mode
        captures = compilation_counter.num_cudagraph_captured
        if not self.model_config.enforce_eager and (
            not self.model_runner.use_aclgraph or captures == 0
        ):
            raise RuntimeError(f"ACL Graph requested but not captured: mode={mode}, captures={captures}")
        if self.vllm_config.additional_config.get("qwen_asr_encoder_graph", False):
            # Enable only after profiling and decoder capture have completed.
            # Encoder graphs must never be captured inside another graph.
            self.model_runner.model.audio_tower.enable_encoder_graph()
        logger.info(
            "Execution ready: eager=%s graph_mode=%s captures=%d allocated_mib=%.1f reserved_mib=%.1f peak_allocated_mib=%.1f",
            self.model_config.enforce_eager, mode, captures,
            torch_npu.npu.memory_allocated() / 2**20,
            torch_npu.npu.memory_reserved() / 2**20,
            torch_npu.npu.max_memory_allocated() / 2**20,
        )
        return warmup_duration

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        budget = self.cache_config.kv_cache_memory_bytes
        if budget is None:
            return super().determine_available_memory()
        gc.collect()
        torch_npu.npu.empty_cache()
        torch_npu.npu.reset_peak_memory_stats()
        self.model_runner.profile_run()
        peak = torch_npu.npu.memory_stats()["allocated_bytes.all.peak"]
        free, total = torch_npu.npu.mem_get_info()
        process_budget = int(total * self.cache_config.gpu_memory_utilization)
        if peak + budget > process_budget:
            raise ValueError(
                f"Worker peak {peak} + KV {budget} exceeds process budget {process_budget}; "
                "increase memory-utilization or reduce workers/KV size"
            )
        if budget > free:
            raise ValueError(f"KV budget {budget} exceeds device free memory {free}")
        torch_npu.npu.empty_cache()
        logger.info(
            "Explicit per-worker KV bytes=%d, Torch peak bytes=%d, process budget=%d, device free=%d",
            budget, peak, process_budget, free,
        )
        return budget
