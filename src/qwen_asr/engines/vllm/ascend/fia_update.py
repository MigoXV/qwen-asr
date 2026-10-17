"""ASR Worker 内的原生 FIA 图更新，不修改上游类或第三方文件。"""

from importlib.metadata import version
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def validate_config(config):
    import torch
    import torch_npu
    from vllm.config.compilation import CUDAGraphMode

    text = config.model_config.hf_text_config
    if (getattr(config.model_config.hf_config, "model_type", None) != "qwen3_asr"
            or (text.num_hidden_layers, text.num_attention_heads,
                text.num_key_value_heads, text.head_dim) != (28, 16, 8, 128)):
        raise ValueError("Native FIA update requires Qwen3-ASR-1.7B (28 layers, Q16/KV8/D128)")
    if (config.model_config.dtype != torch.bfloat16
            or config.model_config.enforce_eager
            or config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
            or config.model_config.max_model_len != 1024
            or config.scheduler_config.max_num_seqs > 8
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1
            or config.parallel_config.data_parallel_size != 1
            or getattr(config.parallel_config, "decode_context_parallel_size", 1) != 1
            or getattr(config.parallel_config, "prefill_context_parallel_size", 1) != 1):
        raise ValueError("Native FIA update requires BF16, FULL_DECODE_ONLY, <=8 slots, "
                         "context 1024, TP/PP/DP/CP1, no speculative decoding or KV transfer")
    expected = {"torch": "2.9.1", "torch-npu": "2.9.1",
                "vllm": "0.18.0", "vllm-ascend": "0.18.0"}
    actual = {name: version(name).split("+")[0] for name in expected}
    if actual != expected:
        raise ValueError(f"Native FIA update versions not validated: {actual}; expected {expected}")
    if not callable(getattr(torch_npu._C, "_npu_fused_infer_attention_score_out_graph", None)):
        raise ValueError("Installed torch-npu lacks the native FIA graph update interface")


def install(runner):
    import torch
    import torch_npu
    from vllm.config import set_current_vllm_config
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionBackendImpl
    from vllm_ascend.attention.utils import using_paged_attention
    from vllm_ascend.compilation.acl_graph import get_graph_params

    validate_config(runner.vllm_config)
    if not torch.npu.get_device_name(runner.device).startswith("Ascend910B"):
        raise ValueError("Native FIA update has only been validated on Ascend910B")
    if runner.attn_backend is not AscendAttentionBackend:
        raise ValueError("Native FIA update requires the standard Ascend attention backend")
    with set_current_vllm_config(runner.vllm_config):
        if runner.attn_backend.get_impl_cls() is not AscendAttentionBackendImpl:
            raise ValueError("Native FIA update requires the standard Ascend attention implementation")

    original_update = AscendAttentionBackendImpl.update_graph_params
    fused_update = torch_npu._C._npu_fused_infer_attention_score_out_graph
    stats = {"updates": 0, "layers": 28, "upstream_fallbacks": 0}

    class ASRFIAImpl(AscendAttentionBackendImpl):
        @staticmethod
        def update_graph_params(update_stream, forward_context, num_tokens, vllm_config,
                                speculative_config=None, num_dcp_pcp_tokens=None,
                                draft_attn_metadatas=None):
            if (_EXTRA_CTX.is_draft_model or speculative_config is not None
                    or draft_attn_metadatas is not None
                    or using_paged_attention(num_tokens, vllm_config)):
                stats["upstream_fallbacks"] += 1
                if stats["upstream_fallbacks"] == 1:
                    logger.info("ASR native FIA update: PA/draft path uses upstream graph update")
                return original_update(update_stream, forward_context, num_tokens, vllm_config,
                                       speculative_config, num_dcp_pcp_tokens, draft_attn_metadatas)

            graph = get_graph_params()
            metadata = forward_context.attn_metadata
            parameters = graph.attn_params[num_tokens]
            handles = graph.handles[num_tokens]
            events = graph.events[num_tokens]
            # Check the complete profile before modifying any graph task. Upstream
            # zip would silently truncate if its internal representation changed.
            if (len(metadata) != 28 or len(parameters) != 28 or len(handles) != 28
                    or len(events) != 28 or any(len(param) != 13 for param in parameters)):
                raise RuntimeError("ASR native FIA graph profile changed; expected 28 FIA layers")
            workspace = graph.workspaces.get(num_tokens)
            with torch.npu.stream(update_stream):
                for name, param, handle, event in zip(metadata, parameters, handles, events):
                    (query, key_cache, value, _block_tables, attn_mask, block_size,
                     _seq_lens, _query_start_loc, num_kv_heads, num_heads, scale,
                     attn_output, softmax_lse) = param
                    current = metadata[name]
                    fused_update(
                        update_stream, handle, event,
                        query=query, key=key_cache, value=value,
                        block_table=current.block_tables, atten_mask=attn_mask,
                        input_layout="TND", block_size=block_size,
                        actual_seq_lengths=current.actual_seq_lengths_q,
                        actual_seq_lengths_kv=current.seq_lens_list,
                        num_key_value_heads=num_kv_heads, num_heads=num_heads,
                        scale=scale, sparse_mode=3, workspace=workspace,
                        out=[attn_output, softmax_lse],
                    )
            stats["updates"] += 1
            if stats["updates"] == 1:
                logger.info("ASR native FIA graph update active: tokens=%d layers=28", num_tokens)

    class ASRFIABackend(AscendAttentionBackend):
        @staticmethod
        def get_impl_cls():
            return ASRFIAImpl

    # Only this runner's update_full_graph_params uses the derived backend.
    # Model attention instances and the process-wide registry retain upstream classes.
    runner.attn_backend = ASRFIABackend
    runner.asr_native_fia_update = stats
    logger.info("ASR native FIA graph update installed: 28 layers, torch-npu C++ interface, runner-local")
    return stats
