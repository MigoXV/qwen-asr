"""将经过验证的解码 kernel 安装到当前 ASR 模型实例，不修改全局后端。"""

from importlib.metadata import version
import logging
from types import MethodType


def validate_config(config):
    from vllm.config.compilation import CUDAGraphMode

    text = config.model_config.hf_text_config
    if (getattr(config.model_config.hf_config, 'model_type', None) != 'qwen3_asr'
            or (text.num_hidden_layers, text.num_attention_heads, text.num_key_value_heads,
                text.head_dim) != (28,16,8,128)):
        raise ValueError('Triton decode requires Qwen3-ASR-1.7B (28 layers, Q16/KV8/D128)')
    if (config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
            or config.model_config.enforce_eager
            or config.speculative_config is not None
            or config.scheduler_config.max_num_seqs > 8
            or config.model_config.max_model_len != 1024
            or config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1
            or config.parallel_config.data_parallel_size != 1):
        raise ValueError('Triton decode requires FULL_DECODE_ONLY, <=8 slots, context 1024, TP/PP/DP1, no speculative decoding')
    expected = {'vllm':'0.18.0', 'torch-npu':'2.9.1', 'triton-ascend':'3.2.2'}
    actual = {name:version(name).split('+')[0] for name in expected}
    if actual != expected:
        raise ValueError(f'Triton decode versions not validated: {actual}; expected {expected}')


def tensor_signature(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


def install(runner):
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState
    from qwen_asr.engines.vllm.ascend.paged_decode import make_workspace, paged_attention

    validate_config(runner.vllm_config)
    if runner.vllm_config.model_config.dtype != torch.bfloat16 or not torch.npu.get_device_name(runner.device).startswith('Ascend910B'):
        raise ValueError('Triton decode requires Ascend910B and BF16')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    config = runner.vllm_config
    layers = []
    workspaces = {}
    profiles = {}
    stats = {'layers':0, 'block':64, 'grouped':True, 'capture_sizes':[], 'eager_fallbacks':0}
    for name, layer in config.compilation_config.static_forward_context.items():
        impl = getattr(layer, 'impl', None)
        if not isinstance(impl, AscendAttentionBackendImpl):
            continue
        if (type(impl) is not AscendAttentionBackendImpl or impl.num_heads != 16
                or impl.num_kv_heads != 8 or impl.head_size != 128
                or impl.sliding_window is not None or impl.sinks is not None):
            raise ValueError(f'Triton decode unsupported attention layer: {name}')
        original = impl.forward_impl

        def forward(self, query, key, value, kv_cache, metadata, output,
                    _original=original, _name=name):
            if metadata.attn_state != AscendAttentionState.DecodeOnly or query.shape[0] > 8:
                return _original(query,key,value,kv_cache,metadata,output)
            batch = query.shape[0]
            q, out = query.view(batch,16,128), output.view(batch,16,128)
            cache_key = (_name,batch)
            if cache_key not in workspaces:
                workspaces[cache_key] = make_workspace(batch,16,128,1024,q.device)
            # Retain a signature for each exact graph profile. Runtime metadata
            # normally points into persistent runner buffers; otherwise use eager.
            from vllm_ascend.ascend_forward_context import _EXTRA_CTX
            if _EXTRA_CTX.capturing:
                if batch not in stats['capture_sizes']:
                    stats['capture_sizes'].append(batch)
                profiles.setdefault(batch,{})[_name] = (
                    tensor_signature(metadata.block_tables),
                    tensor_signature(runner.seq_lens.gpu[:batch]),
                    tensor_signature(self.key_cache), tensor_signature(self.value_cache), self)
            paged_attention(q,self.key_cache,self.value_cache,metadata.block_tables,
                            runner.seq_lens.gpu[:batch],out,workspaces[cache_key])
            return output

        impl.forward_impl = MethodType(forward,impl)
        layers.append(name)
    if len(layers) != 28:
        raise ValueError(f'Triton decode expected 28 layers, found {len(layers)}')

    original_update = runner._update_full_graph_params_if_needed

    def guarded_update(_runner, context, tokens, positions):
        if context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not context.capturing:
            profile = profiles.get(tokens,{})
            valid = len(profile) == len(layers)
            sequence_signature = tensor_signature(runner.seq_lens.gpu[:tokens])
            table_signatures = {}
            for name, (table,seq,key,value,impl) in profile.items():
                current = context.attn_metadata.get(name)
                if current is not None and id(current) not in table_signatures:
                    table_signatures[id(current)] = tensor_signature(current.block_tables)
                if (current is None or table_signatures[id(current)] != table
                        or sequence_signature != seq
                        or tensor_signature(impl.key_cache) != key
                        or tensor_signature(impl.value_cache) != value):
                    valid = False
                    break
            if not valid:
                context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                stats['eager_fallbacks'] += 1
                if stats['eager_fallbacks'] == 1:
                    logger.warning('ASR Triton decode graph fallback: metadata buffer signature changed, tokens=%d',tokens)
            # Triton graph nodes read device lengths and block tables directly.
            # No ExternalEvent/task-update nodes were captured for these layers.
            return
        return original_update(context,tokens,positions)

    runner._update_full_graph_params_if_needed = MethodType(guarded_update,runner)
    # Compile and load the kernels before graph capture. KV page count is not a
    # specialization argument, so this small cache has the production strides.
    q = torch.zeros(8,16,128,device=runner.device,dtype=torch.bfloat16)
    k = torch.zeros(1,128,8,128,device=runner.device,dtype=torch.bfloat16)
    table = torch.zeros(8,8,device=runner.device,dtype=torch.int32)
    lengths = torch.ones(8,device=runner.device,dtype=torch.int32)
    out = torch.empty_like(q)
    paged_attention(q,k,k,table,lengths,out,make_workspace(8,16,128,1024,runner.device))
    torch.npu.synchronize()
    stats['layers'] = len(layers)
    runner.asr_triton_decode = stats
    logger.info('ASR Triton decode installed: layers=%d BF16 Q16/KV8/D128 block=64 context=1024 graph=FULL_DECODE_ONLY',len(layers))
    return stats
