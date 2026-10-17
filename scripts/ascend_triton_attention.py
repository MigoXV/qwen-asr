"""仅诊断 Worker 使用的模型局部 Triton 分页解码注意力。"""

from types import MethodType

import torch

from experimental_paged_attention import make_workspace, paged_attention


def install(runner, *, block=128, grouped=False):
    from vllm.config.compilation import CUDAGraphMode
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState

    config = runner.vllm_config
    if (config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
            or config.speculative_config is not None
            or config.model_config.dtype != torch.bfloat16
            or config.scheduler_config.max_num_seqs > 8
            or config.model_config.max_model_len > 1024
            or config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1):
        raise ValueError('Triton PA prototype requires BF16 FULL_DECODE_ONLY TP1/PP1 <=8 slots <=1024 context')
    installed = []
    workspaces = {}
    for name, layer in config.compilation_config.static_forward_context.items():
        impl = getattr(layer, 'impl', None)
        if not isinstance(impl, AscendAttentionBackendImpl):
            continue
        if (type(impl) is not AscendAttentionBackendImpl or impl.num_heads != 16
                or impl.num_kv_heads != 8 or impl.head_size != 128
                or impl.sliding_window is not None or impl.sinks is not None):
            raise ValueError(f'Unsupported attention layer {name}')
        original = impl.forward_impl

        def forward(self, query, key, value, kv_cache, metadata, output,
                    _original=original, _name=name):
            if metadata.attn_state != AscendAttentionState.DecodeOnly or query.shape[0] > 8:
                return _original(query, key, value, kv_cache, metadata, output)
            batch = query.shape[0]
            q = query.view(batch, 16, 128)
            out = output.view(batch, 16, 128)
            cache_key = (_name, batch)
            if cache_key not in workspaces:
                workspaces[cache_key] = make_workspace(batch, 16, 128, 1024, q.device, block)
            paged_attention(q, self.key_cache, self.value_cache, metadata.block_tables,
                            runner.seq_lens.gpu[:batch], out, workspaces[cache_key], block, grouped)
            return output

        impl.forward_impl = MethodType(forward, impl)
        installed.append(name)
    if len(installed) != 28:
        raise ValueError(f'Expected 28 ASR decoder layers, found {len(installed)}')
    # Compile/load kernels outside graph capture. The kernel specializes strides,
    # not the total number of KV pages, so a small warmup cache is sufficient.
    q = torch.zeros(8,16,128,device=runner.device,dtype=torch.bfloat16)
    k = torch.zeros(1,128,8,128,device=runner.device,dtype=torch.bfloat16)
    table = torch.zeros(8,8,device=runner.device,dtype=torch.int32)
    lengths = torch.ones(8,device=runner.device,dtype=torch.int32)
    output = torch.empty_like(q)
    workspace = make_workspace(8,16,128,1024,runner.device,block)
    paged_attention(q,k,k,table,lengths,output,workspace,block,grouped)
    torch.npu.synchronize()
    runner.asr_triton_decode = {'layers':installed,'block':block,'grouped':grouped}
    return runner.asr_triton_decode
