from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(find_spec('vllm_ascend') is not None,'requires vllm-ascend')
class TritonDecodeConfigTest(unittest.TestCase):
    def make_config(self):
        from vllm.config.compilation import CUDAGraphMode

        return SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type='qwen3_asr'),
                hf_text_config=SimpleNamespace(num_hidden_layers=28,num_attention_heads=16,
                                              num_key_value_heads=8,head_dim=128),
                enforce_eager=False,max_model_len=1024),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
            speculative_config=None,scheduler_config=SimpleNamespace(max_num_seqs=8),
            parallel_config=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1),
        )

    def test_rejects_unsupported_model_parallelism_capacity_and_context(self):
        from qwen_asr.engines.vllm.ascend.triton_decode import validate_config

        for path,value in [('model_config.max_model_len',2048),
                           ('model_config.enforce_eager',True),
                           ('scheduler_config.max_num_seqs',16),
                           ('parallel_config.tensor_parallel_size',2),
                           ('parallel_config.data_parallel_size',2),
                           ('model_config.hf_text_config.head_dim',64)]:
            config=self.make_config()
            fields=path.split('.')
            parent=config
            for field in fields[:-1]:parent=getattr(parent,field)
            setattr(parent,fields[-1],value)
            with self.subTest(path=path), self.assertRaises(ValueError):validate_config(config)

    def test_requires_explicitly_validated_toolchain(self):
        from qwen_asr.engines.vllm.ascend.triton_decode import validate_config

        versions={'vllm':'0.18.0+local','torch-npu':'2.9.1','triton-ascend':'3.2.2'}
        with patch('qwen_asr.engines.vllm.ascend.triton_decode.version',side_effect=versions.__getitem__):
            validate_config(self.make_config())
            versions['triton-ascend']='3.6.0'
            with self.assertRaisesRegex(ValueError,'versions not validated'):validate_config(self.make_config())

    def test_graph_reuses_changed_contents_but_falls_back_for_changed_addresses(self):
        import torch
        from unittest.mock import Mock
        from vllm.config.compilation import CUDAGraphMode
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState
        from qwen_asr.engines.vllm.ascend import paged_decode
        from qwen_asr.engines.vllm.ascend.triton_decode import install

        config=self.make_config()
        config.model_config.dtype=torch.bfloat16
        layers={}
        for i in range(28):
            impl=AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
            impl.num_heads,impl.num_kv_heads,impl.head_size=16,8,128
            impl.sliding_window=impl.sinks=None
            impl.key_cache=torch.empty(1)
            impl.value_cache=torch.empty(1)
            impl.forward_impl=Mock()
            layers[str(i)]=SimpleNamespace(impl=impl)
        config.compilation_config.static_forward_context=layers
        original=Mock()
        runner=SimpleNamespace(vllm_config=config,device=torch.device('cpu'),
            seq_lens=SimpleNamespace(gpu=torch.ones(8,dtype=torch.int32)),
            _update_full_graph_params_if_needed=original)
        metadata=SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly,
                                 block_tables=torch.zeros(1,8,dtype=torch.int32))
        query=torch.zeros(1,16,128,dtype=torch.bfloat16)
        with patch('qwen_asr.engines.vllm.ascend.triton_decode.validate_config'), \
             patch.object(torch.npu,'get_device_name',return_value='Ascend910B4'), \
             patch.object(torch.npu,'synchronize'), \
             patch.object(paged_decode,'paged_attention'), \
             patch('vllm_ascend.ascend_forward_context._EXTRA_CTX',SimpleNamespace(capturing=True)):
            install(runner)
            for layer in layers.values():
                layer.impl.forward_impl(query,None,None,None,metadata,query.clone())
        context=SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL,capturing=False,
                                 attn_metadata={name:metadata for name in layers})
        runner._update_full_graph_params_if_needed(context,1,None)
        self.assertEqual(context.cudagraph_runtime_mode,CUDAGraphMode.FULL)
        metadata.block_tables.add_(1)
        runner.seq_lens.gpu.add_(1)
        runner._update_full_graph_params_if_needed(context,1,None)
        self.assertEqual(context.cudagraph_runtime_mode,CUDAGraphMode.FULL)
        metadata.block_tables=metadata.block_tables.clone()
        runner._update_full_graph_params_if_needed(context,1,None)
        self.assertEqual(context.cudagraph_runtime_mode,CUDAGraphMode.NONE)
        self.assertEqual(runner.asr_triton_decode['eager_fallbacks'],1)
        original.assert_not_called()
