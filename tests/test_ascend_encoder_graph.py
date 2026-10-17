"""Real graph replay, metadata ownership and bounded cache validation."""

import os
import threading
import unittest


@unittest.skipUnless(os.environ.get('QWEN_ASR_TEST_NPU') == '1', 'requires Ascend NPU')
class EncoderGraphCacheTest(unittest.TestCase):
    def test_dynamic_contents_boundaries_output_ownership_and_eviction(self):
        import torch
        import torch_npu
        from qwen_asr.models.vllm_ascend.encoder_graph import EncoderGraphCache

        def forward(x, lengths, maximum):
            return x * int(lengths[1]) + x.sum(-1, keepdim=True)

        cache = EncoderGraphCache(forward)
        with torch.inference_mode():
            x = torch.randn(8,1024,device='npu',dtype=torch.bfloat16)
            boundaries = torch.tensor([0,3,8],dtype=torch.int64)
            cache(x,boundaries,None)
            saved = cache(x,boundaries,None)
            expected_saved = saved.clone()
            x.add_(1)
            actual = cache(x,boundaries,None)
            torch.testing.assert_close(actual,forward(x,boundaries,None),atol=0,rtol=0)
            torch.testing.assert_close(saved,expected_saved,atol=0,rtol=0)
            # Same shape, different window boundaries must not reuse stale tiling.
            boundaries[1] = 5
            cache(x,boundaries,None)
            changed = cache(x,boundaries,None)
            torch.testing.assert_close(changed,forward(x,boundaries,None),atol=0,rtol=0)
            self.assertEqual(cache.stats['captures'],2)
            boundaries[1] = 3
            torch.testing.assert_close(cache(x,boundaries,None),forward(x,boundaries,None),atol=0,rtol=0)
            for size in range(9,15):
                data = torch.randn(size,1024,device='npu',dtype=torch.bfloat16)
                lengths = torch.tensor([0,size],dtype=torch.int32)
                cache(data,lengths,None)
                torch.testing.assert_close(cache(data,lengths,None),forward(data,lengths,None),atol=0,rtol=0)
            self.assertEqual(len(cache.entries),4)
            self.assertEqual(cache.stats['evictions'],4)
            for size in range(20,90):
                cache(torch.zeros(size,1024,device='npu',dtype=torch.bfloat16),torch.tensor([0,size]),None)
            self.assertEqual(len(cache.seen),64)
            self.assertEqual(len(cache.entries),4)
            with self.assertRaisesRegex(ValueError,'partition'):
                cache(x,torch.tensor([0,3,7]),None)
            # Oversized shapes execute eagerly and cannot grow graph storage.
            large = torch.zeros(1025,1024,device='npu',dtype=torch.bfloat16)
            cache(large,torch.tensor([0,1025]),None)
            self.assertEqual(len(cache.entries),4)
        failures = []
        def other_thread():
            try:
                cache(None,None,None)
            except RuntimeError as error:
                failures.append(str(error))
        thread = threading.Thread(target=other_thread)
        thread.start();thread.join()
        self.assertEqual(len(failures),1)
        self.assertIn('owning Worker',failures[0])

    def test_v3_attention_changed_inputs_match_original_packed_attention(self):
        import torch
        import torch_npu
        from qwen_asr.models.vllm_ascend.qwen3_asr import AscendEncoderAttention

        module = AscendEncoderAttention(16,64**-.5)
        with torch.inference_mode():
            for ends in ([0,1,7,70],[0,104,208,312,390],[0,104,208,312,390,494,598,702,780]):
                lengths = torch.tensor(ends,dtype=torch.int64)
                q,k,v = [torch.randn(1,ends[-1],16,64,device='npu',dtype=torch.bfloat16) for _ in range(3)]
                module.graph_attention = False
                expected = module(q,k,v,lengths)
                module.graph_attention = True
                module(q,k,v,lengths)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    actual = module(q,k,v,lengths)
                for _ in range(3):
                    q.normal_();k.normal_();v.normal_()
                    module.graph_attention = False
                    expected = module(q,k,v,lengths)
                    module.graph_attention = True
                    graph.replay()
                    torch.npu.synchronize()
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
