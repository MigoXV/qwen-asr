from contextlib import nullcontext
from importlib.util import find_spec
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch


@skipUnless(find_spec("vllm_ascend") is not None, "requires vllm-ascend")
class NativeFIAUpdateTest(TestCase):
    def make_config(self):
        import torch
        from vllm.config.compilation import CUDAGraphMode

        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen3_asr"),
                hf_text_config=SimpleNamespace(num_hidden_layers=28, num_attention_heads=16,
                                              num_key_value_heads=8, head_dim=128),
                dtype=torch.bfloat16, enforce_eager=False, max_model_len=1024),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
            scheduler_config=SimpleNamespace(max_num_seqs=8),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
                                            data_parallel_size=1),
            speculative_config=None, kv_transfer_config=None,
        )

    def test_rejects_unvalidated_runtime_and_parallelism(self):
        from qwen_asr.engines.vllm.ascend.fia_update import validate_config

        versions = {"torch": "2.9.1+cpu", "torch-npu": "2.9.1",
                    "vllm": "0.18.0+empty", "vllm-ascend": "0.18.0+torch291.cann911"}
        with patch("qwen_asr.engines.vllm.ascend.fia_update.version", side_effect=versions.__getitem__):
            validate_config(self.make_config())
            config = self.make_config()
            config.parallel_config.decode_context_parallel_size = 2
            with self.assertRaisesRegex(ValueError, "TP/PP/DP/CP1"):
                validate_config(config)
            versions["torch-npu"] = "2.8.0"
            with self.assertRaisesRegex(ValueError, "versions not validated"):
                validate_config(self.make_config())

    def test_uses_live_metadata_and_keeps_backend_local_to_runner(self):
        import torch
        import torch_npu
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionBackendImpl
        from qwen_asr.engines.vllm.ascend.fia_update import install

        runner = SimpleNamespace(vllm_config=self.make_config(), device=torch.device("cpu"),
                                 attn_backend=AscendAttentionBackend)
        other_runner = SimpleNamespace(attn_backend=AscendAttentionBackend)
        metadata = {str(i): SimpleNamespace(block_tables=object(), seq_lens_list=[127],
                                            actual_seq_lengths_q=[1]) for i in range(28)}
        parameters = [(object(), object(), object(), "stale table", object(), 128,
                       "stale lengths", None, 8, 16, 0.125, object(), object()) for _ in range(28)]
        graph = SimpleNamespace(attn_params={1: parameters}, handles={1: list(range(28))},
                                events={1: [object() for _ in range(28)]}, workspaces={1: object()})
        context = SimpleNamespace(attn_metadata=metadata)
        native = Mock()
        original = Mock()
        with (
            patch.object(AscendAttentionBackendImpl, "update_graph_params", original),
            patch("vllm_ascend.attention.attention_v1.enable_cp", return_value=False),
            patch("vllm_ascend.ascend_forward_context._EXTRA_CTX", SimpleNamespace(is_draft_model=False)),
            patch("vllm_ascend.compilation.acl_graph.get_graph_params", return_value=graph),
            patch("vllm_ascend.attention.utils.using_paged_attention", return_value=False) as paged,
            patch.object(torch.npu, "get_device_name", return_value="Ascend910B4-1"),
            patch.object(torch.npu, "stream", return_value=nullcontext()),
            patch.object(torch_npu._C, "_npu_fused_infer_attention_score_out_graph", native),
        ):
            stats = install(runner)
            update = runner.attn_backend.get_impl_cls().update_graph_params
            update("stream", context, 1, runner.vllm_config)
            self.assertEqual(native.call_count, 28)
            self.assertEqual(native.call_args_list[0].kwargs["actual_seq_lengths_kv"], [127])
            self.assertIs(native.call_args_list[0].kwargs["block_table"], metadata["0"].block_tables)
            self.assertIs(native.call_args_list[0].kwargs["workspace"], graph.workspaces[1])
            new_table = object()
            metadata["0"].block_tables = new_table
            metadata["0"].seq_lens_list = [129]
            update("stream", context, 1, runner.vllm_config)
            self.assertIs(native.call_args_list[28].kwargs["block_table"], new_table)
            self.assertEqual(native.call_args_list[28].kwargs["actual_seq_lengths_kv"], [129])
            self.assertEqual(stats["updates"], 2)
            self.assertIs(other_runner.attn_backend, AscendAttentionBackend)
            self.assertIs(AscendAttentionBackend.get_impl_cls(), AscendAttentionBackendImpl)
            self.assertIs(AscendAttentionBackendImpl.update_graph_params, original)
            original.assert_not_called()

            # A malformed profile fails before any task is updated.
            graph.handles[1].pop()
            with self.assertRaisesRegex(RuntimeError, "expected 28 FIA layers"):
                update("stream", context, 1, runner.vllm_config)
            self.assertEqual(native.call_count, 56)

            # PA graphs use their original updater even when FIA profiles are absent.
            paged.return_value = True
            update("stream", context, 1, runner.vllm_config)
            original.assert_called_once()
            self.assertEqual(stats["upstream_fallbacks"], 1)
