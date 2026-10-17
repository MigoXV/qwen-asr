from importlib.util import find_spec
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import Mock, patch


@skipUnless(find_spec("vllm_ascend") is not None, "requires vllm-ascend")
class SharedNPUMemoryTest(TestCase):
    def make_worker(self):
        from qwen_asr.engines.vllm.ascend.worker import SharedNPUWorker

        worker = SharedNPUWorker.__new__(SharedNPUWorker)
        worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=1024 ** 3,
                                             gpu_memory_utilization=0.09)
        worker.model_runner = Mock()
        return worker

    def test_other_process_memory_is_not_charged_to_this_worker(self):
        from qwen_asr.engines.vllm.ascend import worker as module

        worker = self.make_worker()
        with (
            patch.object(module.torch_npu.npu, "reset_peak_memory_stats"),
            patch.object(module.torch_npu.npu, "empty_cache"),
            # Only 10 GiB globally free, but this process owns a 3 GiB peak.
            patch.object(module.torch_npu.npu, "mem_get_info", return_value=(10 * 1024 ** 3, 64 * 1024 ** 3)),
            patch.object(module.torch_npu.npu, "memory_stats", return_value={"allocated_bytes.all.peak": 3 * 1024 ** 3}),
        ):
            self.assertEqual(worker.determine_available_memory(), 1024 ** 3)
        worker.model_runner.profile_run.assert_called_once()

    def test_rejects_worker_budget_overcommit(self):
        from qwen_asr.engines.vllm.ascend import worker as module

        worker = self.make_worker()
        with (
            patch.object(module.torch_npu.npu, "reset_peak_memory_stats"),
            patch.object(module.torch_npu.npu, "empty_cache"),
            patch.object(module.torch_npu.npu, "mem_get_info", return_value=(10 * 1024 ** 3, 64 * 1024 ** 3)),
            patch.object(module.torch_npu.npu, "memory_stats", return_value={"allocated_bytes.all.peak": 6 * 1024 ** 3}),
        ):
            with self.assertRaisesRegex(ValueError, "exceeds process budget"):
                worker.determine_available_memory()
