from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


utils_path = Path(__file__).resolve().parents[1] / "src/qwen_asr/commands/utils.py"
spec = importlib.util.spec_from_file_location("_test_qwen_asr_commands_utils", utils_path)
command_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(command_utils)


class AppDeviceSelectionTest(unittest.TestCase):
    def test_auto_with_cuda_keeps_gpu_defaults(self):
        kwargs = command_utils.build_llm_kwargs(
            max_new_tokens=512,
            gpu_memory_utilization=0.5,
            max_model_len=4096,
            device="auto",
            cuda_available=True,
        )

        self.assertEqual(kwargs["max_new_tokens"], 512)
        self.assertEqual(kwargs["gpu_memory_utilization"], 0.5)
        self.assertEqual(kwargs["max_model_len"], 4096)
        self.assertNotIn("device", kwargs)

    def test_auto_without_cuda_uses_cpu(self):
        kwargs = command_utils.build_llm_kwargs(
            max_new_tokens=512,
            gpu_memory_utilization=0.5,
            max_model_len=4096,
            device="auto",
            cuda_available=False,
        )

        self.assertEqual(kwargs["device"], "cpu")

    def test_cuda_without_cuda_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "DEVICE=cuda was requested"):
            command_utils.build_llm_kwargs(
                max_new_tokens=512,
                gpu_memory_utilization=0.5,
                max_model_len=4096,
                device="cuda",
                cuda_available=False,
            )

    def test_cpu_always_uses_cpu(self):
        kwargs = command_utils.build_llm_kwargs(
            max_new_tokens=512,
            gpu_memory_utilization=0.5,
            max_model_len=4096,
            device="cpu",
            cuda_available=True,
        )

        self.assertEqual(kwargs["device"], "cpu")

    def test_invalid_device_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Invalid DEVICE value"):
            command_utils.build_llm_kwargs(
                max_new_tokens=512,
                gpu_memory_utilization=0.5,
                max_model_len=4096,
                device="tpu",
                cuda_available=True,
            )


if __name__ == "__main__":
    unittest.main()
