from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


utils_path = Path(__file__).resolve().parents[1] / "src/qwen_asr/commands/utils.py"
spec = importlib.util.spec_from_file_location("_test_qwen_asr_commands_utils", utils_path)
command_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(command_utils)

src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))
from qwen_asr.configs import (  # noqa: E402
    AppConfig,
    DeviceConfig,
    GenerationConfig,
    VLLMConfig,
)


class AppDeviceSelectionTest(unittest.TestCase):
    def test_cpu_always_uses_cpu(self):
        config = AppConfig(
            model="/tmp/model",
            generation=GenerationConfig(max_new_tokens=512),
            device=DeviceConfig(mode="cpu"),
            vllm=VLLMConfig(gpu_memory_utilization=0.5, max_model_len=4096),
        )
        kwargs = command_utils.build_vllm_kwargs(config)

        self.assertEqual(kwargs["device"], "cpu")

    def test_enforce_eager_is_omitted_by_default(self):
        config = AppConfig(
            model="/tmp/model",
            generation=GenerationConfig(max_new_tokens=512),
            device=DeviceConfig(mode="cpu"),
            vllm=VLLMConfig(gpu_memory_utilization=0.5, max_model_len=4096),
        )
        kwargs = command_utils.build_vllm_kwargs(config)

        self.assertNotIn("enforce_eager", kwargs)

    def test_enforce_eager_is_added_when_enabled(self):
        config = AppConfig(
            model="/tmp/model",
            generation=GenerationConfig(max_new_tokens=512),
            device=DeviceConfig(mode="cpu"),
            vllm=VLLMConfig(
                gpu_memory_utilization=0.5,
                max_model_len=4096,
                enforce_eager=True,
            ),
        )
        kwargs = command_utils.build_vllm_kwargs(config)

        self.assertTrue(kwargs["enforce_eager"])

    def test_cpu_and_enforce_eager_can_be_used_together(self):
        config = AppConfig(
            model="/tmp/model",
            generation=GenerationConfig(max_new_tokens=512),
            device=DeviceConfig(mode="cpu"),
            vllm=VLLMConfig(
                gpu_memory_utilization=0.5,
                max_model_len=4096,
                enforce_eager=True,
            ),
        )
        kwargs = command_utils.build_vllm_kwargs(config)

        self.assertEqual(kwargs["device"], "cpu")
        self.assertTrue(kwargs["enforce_eager"])

    def test_invalid_device_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Invalid DEVICE value"):
            AppConfig(
                model="/tmp/model",
                generation=GenerationConfig(max_new_tokens=512),
                device=DeviceConfig(mode="tpu"),
                vllm=VLLMConfig(gpu_memory_utilization=0.5, max_model_len=4096),
            )


if __name__ == "__main__":
    unittest.main()
