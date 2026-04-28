from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

from tests.grpc_stub import install_fake_grpc


install_fake_grpc()

_FAKES = {}


def _install_fake_module(name: str, module: types.ModuleType) -> None:
    _FAKES[name] = sys.modules.get(name)
    sys.modules[name] = module


fake_qwen_asr = types.ModuleType("qwen_asr")
fake_qwen_asr.__path__ = []
_install_fake_module("qwen_asr", fake_qwen_asr)

fake_inference_pkg = types.ModuleType("qwen_asr.inference")
fake_inference_pkg.__path__ = []
_install_fake_module("qwen_asr.inference", fake_inference_pkg)

fake_inference = types.ModuleType("qwen_asr.inference.qwen3_asr")


class FakeQwen3ASRModel:
    pass


fake_inference.Qwen3ASRModel = FakeQwen3ASRModel
_install_fake_module("qwen_asr.inference.qwen3_asr", fake_inference)

fake_protos_pkg = types.ModuleType("qwen_asr.protos")
fake_protos_pkg.__path__ = []
_install_fake_module("qwen_asr.protos", fake_protos_pkg)

fake_asr_pkg = types.ModuleType("qwen_asr.protos.asr")
fake_asr_pkg.__path__ = []
_install_fake_module("qwen_asr.protos.asr", fake_asr_pkg)

fake_pb2_grpc = types.ModuleType("qwen_asr.protos.asr.ux_speech_pb2_grpc")
fake_pb2_grpc.add_UxSpeechServicer_to_server = lambda *args, **kwargs: None
_install_fake_module("qwen_asr.protos.asr.ux_speech_pb2_grpc", fake_pb2_grpc)

fake_servicer_pkg = types.ModuleType("qwen_asr.servicer")
fake_servicer_pkg.__path__ = []
_install_fake_module("qwen_asr.servicer", fake_servicer_pkg)

fake_servicer = types.ModuleType("qwen_asr.servicer.servicer")
fake_servicer.ASRServicer = type("ASRServicer", (), {})
_install_fake_module("qwen_asr.servicer.servicer", fake_servicer)

app_path = Path(__file__).resolve().parents[1] / "src/qwen_asr/commands/app.py"
spec = importlib.util.spec_from_file_location("_test_qwen_asr_commands_app", app_path)
app_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(app_module)

for module_name, previous_module in _FAKES.items():
    if previous_module is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous_module


class AppDeviceSelectionTest(unittest.TestCase):
    def test_auto_with_cuda_keeps_gpu_defaults(self):
        kwargs = app_module._build_llm_kwargs(
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
        kwargs = app_module._build_llm_kwargs(
            max_new_tokens=512,
            gpu_memory_utilization=0.5,
            max_model_len=4096,
            device="auto",
            cuda_available=False,
        )

        self.assertEqual(kwargs["device"], "cpu")

    def test_cuda_without_cuda_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "DEVICE=cuda was requested"):
            app_module._build_llm_kwargs(
                max_new_tokens=512,
                gpu_memory_utilization=0.5,
                max_model_len=4096,
                device="cuda",
                cuda_available=False,
            )

    def test_cpu_always_uses_cpu(self):
        kwargs = app_module._build_llm_kwargs(
            max_new_tokens=512,
            gpu_memory_utilization=0.5,
            max_model_len=4096,
            device="cpu",
            cuda_available=True,
        )

        self.assertEqual(kwargs["device"], "cpu")

    def test_invalid_device_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Invalid DEVICE value"):
            app_module._build_llm_kwargs(
                max_new_tokens=512,
                gpu_memory_utilization=0.5,
                max_model_len=4096,
                device="tpu",
                cuda_available=True,
            )


if __name__ == "__main__":
    unittest.main()
