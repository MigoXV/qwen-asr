from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import textwrap
from typing import Any
import unittest


src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))

grpc_module = type(sys)("grpc")
grpc_module.aio = type("aio", (), {"server": staticmethod(lambda: None)})
sys.modules.setdefault("grpc", grpc_module)

typer_module = type(sys)("typer")


class _FakeTyper:
    def __init__(self, *args, **kwargs):
        pass

    def command(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


typer_module.Typer = _FakeTyper
typer_module.Option = lambda default, *args, **kwargs: default
sys.modules.setdefault("typer", typer_module)

proto_module = type(sys)("qwen_asr.protos.asr.ux_speech_pb2_grpc")
proto_module.add_UxSpeechServicer_to_server = lambda servicer, server: None
sys.modules.setdefault("qwen_asr.protos.asr.ux_speech_pb2_grpc", proto_module)

servicer_module = type(sys)("qwen_asr.servicer.servicer")
servicer_module.ASRServicer = type("ASRServicer", (), {})
sys.modules.setdefault("qwen_asr.servicer.servicer", servicer_module)

from qwen_asr.commands import app as app_module  # noqa: E402


class ConfigLoadingTest(unittest.TestCase):
    def _write_config(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(textwrap.dedent(content))
        tmp.close()
        return Path(tmp.name)

    def _load_config_via_serve(self, path: Path):
        captured: dict[str, Any] = {}

        async def fake_run_server(config):
            captured["config"] = config

        original_run_server = app_module.run_server
        app_module.run_server = fake_run_server
        try:
            app_module.serve(config=path)
        finally:
            app_module.run_server = original_run_server

        return captured["config"]

    def test_load_vllm_config(self):
        path = self._write_config(
            """
            model: /models/qwen3-asr
            backend: vllm
            server:
              port: 50052
            generation:
              max_new_tokens: 512
            device:
              mode: cpu
            vllm:
              gpu_memory_utilization: 0.5
              max_model_len: 1024
              enforce_eager: true
            transformers: null
            """
        )

        config = self._load_config_via_serve(path)

        self.assertEqual(config.model, "/models/qwen3-asr")
        self.assertEqual(config.backend, "vllm")
        self.assertEqual(config.server.port, 50052)
        self.assertEqual(config.generation.max_new_tokens, 512)
        self.assertEqual(config.device.mode, "cpu")
        self.assertIsNotNone(config.vllm)
        self.assertEqual(config.vllm.max_model_len, 1024)
        self.assertTrue(config.vllm.enforce_eager)

    def test_load_transformers_config_with_default_block(self):
        path = self._write_config(
            """
            model: /models/qwen3-asr
            backend: transformers
            context: domain prompt
            device:
              mode: cpu
            vllm: null
            transformers: {}
            """
        )

        config = self._load_config_via_serve(path)

        self.assertEqual(config.backend, "transformers")
        self.assertEqual(config.context, "domain prompt")
        self.assertIsNotNone(config.transformers)
        self.assertIsNone(config.vllm)

    def test_context_defaults_to_empty_string(self):
        path = self._write_config(
            """
            model: /models/qwen3-asr
            backend: vllm
            """
        )

        config = self._load_config_via_serve(path)

        self.assertEqual(config.context, "")

    def test_missing_active_backend_block_gets_defaults(self):
        path = self._write_config(
            """
            model: /models/qwen3-asr
            backend: vllm
            """
        )

        config = self._load_config_via_serve(path)

        self.assertIsNotNone(config.vllm)
        self.assertEqual(config.vllm.max_model_len, 4096)

    def test_unknown_key_fails(self):
        path = self._write_config(
            """
            model: /models/qwen3-asr
            backend: vllm
            unknown: true
            """
        )

        with self.assertRaises(Exception):
            self._load_config_via_serve(path)

    def test_invalid_backend_fails_clearly(self):
        path = self._write_config(
            """
            model: /models/qwen3-asr
            backend: bad
            """
        )

        with self.assertRaisesRegex(ValueError, "Invalid BACKEND value"):
            self._load_config_via_serve(path)

    def test_empty_backend_fails_clearly(self):
        path = self._write_config(
            """
            model: /models/qwen3-asr
            backend: ""
            """
        )

        with self.assertRaisesRegex(ValueError, "Invalid BACKEND value"):
            self._load_config_via_serve(path)

    def test_invalid_device_fails_clearly(self):
        path = self._write_config(
            """
            model: /models/qwen3-asr
            backend: vllm
            device:
              mode: tpu
            """
        )

        with self.assertRaisesRegex(ValueError, "Invalid DEVICE value"):
            self._load_config_via_serve(path)


if __name__ == "__main__":
    unittest.main()
