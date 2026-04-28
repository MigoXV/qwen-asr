from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = ROOT / "src/qwen_asr/commands/utils.py"
APP_PATH = ROOT / "src/qwen_asr/commands/app.py"


def _load_command_utils():
    spec = importlib.util.spec_from_file_location(
        "_test_qwen_asr_commands_utils_for_app",
        UTILS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_app_module():
    command_utils = _load_command_utils()

    typer_module = types.ModuleType("typer")

    class _FakeTyper:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def command(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    def _fake_option(default, *args, **kwargs):
        return default

    typer_module.Typer = _FakeTyper
    typer_module.Option = _fake_option
    typer_module.Exit = RuntimeError

    grpc_module = types.ModuleType("grpc")
    grpc_module.aio = types.SimpleNamespace(server=lambda: None)

    qwen_asr_pkg = types.ModuleType("qwen_asr")
    qwen_asr_pkg.__path__ = []
    commands_pkg = types.ModuleType("qwen_asr.commands")
    commands_pkg.__path__ = []
    inference_pkg = types.ModuleType("qwen_asr.inference")
    inference_pkg.__path__ = []
    protos_pkg = types.ModuleType("qwen_asr.protos")
    protos_pkg.__path__ = []
    asr_pkg = types.ModuleType("qwen_asr.protos.asr")
    asr_pkg.__path__ = []
    servicer_pkg = types.ModuleType("qwen_asr.servicer")
    servicer_pkg.__path__ = []

    qwen3_asr_module = types.ModuleType("qwen_asr.inference.qwen3_asr")
    qwen3_asr_module.Qwen3ASRModel = type("Qwen3ASRModel", (), {})

    proto_module = types.ModuleType("qwen_asr.protos.asr.ux_speech_pb2_grpc")
    proto_module.add_UxSpeechServicer_to_server = lambda servicer, server: None

    servicer_module = types.ModuleType("qwen_asr.servicer.servicer")
    servicer_module.ASRServicer = type("ASRServicer", (), {})

    modules = {
        "typer": typer_module,
        "grpc": grpc_module,
        "qwen_asr": qwen_asr_pkg,
        "qwen_asr.commands": commands_pkg,
        "qwen_asr.commands.utils": command_utils,
        "qwen_asr.inference": inference_pkg,
        "qwen_asr.inference.qwen3_asr": qwen3_asr_module,
        "qwen_asr.protos": protos_pkg,
        "qwen_asr.protos.asr": asr_pkg,
        "qwen_asr.protos.asr.ux_speech_pb2_grpc": proto_module,
        "qwen_asr.servicer": servicer_pkg,
        "qwen_asr.servicer.servicer": servicer_module,
    }

    previous_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_qwen_asr_commands_app",
            APP_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class AppEnforceEagerTest(unittest.TestCase):
    def test_serve_threads_enforce_eager_to_serve_async(self):
        app_module = _load_app_module()
        captured = {}

        async def fake_serve_async(**kwargs):
            captured.update(kwargs)

        original_run = app_module.asyncio.run
        original_serve_async = app_module._serve_async
        app_module._serve_async = fake_serve_async
        app_module.asyncio.run = asyncio.run
        try:
            app_module.serve(
                model="/tmp/model",
                port=50051,
                max_new_tokens=512,
                gpu_memory_utilization=0.5,
                max_model_len=1024,
                device="cpu",
                enforce_eager=True,
            )
        finally:
            app_module.asyncio.run = original_run
            app_module._serve_async = original_serve_async

        self.assertTrue(captured["enforce_eager"])
        self.assertEqual(captured["device"], "cpu")
        self.assertEqual(captured["max_model_len"], 1024)


if __name__ == "__main__":
    unittest.main()
