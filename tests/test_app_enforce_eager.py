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
    configs_module = types.ModuleType("qwen_asr.configs")
    configs_module.AppConfig = type("AppConfig", (), {})
    configs_module.TransformersConfig = type("TransformersConfig", (), {})
    configs_module.VLLMConfig = type("VLLMConfig", (), {})
    constants_module = types.ModuleType("qwen_asr.configs.constants")
    constants_module.BACKEND_TRANSFORMERS = "transformers"
    constants_module.BACKEND_VLLM = "vllm"

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
        "qwen_asr.configs": configs_module,
        "qwen_asr.configs.constants": constants_module,
        "qwen_asr.inference": inference_pkg,
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
    def test_serve_loads_config_before_starting_server(self):
        app_module = _load_app_module()
        captured = {}
        loaded_config = types.SimpleNamespace(
            model="/tmp/model",
            backend="vllm",
            server=types.SimpleNamespace(port=50051),
            generation=types.SimpleNamespace(max_new_tokens=512),
            device=types.SimpleNamespace(mode="cpu"),
            vllm=types.SimpleNamespace(max_model_len=1024, enforce_eager=True),
        )

        class FakeOmegaConf:
            @staticmethod
            def structured(schema):
                return schema

            @staticmethod
            def load(path):
                captured["path"] = path
                return {"loaded": True}

            @staticmethod
            def merge(schema, loaded):
                captured["merged"] = (schema, loaded)
                return loaded

            @staticmethod
            def to_object(merged):
                return loaded_config

        async def fake_run_server(config):
            captured["config"] = config

        original_omegaconf = app_module.OmegaConf
        original_run_server = app_module.run_server
        app_module.OmegaConf = FakeOmegaConf
        app_module.run_server = fake_run_server
        try:
            app_module.serve(config=Path("/tmp/config.yaml"))
        finally:
            app_module.OmegaConf = original_omegaconf
            app_module.run_server = original_run_server

        self.assertIs(captured["config"], loaded_config)
        self.assertEqual(captured["path"], Path("/tmp/config.yaml"))


if __name__ == "__main__":
    unittest.main()
