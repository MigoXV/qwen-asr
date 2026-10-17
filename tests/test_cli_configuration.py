from __future__ import annotations

from unittest.mock import patch
import unittest

from typer.testing import CliRunner

from tests.grpc_stub import install_fake_grpc

install_fake_grpc()

from qwen_asr.commands import app as app_module
from qwen_asr.commands import serving as serving_module


runner = CliRunner()


class CLIConfigurationTest(unittest.TestCase):
    def test_vllm_defaults_fit_twelve_gib_a100_budget(self):
        captured = {}

        def fake_create_vllm_inferencer(**kwargs):
            captured.update(kwargs)
            return object()

        async def fake_run_server(inferencer, *, context, server_port):
            return None

        with (
            patch.object(app_module, "configure_logging"),
            patch.object(
                serving_module,
                "create_vllm_inferencer",
                side_effect=fake_create_vllm_inferencer,
            ),
            patch.object(serving_module, "run_server", side_effect=fake_run_server),
        ):
            result = runner.invoke(
                app_module.app,
                ["vllm", "--model", "/models/qwen3-asr", "--device", "cuda"],
                env={},
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(captured["max_new_tokens"], 256)
        self.assertEqual(captured["gpu_memory_utilization"], 0.27)
        self.assertEqual(captured["max_model_len"], 1536)
        self.assertEqual(captured["max_num_seqs"], 4)
        self.assertEqual(captured["max_num_batched_tokens"], 1536)

    def test_vllm_reads_all_options_from_environment(self):
        captured = {}
        fake_inferencer = object()

        def fake_create_vllm_inferencer(**kwargs):
            captured["backend"] = kwargs
            return fake_inferencer

        async def fake_run_server(inferencer, *, context, server_port):
            captured["server"] = {
                "inferencer": inferencer,
                "context": context,
                "server_port": server_port,
            }

        env = {
            "QWEN_ASR_MODEL": "/models/qwen3-asr",
            "QWEN_ASR_CONTEXT": "医疗词汇",
            "QWEN_ASR_SERVER_PORT": "50052",
            "QWEN_ASR_MAX_NEW_TOKENS": "512",
            "QWEN_ASR_DEVICE": "cuda",
            "QWEN_ASR_VLLM_GPU_MEMORY_UTILIZATION": "0.5",
            "QWEN_ASR_VLLM_MAX_MODEL_LEN": "1024",
            "QWEN_ASR_VLLM_MAX_NUM_SEQS": "8",
            "QWEN_ASR_VLLM_MAX_NUM_BATCHED_TOKENS": "2048",
            "QWEN_ASR_VLLM_ENFORCE_EAGER": "true",
        }

        with (
            patch.object(app_module, "configure_logging"),
            patch.object(
                serving_module,
                "create_vllm_inferencer",
                side_effect=fake_create_vllm_inferencer,
            ),
            patch.object(serving_module, "run_server", side_effect=fake_run_server),
        ):
            result = runner.invoke(app_module.app, ["vllm"], env=env)

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            captured["backend"],
            {
                "model": "/models/qwen3-asr",
                "max_new_tokens": 512,
                "device": app_module.Device.CUDA,
                "gpu_memory_utilization": 0.5,
                "max_model_len": 1024,
                "max_num_seqs": 8,
                "max_num_batched_tokens": 2048,
                "enforce_eager": True,
            },
        )
        self.assertEqual(
            captured["server"],
            {
                "inferencer": fake_inferencer,
                "context": "医疗词汇",
                "server_port": 50052,
            },
        )

    def test_transformers_reads_options_from_environment(self):
        captured = {}
        fake_inferencer = object()

        def fake_create_transformers_inferencer(**kwargs):
            captured["backend"] = kwargs
            return fake_inferencer

        async def fake_run_server(inferencer, *, context, server_port):
            captured["server"] = (inferencer, context, server_port)

        env = {
            "QWEN_ASR_MODEL": "/models/qwen3-asr",
            "QWEN_ASR_CONTEXT": "会议记录",
            "QWEN_ASR_SERVER_PORT": "50053",
            "QWEN_ASR_MAX_NEW_TOKENS": "256",
            "QWEN_ASR_DEVICE": "cpu",
        }

        with (
            patch.object(app_module, "configure_logging"),
            patch.object(
                serving_module,
                "create_transformers_inferencer",
                side_effect=fake_create_transformers_inferencer,
            ),
            patch.object(serving_module, "run_server", side_effect=fake_run_server),
        ):
            result = runner.invoke(app_module.app, ["transformers"], env=env)

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            captured["backend"],
            {
                "model": "/models/qwen3-asr",
                "max_new_tokens": 256,
                "device": app_module.Device.CPU,
            },
        )
        self.assertEqual(
            captured["server"],
            (fake_inferencer, "会议记录", 50053),
        )

    def test_command_line_overrides_environment(self):
        captured = {}

        def fake_create_transformers_inferencer(**kwargs):
            captured.update(kwargs)
            return object()

        async def fake_run_server(inferencer, *, context, server_port):
            return None

        with (
            patch.object(app_module, "configure_logging"),
            patch.object(
                serving_module,
                "create_transformers_inferencer",
                side_effect=fake_create_transformers_inferencer,
            ),
            patch.object(serving_module, "run_server", side_effect=fake_run_server),
        ):
            result = runner.invoke(
                app_module.app,
                [
                    "transformers",
                    "--model",
                    "/models/from-cli",
                    "--device",
                    "cpu",
                ],
                env={
                    "QWEN_ASR_MODEL": "/models/from-env",
                    "QWEN_ASR_DEVICE": "cuda",
                },
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(captured["model"], "/models/from-cli")
        self.assertEqual(captured["device"], app_module.Device.CPU)

    def test_model_is_required(self):
        result = runner.invoke(app_module.app, ["transformers"], env={})

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--model", result.output)


class BackendConstructionTest(unittest.TestCase):
    def test_vllm_only_passes_enforce_eager_when_enabled(self):
        fake_backend = unittest.mock.Mock()
        fake_backend.LLM.return_value = object()

        with patch.dict(
            "sys.modules",
            {"qwen_asr.inferencers.vllm": unittest.mock.Mock(VLLMInferencer=fake_backend)},
        ):
            serving_module.create_vllm_inferencer(
                model="/models/qwen3-asr",
                max_new_tokens=512,
                device=app_module.Device.CPU,
                gpu_memory_utilization=0.5,
                max_model_len=1024,
                max_num_seqs=4,
                max_num_batched_tokens=1536,
                enforce_eager=False,
            )
            first_kwargs = fake_backend.LLM.call_args.kwargs

            serving_module.create_vllm_inferencer(
                model="/models/qwen3-asr",
                max_new_tokens=512,
                device=app_module.Device.CUDA,
                gpu_memory_utilization=0.5,
                max_model_len=1024,
                max_num_seqs=4,
                max_num_batched_tokens=1536,
                enforce_eager=True,
            )
            second_kwargs = fake_backend.LLM.call_args.kwargs

        self.assertNotIn("enforce_eager", first_kwargs)
        self.assertTrue(second_kwargs["enforce_eager"])
        self.assertNotIn("device", first_kwargs)
        self.assertNotIn("device", second_kwargs)


class AscendConfigurationTest(unittest.TestCase):
    def test_native_fia_update_option_and_conflicting_decode_path(self):
        with patch.object(serving_module, "start_vllm_ascend") as start:
            result = runner.invoke(app_module.app, ["vllm-ascend", "--model", "/models/asr"],
                                   env={"QWEN_ASR_ASCEND_NATIVE_FIA_UPDATE": "true"})
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(start.call_args.kwargs["native_fia_update"])
        for options in (["--triton-decode"], ["--enforce-eager"], ["--workers", "9"],
                        ["--max-model-len", "2048"], ["--graph-mode", "PIECEWISE"]):
            with self.subTest(options=options), patch.object(serving_module, "start_vllm_ascend") as start:
                result = runner.invoke(app_module.app,
                    ["vllm-ascend", "--model", "/models/asr", "--native-fia-update", *options])
                self.assertNotEqual(result.exit_code, 0)
                start.assert_not_called()

    def test_triton_decode_environment_and_incompatible_graphs(self):
        with patch.object(serving_module,'start_vllm_ascend') as start:
            result=runner.invoke(app_module.app,['vllm-ascend','--model','/models/asr','--workers','8'],
                                 env={'QWEN_ASR_ASCEND_TRITON_DECODE':'true'})
        self.assertEqual(result.exit_code,0,result.output)
        self.assertTrue(start.call_args.kwargs['triton_decode'])
        for options in (['--workers','9'],['--enforce-eager'],['--max-model-len','2048'],['--graph-mode','PIECEWISE']):
            with self.subTest(options=options), patch.object(serving_module,'start_vllm_ascend') as start:
                result=runner.invoke(app_module.app,['vllm-ascend','--model','/models/asr','--triton-decode',*options])
            self.assertNotEqual(result.exit_code,0)
            start.assert_not_called()

    def test_engine_concurrency_shares_one_model_and_sized_kv_cache(self):
        from qwen_asr.engines.vllm import process as pool_module

        for workers in (4, 8):
            captured = {}

            class FakePool:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

                async def start(self):
                    captured["started"] = True

                async def aclose(self):
                    captured["closed"] = True

            async def fake_server(inferencer, **kwargs):
                captured["server"] = kwargs

            with (
                patch.object(pool_module, "VLLMEngineClient", FakePool),
                patch("qwen_asr.engines.vllm.ascend.config.kv_cache_bytes", return_value=112 * workers * 1024 ** 2),
                patch.object(app_module, "configure_logging"),
                patch.object(serving_module, "run_server", side_effect=fake_server),
            ):
                result = runner.invoke(app_module.app, ["vllm-ascend", "--model", "/models/asr", "--workers", str(workers)])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertNotIn("workers", captured)
            self.assertEqual(captured["engine_config"]["max_num_seqs"], workers)
            self.assertEqual(captured["max_inflight"], 32)
            self.assertEqual(captured["engine_config"]["gpu_memory_utilization"], 0.10)
            self.assertEqual(captured["engine_config"]["kv_cache_memory_bytes"], 112 * workers * 1024 ** 2)
            self.assertEqual(captured["engine_config"]["dtype"], "bfloat16")
            self.assertFalse(captured["engine_config"]["enforce_eager"])
            self.assertTrue(captured["engine_config"]["async_scheduling"])
            self.assertEqual(captured["engine_config"]["max_num_batched_tokens"], 4096)
            self.assertEqual(captured["engine_config"]["compilation_config"], {
                "mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": list(range(1, workers + 1)),
            })
            self.assertFalse(captured["engine_config"]["enable_prefix_caching"])
            self.assertEqual(captured["server"]["server_port"], 50011)
            self.assertEqual(captured["server"]["max_concurrent_rpcs"], 32)
            self.assertTrue(captured["started"] and captured["closed"])

    def test_workers_cannot_exceed_rpc_capacity(self):
        with patch.object(app_module, "configure_logging"):
            result = runner.invoke(app_module.app, ["vllm-ascend", "--model", "/models/asr", "--workers", "8", "--max-concurrent-rpcs", "4"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("workers", result.output)

    def test_encoder_graph_requires_validated_triton_profile(self):
        with patch.object(serving_module,'start_vllm_ascend') as start:
            result = runner.invoke(app_module.app,
                ['vllm-ascend','--model','/models/asr','--workers','8',
                 '--triton-decode','--max-num-batched-tokens','1024'],
                env={'QWEN_ASR_ASCEND_ENCODER_GRAPH':'true'})
        self.assertEqual(result.exit_code,0,result.output)
        self.assertTrue(start.call_args.kwargs['encoder_graph'])
        for options in ([],['--triton-decode'],['--max-num-batched-tokens','1024']):
            with patch.object(serving_module,'start_vllm_ascend') as start:
                result = runner.invoke(app_module.app,
                    ['vllm-ascend','--model','/models/asr','--encoder-graph',*options])
                self.assertNotEqual(result.exit_code,0)
                start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
