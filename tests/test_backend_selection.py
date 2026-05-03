from __future__ import annotations

from pathlib import Path
import sys
import unittest


src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))

from qwen_asr.configs import AppConfig  # noqa: E402


class BackendSelectionTest(unittest.TestCase):
    def test_default_backend_vllm(self):
        self.assertEqual(AppConfig(model="/tmp/model").backend, "vllm")

    def test_empty_backend_fails(self):
        with self.assertRaisesRegex(ValueError, "Invalid BACKEND value"):
            AppConfig.normalize_backend("")

    def test_transformers_backend_ok(self):
        self.assertEqual(AppConfig.normalize_backend("transformers"), "transformers")

    def test_invalid_backend(self):
        with self.assertRaisesRegex(ValueError, "Invalid BACKEND value"):
            AppConfig.normalize_backend("foo")


if __name__ == "__main__":
    unittest.main()
