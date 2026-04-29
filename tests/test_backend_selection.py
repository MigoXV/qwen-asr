from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


utils_path = Path(__file__).resolve().parents[1] / "src/qwen_asr/commands/utils.py"
spec = importlib.util.spec_from_file_location("_test_utils_backend", utils_path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


class BackendSelectionTest(unittest.TestCase):
    def test_default_backend_vllm(self):
        self.assertEqual(mod.normalize_backend(""), "vllm")

    def test_transformers_backend_ok(self):
        self.assertEqual(mod.normalize_backend("transformers"), "transformers")

    def test_invalid_backend(self):
        with self.assertRaisesRegex(ValueError, "Invalid BACKEND value"):
            mod.normalize_backend("foo")


if __name__ == "__main__":
    unittest.main()
