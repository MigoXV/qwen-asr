from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))

from qwen_asr.inferencers.transformers import TransformersInferencer  # noqa: E402


class FakeTransformersModel:
    device = torch.device("cpu")
    dtype = torch.float16


class TransformersInputPreparationTest(unittest.TestCase):
    def test_float_inputs_are_cast_to_model_dtype(self):
        inferencer = TransformersInferencer(FakeTransformersModel(), processor=None)

        inputs = inferencer._prepare_model_inputs(
            {
                "input_features": torch.ones((1, 80, 10), dtype=torch.float32),
                "input_ids": torch.ones((1, 4), dtype=torch.long),
                "feature_attention_mask": torch.ones((1, 10), dtype=torch.long),
                "metadata": "kept",
            }
        )

        self.assertEqual(inputs["input_features"].dtype, torch.float16)
        self.assertEqual(inputs["input_ids"].dtype, torch.long)
        self.assertEqual(inputs["feature_attention_mask"].dtype, torch.long)
        self.assertEqual(inputs["metadata"], "kept")


if __name__ == "__main__":
    unittest.main()
