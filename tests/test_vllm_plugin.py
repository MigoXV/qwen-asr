from __future__ import annotations

import unittest


class VLLMPluginTest(unittest.TestCase):
    def test_registration_is_repeatable(self):
        from vllm import ModelRegistry

        from qwen_asr.inferencers.vllm.plugin import register_qwen_asr

        register_qwen_asr()
        register_qwen_asr()

        self.assertIn(
            "Qwen3ASRForConditionalGeneration",
            ModelRegistry.get_supported_archs(),
        )


if __name__ == "__main__":
    unittest.main()
