"""Real BF16 operator alignment and ASR-only module ownership checks."""

import os
import unittest
from types import SimpleNamespace


@unittest.skipUnless(os.environ.get("QWEN_ASR_TEST_NPU") == "1", "requires Ascend NPU")
class AscendRotaryTest(unittest.TestCase):
    def make_original(self):
        import torch
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

        with set_current_vllm_config(VllmConfig()):
            return MRotaryEmbedding(
                head_size=128, rotary_dim=128, max_position_embeddings=1024,
                base=1000000, is_neox_style=True, dtype=torch.bfloat16,
                mrope_section=[24, 20, 20], mrope_interleaved=True,
            )

    def test_bf16_matches_native_with_linear_and_three_axis_positions(self):
        import torch
        from qwen_asr.models.vllm_ascend.rotary_embedding import LinearPositionRotaryEmbedding

        torch.manual_seed(42)
        for count in (1, 4, 8, 405):
            original = self.make_original().npu()
            fused = LinearPositionRotaryEmbedding(original)
            positions = torch.arange(17, 17 + count, device="npu")
            q = torch.randn(count, 16 * 128, device="npu", dtype=torch.bfloat16)
            k = torch.randn(count, 8 * 128, device="npu", dtype=torch.bfloat16)
            for axes in (positions, positions.expand(3, -1)):
                expected = original.forward_native(axes, q.clone(), k.clone())
                actual = fused(axes, q.clone(), k.clone())
                for a, b in zip(actual, expected):
                    self.assertLessEqual((a.float() - b.float()).abs().max().item(), 0.0625)
                    similarity = torch.nn.functional.cosine_similarity(
                        a.float().flatten(), b.float().flatten(), dim=0,
                    ).item()
                    self.assertGreater(similarity, 0.9999)

    def test_install_does_not_change_other_models_or_global_dispatch(self):
        from qwen_asr.models.vllm_ascend.rotary_embedding import (
            LinearPositionRotaryEmbedding, install_asr_fused_rope,
        )

        original = self.make_original()
        method = type(original).forward_oot
        attention = SimpleNamespace(rotary_emb=original)
        decoder = SimpleNamespace(model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)],
        ))
        self.assertEqual(install_asr_fused_rope(decoder), 1)
        self.assertIsInstance(attention.rotary_emb, LinearPositionRotaryEmbedding)
        self.assertIs(type(original).forward_oot, method)
        self.assertNotIsInstance(self.make_original(), LinearPositionRotaryEmbedding)
        self.assertEqual(install_asr_fused_rope(decoder), 1)

    def test_rejects_partial_head_rotation(self):
        from qwen_asr.models.vllm_ascend.rotary_embedding import LinearPositionRotaryEmbedding

        original = self.make_original()
        original.rotary_dim = 64
        with self.assertRaisesRegex(ValueError, "full-head"):
            LinearPositionRotaryEmbedding(original)
