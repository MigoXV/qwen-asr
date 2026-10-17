"""显式开启 QWEN_ASR_TEST_NPU=1 时运行真实 NPU 数值对齐。"""

import os
import unittest


@unittest.skipUnless(os.environ.get("QWEN_ASR_TEST_NPU") == "1", "requires Ascend NPU")
class AscendAttentionTest(unittest.TestCase):
    def test_packed_windows_match_independent_cpu_attention(self):
        import torch
        from qwen_asr.models.vllm_ascend.qwen3_asr import AscendEncoderAttention

        generator = torch.Generator().manual_seed(42)
        q, k, v = [torch.randn((1, 18, 4, 64), generator=generator).to(torch.bfloat16)
                   for _ in range(3)]
        boundaries = [0, 7, 18]
        reference = torch.cat([
            torch.nn.functional.scaled_dot_product_attention(
                q[:, start:end].transpose(1, 2).float(),
                k[:, start:end].transpose(1, 2).float(),
                v[:, start:end].transpose(1, 2).float(),
            ).transpose(1, 2)
            for start, end in zip(boundaries, boundaries[1:])
        ], dim=1)
        output = AscendEncoderAttention(4, 64 ** -0.5)(
            q.npu(), k.npu(), v.npu(),
            torch.tensor(boundaries, device="npu", dtype=torch.int32),
        ).cpu().float()
        self.assertLess((output - reference).abs().max().item(), 0.02)


if __name__ == "__main__":
    unittest.main()
