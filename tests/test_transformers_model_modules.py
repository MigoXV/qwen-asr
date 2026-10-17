import subprocess
import sys
import tempfile
import unittest

import torch

from qwen_asr.models.transformers.configuration_qwen3_asr import Qwen3ASRConfig
from qwen_asr.models.transformers.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen_asr.models.transformers.modeling.layers import (
    Qwen3ASRTextAttention,
    Qwen3ASRThinkerTextAttention,
    Qwen3ASRTextMLP,
    Qwen3ASRThinkerTextMLP,
    Qwen3ASRTextRMSNorm,
    Qwen3ASRThinkerTextRMSNorm,
)


def tiny_config():
    config = Qwen3ASRConfig(thinker_config={
        "audio_config": dict(
            num_mel_bins=16, encoder_layers=1, encoder_attention_heads=2,
            encoder_ffn_dim=32, d_model=16, output_dim=16, downsample_hidden_size=4,
            max_source_positions=64, n_window=4, n_window_infer=8, conv_chunksize=8,
        ),
        "text_config": dict(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, max_position_embeddings=64, pad_token_id=0,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 0]},
        ),
        "audio_token_id": 30, "audio_start_token_id": 31,
    })
    config._attn_implementation = "eager"
    return config


class TransformersModelModulesTest(unittest.TestCase):
    def test_config_import_does_not_load_network_or_accelerators(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; from qwen_asr.models.transformers.configuration_qwen3_asr "
            "import Qwen3ASRConfig; "
            "assert not any(n.startswith('qwen_asr.models.transformers.modeling') "
            "for n in sys.modules); assert 'vllm' not in sys.modules; "
            "assert 'torch_npu' not in sys.modules",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_vllm_config_and_processor_exports_do_not_import_vllm(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; from qwen_asr.models.vllm import Qwen3ASRConfig, Qwen3ASRProcessor; "
            "assert 'vllm' not in sys.modules; assert 'torch_npu' not in sys.modules; "
            "assert 'qwen_asr.models.transformers.modeling.model' not in sys.modules",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checkpoint_audio_logits_and_cached_decode_roundtrip(self):
        torch.manual_seed(1729)
        config = tiny_config()
        self.assertEqual(Qwen3ASRConfig.from_dict(config.to_dict()).to_dict(), config.to_dict())
        model = Qwen3ASRForConditionalGeneration(config).eval()
        audio = torch.randn(16, 23)
        tokens = torch.tensor([[1, 2, 3, 4]])

        def outputs(instance):
            with torch.no_grad():
                features = instance.thinker.audio_tower(
                    audio, feature_lens=torch.tensor([23]),
                ).last_hidden_state
                text = instance.thinker.model(input_ids=tokens).last_hidden_state
                return features, instance.thinker.lm_head(text)

        reference = outputs(model)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = Qwen3ASRForConditionalGeneration.from_pretrained(
                directory, attn_implementation="eager",
            ).eval()
        self.assertEqual(model.state_dict().keys(), restored.state_dict().keys())
        for before, after in zip(reference, outputs(restored)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        with torch.no_grad():
            prefix = model.thinker.model(input_ids=tokens[:, :3], use_cache=True)
            cached = model.thinker.model(
                input_ids=tokens[:, 3:], past_key_values=prefix.past_key_values, use_cache=True,
            )
        torch.testing.assert_close(
            model.thinker.lm_head(cached.last_hidden_state), reference[1][:, -1:],
            rtol=1e-5, atol=1e-6,
        )
        self.assertNotIn("thinker.audio_tower.positional_embedding.positional_embedding",
                         model.state_dict())

    def test_named_layer_variants_preserve_equivalent_math(self):
        config = tiny_config().thinker_config.text_config
        config._attn_implementation = "eager"
        hidden = torch.randn(1, 3, 16)
        for first, second in (
            (Qwen3ASRTextRMSNorm(16), Qwen3ASRThinkerTextRMSNorm(16)),
            (Qwen3ASRTextMLP(config), Qwen3ASRThinkerTextMLP(config)),
        ):
            second.load_state_dict(first.state_dict(), strict=True)
            torch.testing.assert_close(first(hidden), second(hidden), rtol=0, atol=0)
        first = Qwen3ASRTextAttention(config, 0).eval()
        second = Qwen3ASRThinkerTextAttention(config, 0).eval()
        second.load_state_dict(first.state_dict(), strict=True)
        positions = (torch.ones(1, 3, 4), torch.zeros(1, 3, 4))
        for before, after in zip(first(hidden, positions, None), second(hidden, positions, None)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(Qwen3ASRThinkerTextMLP(config, intermediate_size=24).gate_proj.out_features, 24)
