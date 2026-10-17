import unittest

import numpy as np
import torch

from qwen_asr.models.common.audio_lengths import get_audio_output_lengths
from qwen_asr.models.common.position_embedding import SinusoidsPositionEmbedding


class ModelCommonTest(unittest.TestCase):
    def test_lengths_at_convolution_and_window_boundaries(self):
        lengths = [0, 1, 7, 8, 9, 99, 100, 101, 199, 200, 201]
        expected = [0, 1, 1, 1, 2, 13, 13, 14, 26, 26, 27]
        self.assertEqual([get_audio_output_lengths(n) for n in lengths], expected)
        np.testing.assert_array_equal(get_audio_output_lengths(np.array(lengths)), expected)
        torch.testing.assert_close(
            get_audio_output_lengths(torch.tensor(lengths)), torch.tensor(expected),
        )

    def test_position_embedding_values_and_nonpersistent_buffer(self):
        embedding = SinusoidsPositionEmbedding(8, 4)
        positions = torch.arange(8).float()[:, None]
        scales = torch.exp(-np.log(10000) * torch.arange(2).float())
        expected = torch.cat([(positions * scales).sin(), (positions * scales).cos()], dim=1)
        torch.testing.assert_close(embedding(5), expected[:5], rtol=0, atol=0)
        self.assertEqual(embedding.state_dict(), {})
        self.assertEqual(list(dict(embedding.named_buffers())), ["positional_embedding"])
        with self.assertRaises(ValueError):
            SinusoidsPositionEmbedding(8, 3)
