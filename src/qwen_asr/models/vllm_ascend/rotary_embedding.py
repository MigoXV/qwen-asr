"""Fused rotary embedding for Qwen3-ASR's identical position axes."""

import torch
from torch import nn
import torch_npu


class LinearPositionRotaryEmbedding(nn.Module):
    """Model-local replacement; Qwen3-ASR constructs equal T/H/W positions.

    Equal axes make the interleaved MRoPE frequency selection equivalent to
    ordinary one-dimensional RoPE. This module must only be installed on the
    ASR decoder, never on a vision model with independent spatial positions.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        if original.rotary_dim != original.head_size or not original.is_neox_style:
            raise ValueError("ASR fused RoPE requires full-head, NeoX-style rotation")
        self.head_size = original.head_size
        self.register_buffer("cos_sin_cache", original.cos_sin_cache, persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        if positions.ndim == 2:
            positions = positions[0]
        cache = self.cos_sin_cache
        if cache.device != query.device or cache.dtype != query.dtype:
            cache = cache.to(device=query.device, dtype=query.dtype)
            self.cos_sin_cache = cache
        return torch_npu.npu_mrope(
            positions.contiguous(), query.contiguous(), key.contiguous(),
            cache.contiguous(), self.head_size,
            mrope_section=[0, 0, 0], rotary_mode="half",
        )


def install_asr_fused_rope(language_model: nn.Module) -> int:
    """Replace this ASR decoder's modules before compilation and graph capture."""
    layers = language_model.model.layers
    count = 0
    for layer in layers:
        attention = layer.self_attn
        if not isinstance(attention.rotary_emb, LinearPositionRotaryEmbedding):
            attention.rotary_emb = LinearPositionRotaryEmbedding(attention.rotary_emb)
        count += 1
    if not count:
        raise ValueError("ASR fused RoPE found no decoder layers")
    return count
