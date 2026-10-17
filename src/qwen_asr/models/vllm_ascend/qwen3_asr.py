"""Qwen3-ASR NPU encoder with the shared vLLM processor and text decoder."""

import logging
import os

import torch
from torch import nn
import torch_npu
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from qwen_asr.models.vllm.audio import Qwen3ASRAudioEncoder
from qwen_asr.models.vllm.model import Qwen3ASRForConditionalGeneration as BaseQwen3ASR
from qwen_asr.models.vllm.processing import (
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
)
from qwen_asr.models.vllm_ascend.rotary_embedding import install_asr_fused_rope

logger = logging.getLogger(__name__)


class AscendEncoderAttention(nn.Module):
    """Packed, noncausal attention; chunk boundaries isolate audio windows."""

    def __init__(self, num_heads: int, scale: float):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.graph_attention = False

    def forward(self, query, key, value, cu_seqlens, max_seqlen=None):
        if self.graph_attention:
            # v3 retains tensor metadata/outputs and supports capture on this
            # validated stack; the legacy TND call leaves an unjoined stream.
            ends = cu_seqlens[1:]
            return torch_npu.npu_fusion_attention_v3(
                query.squeeze(0).contiguous(), key.squeeze(0).contiguous(),
                value.squeeze(0).contiguous(), head_num=self.num_heads,
                input_layout="TND", scale=self.scale, keep_prob=1.0,
                actual_seq_qlen=ends, actual_seq_kvlen=ends, sparse_mode=0,
            )[0].unsqueeze(0)
        ends = cu_seqlens[1:].to(device="cpu", dtype=torch.int64).tolist()
        output = torch_npu.npu_fusion_attention(
            query.squeeze(0).contiguous(),
            key.squeeze(0).contiguous(),
            value.squeeze(0).contiguous(),
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.scale,
            keep_prob=1.0,
            actual_seq_qlen=ends,
            actual_seq_kvlen=ends,
            sparse_mode=0,
        )[0]
        return output.unsqueeze(0)


class AscendAudioEncoder(Qwen3ASRAudioEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_graph = None
        for layer in self.layers:
            attention = layer.self_attn
            attention.attn = AscendEncoderAttention(
                attention.num_local_heads, attention.scaling
            )

    def prepare_attention_inputs(self, cu_seqlens: torch.Tensor):
        # The TND API takes host sequence lengths. Transfer once per encoder
        # invocation rather than synchronizing the NPU at every encoder layer.
        return cu_seqlens.cpu(), None

    def enable_encoder_graph(self):
        from .encoder_graph import EncoderGraphCache

        if self.encoder_graph is not None:
            return
        if len(self.layers) != 24 or self.dtype != torch.bfloat16:
            raise ValueError("Encoder graph requires the 24-layer BF16 1.7B encoder")
        self.encoder_graph = EncoderGraphCache(super()._forward_transformer)
        for layer in self.layers:
            layer.self_attn.attn.graph_attention = True
        logger.info("ASR encoder graph enabled: attention=v3 profiles=4 max_tokens=1024")

    def _forward_transformer(self, hidden_states, cu_seqlens, max_seqlen):
        if self.encoder_graph is not None:
            return self.encoder_graph(hidden_states, cu_seqlens, max_seqlen)
        return super()._forward_transformer(hidden_states, cu_seqlens, max_seqlen)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRMultiModalProcessor,
    info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRForConditionalGeneration(BaseQwen3ASR):
    audio_encoder_cls = AscendAudioEncoder

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        value = os.environ.get("QWEN_ASR_ASCEND_FUSED_ROPE", "true").lower()
        if value not in {"true", "false", "1", "0"}:
            raise ValueError("QWEN_ASR_ASCEND_FUSED_ROPE must be true/false or 1/0")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        enabled = value in {"true", "1"}
        layers = install_asr_fused_rope(self.language_model) if enabled else 0
        logger.info("ASR fused RoPE: enabled=%s decoder_layers=%d", enabled, layers)
