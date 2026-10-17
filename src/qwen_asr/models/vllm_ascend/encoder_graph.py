"""有界、精确长度的音频编码图；缓存图执行资源，不缓存音频结果。"""

from collections import OrderedDict
from dataclasses import dataclass
import logging
import threading

import torch

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    graph: object
    inputs: torch.Tensor
    output: torch.Tensor
    lengths: torch.Tensor  # Keep the captured host metadata alive.


class EncoderGraphCache:
    """Single Worker/stream owner, four exact profiles, bounded admission history."""

    max_entries = 4
    max_history = 64
    max_tokens = 1024

    def __init__(self, forward):
        self.forward = forward
        self.entries = OrderedDict()
        self.seen = OrderedDict()
        self.owner = threading.get_ident()
        self.stats = dict(captures=0, replays=0, eager=0, evictions=0, profiles=0)

    def __call__(self, hidden, lengths, maximum):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Encoder graph must run on its owning Worker thread")
        lengths = lengths.to(dtype=torch.int64)
        if (hidden.device.type != 'npu' or hidden.dtype != torch.bfloat16
                or hidden.ndim != 2 or hidden.shape[1] != 1024
                or not 0 < hidden.shape[0] <= self.max_tokens
                or torch.is_grad_enabled() or maximum is not None):
            self.stats['eager'] += 1
            return self.forward(hidden, lengths, maximum)
        if lengths.device.type != 'cpu' or lengths.ndim != 1:
            raise ValueError("Encoder graph requires host cumulative lengths")
        boundaries = tuple(lengths.tolist())
        if (len(boundaries) < 2 or boundaries[0] != 0
                or boundaries[-1] != hidden.shape[0]
                or any(a >= b for a, b in zip(boundaries, boundaries[1:]))):
            raise ValueError("Encoder graph boundaries must partition every input token")
        key = (tuple(hidden.shape), hidden.dtype, hidden.device, boundaries)
        if key not in self.seen:
            self.seen[key] = None
            if len(self.seen) > self.max_history:
                self.seen.popitem(last=False)
            self.stats['eager'] += 1
            return self.forward(hidden, lengths, maximum)
        self.seen.move_to_end(key)
        if key not in self.entries:
            if len(self.entries) == self.max_entries:
                torch.npu.synchronize()
                _, entry = self.entries.popitem(last=False)
                entry.graph.reset()
                del entry
                self.stats['evictions'] += 1
            inputs = hidden.contiguous().clone()
            captured_lengths = lengths.clone()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = self.forward(inputs, captured_lengths, maximum)
            self.entries[key] = _Entry(graph, inputs, output, captured_lengths)
            self.stats['captures'] += 1
            self.stats['profiles'] = len(self.entries)
            logger.info("ASR encoder graph captured: tokens=%d windows=%d profiles=%d/%d",
                        hidden.shape[0], len(boundaries)-1, len(self.entries), self.max_entries)
        entry = self.entries[key]
        self.entries.move_to_end(key)
        entry.inputs.copy_(hidden)
        entry.graph.replay()
        self.stats['replays'] += 1
        # The upstream multimodal cache retains these outputs across requests.
        # Never expose storage overwritten by a later graph replay.
        return entry.output.clone()
