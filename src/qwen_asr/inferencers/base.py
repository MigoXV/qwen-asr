from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Optional, Protocol

import numpy as np


class ASRInferencer(Protocol):
    """支持流式 ASR 转写的推理器接口。"""

    def transcribe_stream(
        self,
        audio: np.ndarray,
        sample_rate: int,
        context: str = "",
        language: Optional[str] = None,
    ) -> AsyncIterator[str]:
        ...
