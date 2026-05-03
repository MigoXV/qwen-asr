from __future__ import annotations

from typing import Tuple

from qwen_asr.inferencers.language import resolve_language_code
from qwen_asr.inferencers.text.asr_output import parse_asr_output


TranscriptUpdate = Tuple[str, str, bool]


def suppress_incomplete_protocol_prefix(raw_text: str, parsed_text: str) -> str:
    """
    在 gRPC 中间结果中隐藏泄漏的协议前缀。

    流式生成时，模型可能在生成 ``<asr_text>`` 前先输出
    ``language None`` 这类元信息。部分模型还会输出 ``lang`` /
    ``langute`` 等未完成片段。此时 ``parse_asr_output()`` 还无法区分
    协议前缀和真实文本，因此 gRPC 推理层先抑制这些内容，直到真实转写出现。
    """
    text = (parsed_text or "").strip()
    raw = (raw_text or "").strip()
    if not text or not raw:
        return ""

    if "<asr_text>" in raw:
        return parsed_text

    # 未出现正式文本标签前，形似 language 前缀的内容都不下发。
    lowered = raw.lower()
    first_token = lowered.split(None, 1)[0]
    if first_token.startswith("lang"):
        return ""
    return parsed_text


class StreamingTranscriptParser:
    """
    累积模型原始分片，并输出清理后的转写更新。

    模型可能回显 ``language X<asr_text>`` 这类协议前缀。
    该工具只返回清理后的文本以及本次新增的片段。
    """

    def __init__(self, language_code: str | None = None) -> None:
        self.force_language = resolve_language_code(language_code)
        self.raw_text = ""
        self.transcript = ""

    def push(self, chunk: str) -> TranscriptUpdate:
        # 基于完整历史文本解析，避免分片边界导致协议前缀泄漏。
        self.raw_text += chunk
        _, parsed_text = parse_asr_output(
            self.raw_text, user_language=self.force_language
        )
        parsed_text = suppress_incomplete_protocol_prefix(
            raw_text=self.raw_text,
            parsed_text=parsed_text,
        )
        previous = self.transcript
        # 优先返回新增文本，无法按前缀匹配时退回完整文本。
        delta = (
            parsed_text[len(previous) :]
            if parsed_text.startswith(previous)
            else parsed_text
        )
        self.transcript = parsed_text
        return parsed_text, delta, parsed_text != previous
