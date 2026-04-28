from __future__ import annotations

from dataclasses import dataclass

from qwen_asr.inference.utils import parse_asr_output, resolve_language_code
from qwen_asr.servicer.utils import suppress_incomplete_protocol_prefix


@dataclass
class ParsedTranscriptUpdate:
    transcript: str
    delta: str
    changed: bool


class StreamingTranscriptParser:
    """
    Accumulate raw model chunks and expose cleaned transcript updates.

    The model may echo protocol prefixes such as ``language X<asr_text>``.
    This helper keeps that parsing logic out of the gRPC servicer and returns
    only cleaned transcript text plus the newly appended delta.
    """

    def __init__(self, language_code: str | None = None) -> None:
        self.force_language = resolve_language_code(language_code)
        self.raw_text = ""
        self.transcript = ""

    def push(self, chunk: str) -> ParsedTranscriptUpdate:
        self.raw_text += chunk
        _, parsed_text = parse_asr_output(
            self.raw_text, user_language=self.force_language
        )
        parsed_text = suppress_incomplete_protocol_prefix(
            raw_text=self.raw_text,
            parsed_text=parsed_text,
        )
        previous = self.transcript
        delta = (
            parsed_text[len(previous) :]
            if parsed_text.startswith(previous)
            else parsed_text
        )
        self.transcript = parsed_text
        return ParsedTranscriptUpdate(
            transcript=parsed_text,
            delta=delta,
            changed=parsed_text != previous,
        )
