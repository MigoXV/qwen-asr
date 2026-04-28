from __future__ import annotations


def suppress_incomplete_protocol_prefix(raw_text: str, parsed_text: str) -> str:
    """
    Hide leaked protocol prefixes from interim gRPC responses.

    During streaming the model may emit partial metadata such as
    ``language None`` before it has generated ``<asr_text>``. Some models
    also emit malformed partial fragments such as ``lang`` / ``langute``
    while they are still working toward that prefix. At that point
    ``parse_asr_output()`` cannot distinguish the prefix from real text yet,
    so the gRPC layer suppresses it until actual transcript text appears.
    """
    text = (parsed_text or "").strip()
    raw = (raw_text or "").strip()
    if not text or not raw:
        return ""

    if "<asr_text>" in raw:
        return parsed_text

    lowered = raw.lower()
    first_token = lowered.split(None, 1)[0]
    if first_token.startswith("lang"):
        return ""
    return parsed_text
