from __future__ import annotations

import json
from typing import Any


PROMPT_PAYLOAD_PREFIX = "__PARSEQ_PROMPT_PAYLOAD__:"


def encode_prompt_payload(payload: dict[str, Any]) -> str:
    """Serialize prompt-rendering metadata into a sampler input string."""
    return PROMPT_PAYLOAD_PREFIX + json.dumps(payload, ensure_ascii=False)


def decode_prompt_payload(input_text: str) -> dict[str, Any] | None:
    """Deserialize prompt-rendering metadata if the sentinel prefix is present."""
    if not input_text.startswith(PROMPT_PAYLOAD_PREFIX):
        return None
    raw = input_text[len(PROMPT_PAYLOAD_PREFIX):]
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Prompt payload must decode to a JSON object.")
    return value
