from __future__ import annotations

"""Canonical dataset row schema used across prep, training, and evaluation."""

from typing import Any


def make_record(
    *,
    record_id: str,
    src_lang: str,
    tgt_lang: str,
    src_text: str,
    tgt_text: str,
    split: str,
    source: str,
    iteration: int,
    strategy: str,
    parent_id: str | None = None,
) -> dict[str, Any]:
    # Keep a single normalized row structure for all sources (flores/opus/synthetic).
    return {
        "id": record_id,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src_text": src_text,
        "tgt_text": tgt_text,
        "split": split,
        "provenance": {
            "source": source,
            "iter": iteration,
            "strategy": strategy,
            "parent_id": parent_id,
        },
    }
