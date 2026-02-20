from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


@dataclass(frozen=True)
class SampleRecord:
    prompt_id: Optional[str]
    response_idx: int
    src: str
    mt: str
    ref: Optional[str] = None


def _extract_source(row: dict[str, Any]) -> str:
    value = row.get("source")
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("Missing non-empty 'source' field; quality evaluation requires source text under key 'source'.")


def _extract_reference(row: dict[str, Any]) -> Optional[str]:
    for key in ("ref", "reference", "target", "gold"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _find_generations(row: dict[str, Any], generation_keys: Sequence[str]) -> Optional[list[Any]]:
    for key in generation_keys:
        value = row.get(key)
        if isinstance(value, list):
            return value
    return None


def extract_records_with_ids(
    path: Path,
    generation_keys: Sequence[str] = ("generations", "solutions"),
    max_samples: Optional[int] = None,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_samples is not None and len(records) >= max_samples:
                break
            row = json.loads(line)
            prompt_id_raw = row.get("prompt_id")
            prompt_id = None if prompt_id_raw is None else str(prompt_id_raw)
            src = _extract_source(row)
            ref = _extract_reference(row)
            generations = _find_generations(row, generation_keys)
            if not generations:
                continue
            for idx, generation in enumerate(generations):
                if max_samples is not None and len(records) >= max_samples:
                    break
                records.append(
                    SampleRecord(
                        prompt_id=prompt_id,
                        response_idx=idx,
                        src=src,
                        mt=str(generation),
                        ref=ref,
                    )
                )
    return records


def has_generation_key(path: Path, key: str) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if isinstance(row.get(key), list):
                return True
    return False


def as_comet_qe_payload(records: Iterable[SampleRecord]) -> list[dict[str, str]]:
    return [{"src": record.src, "mt": record.mt} for record in records]


def as_comet_ref_payload(records: Iterable[SampleRecord]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for record in records:
        if not record.ref:
            raise ValueError("Reference-based metric requested, but one or more records do not contain references.")
        payload.append({"src": record.src, "mt": record.mt, "ref": record.ref})
    return payload
