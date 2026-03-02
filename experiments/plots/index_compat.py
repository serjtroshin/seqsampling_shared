from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _candidate_output_count(row: Mapping[str, Any]) -> int:
    candidate_outputs = row.get("solutions")
    if not isinstance(candidate_outputs, list):
        candidate_outputs = row.get("generations")
    if not isinstance(candidate_outputs, list):
        candidate_outputs = row.get("raw_generations")
    if not isinstance(candidate_outputs, list):
        candidate_outputs = []
    return len(candidate_outputs)


def fallback_sequential_id(response_idx: int, scenario_meta: Mapping[str, Any]) -> int:
    """Fallback mapping from response index to sequential_id."""
    sampling_mode = str(scenario_meta.get("sampling_mode", "")).strip().lower()
    scenario_name = str(scenario_meta.get("name", "")).strip().lower()
    if sampling_mode == "parallel" or "parallel" in scenario_name:
        return 0
    return response_idx


def build_response_to_turn_map(
    rows: Sequence[Mapping[str, Any]],
    scenario_meta: Mapping[str, Any],
) -> dict[tuple[str, int], int]:
    """Build (prompt_id, response_idx) -> sequential_id from sample rows."""
    turn_by_key: dict[tuple[str, int], int] = {}
    for row in rows:
        prompt_id = str(row.get("prompt_id", "unknown"))
        sequential_ids = row.get("sequential_ids")
        if not isinstance(sequential_ids, list):
            sequential_ids = []

        row_count = max(len(sequential_ids), _candidate_output_count(row))
        for response_idx in range(row_count):
            sequential_id = (
                _safe_int(sequential_ids[response_idx])
                if response_idx < len(sequential_ids)
                else None
            )
            if sequential_id is None:
                sequential_id = fallback_sequential_id(response_idx, scenario_meta)
            turn_by_key[(prompt_id, response_idx)] = sequential_id
    return turn_by_key


def build_turn_parallel_to_response_map(
    rows: Sequence[Mapping[str, Any]],
    scenario_meta: Mapping[str, Any],
) -> dict[tuple[str, int, int], int]:
    """Build (prompt_id, parallel_idx, sequential_id) -> response_idx map."""
    response_by_key: dict[tuple[str, int, int], int] = {}
    for row in rows:
        prompt_id = str(row.get("prompt_id", "unknown"))
        sequential_ids = row.get("sequential_ids")
        if not isinstance(sequential_ids, list):
            sequential_ids = []
        parallel_ids = row.get("parallel_ids")
        if not isinstance(parallel_ids, list):
            parallel_ids = []

        row_count = max(
            len(sequential_ids),
            len(parallel_ids),
            _candidate_output_count(row),
        )
        for response_idx in range(row_count):
            sequential_id = (
                _safe_int(sequential_ids[response_idx])
                if response_idx < len(sequential_ids)
                else None
            )
            if sequential_id is None:
                sequential_id = fallback_sequential_id(response_idx, scenario_meta)
            parallel_idx = (
                _safe_int(parallel_ids[response_idx])
                if response_idx < len(parallel_ids)
                else None
            )
            if parallel_idx is None:
                continue
            response_by_key[(prompt_id, parallel_idx, sequential_id)] = response_idx
    return response_by_key


@dataclass(frozen=True)
class ResolvedScoreIndices:
    prompt_id: str
    response_idx: int
    parallel_idx: int | None
    sequential_id: int


def resolve_score_item_indices(
    score_item: Mapping[str, Any],
    turn_map: Mapping[tuple[str, int], int],
    response_map: Mapping[tuple[str, int, int], int],
    scenario_meta: Mapping[str, Any],
) -> ResolvedScoreIndices | None:
    """Resolve score indices across new and legacy evaluation report schemas."""
    prompt_id = str(score_item.get("prompt_id", "unknown"))
    response_idx = _safe_int(score_item.get("response_idx"))
    parallel_idx = _safe_int(score_item.get("parallel_idx"))
    sequential_id = _safe_int(score_item.get("sequential_idx"))

    if sequential_id is None:
        if response_idx is None:
            return None
        sequential_id = turn_map.get((prompt_id, response_idx))
        if sequential_id is None:
            sequential_id = fallback_sequential_id(response_idx, scenario_meta)

    if response_idx is None and parallel_idx is not None:
        response_idx = response_map.get((prompt_id, parallel_idx, sequential_id))
    if response_idx is None:
        response_idx = sequential_id

    return ResolvedScoreIndices(
        prompt_id=prompt_id,
        response_idx=response_idx,
        parallel_idx=parallel_idx,
        sequential_id=sequential_id,
    )
