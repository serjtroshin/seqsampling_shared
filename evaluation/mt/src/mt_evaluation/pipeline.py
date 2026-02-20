from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .metrics.base import EvaluationMetric
from .records import SampleRecord, extract_records_with_ids, has_generation_key


def _build_report(
    metric: EvaluationMetric,
    input_file: Path,
    records: Sequence[SampleRecord],
    scores: Sequence[float],
) -> dict[str, Any]:
    if len(scores) != len(records):
        raise ValueError(
            f"Metric {metric.config.name!r} returned {len(scores)} scores for {len(records)} records."
        )
    average = float(sum(scores) / len(scores)) if scores else float("nan")
    scored = [
        {
            "prompt_id": record.prompt_id,
            "response_idx": record.response_idx,
            "score": float(score),
        }
        for record, score in zip(records, scores)
    ]
    report = {
        "file": str(input_file),
        "n": len(scores),
        "average_score": average,
        "scores": scored,
    }
    report.update(metric.report_metadata())
    metric.augment_report(report, records)
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def evaluate_file(
    input_file: Path,
    metrics: Sequence[EvaluationMetric],
    output_dir: Path | None = None,
    generation_keys: Sequence[str] = ("generations", "solutions"),
    include_final_answers: bool = True,
    max_samples: int | None = None,
) -> dict[str, Path]:
    if not metrics:
        raise ValueError("At least one metric is required.")

    output_root = output_dir or (input_file.parent / "evaluation")
    results: dict[str, Path] = {}

    records = extract_records_with_ids(
        input_file,
        generation_keys=generation_keys,
        max_samples=max_samples,
    )
    if not records:
        raise ValueError(
            "No records found with the requested generation keys. "
            f"Tried keys: {tuple(generation_keys)!r}"
        )

    fa_records: list[SampleRecord] = []
    if include_final_answers and has_generation_key(input_file, "final_answers"):
        fa_records = extract_records_with_ids(
            input_file,
            generation_keys=("final_answers",),
            max_samples=max_samples,
        )

    for metric in metrics:
        try:
            scores = metric.score(records)
            report = _build_report(metric, input_file, records, scores)
            out_path = output_root / f"{metric.config.name}.json"
            _write_report(out_path, report)
            results[metric.config.name] = out_path

            if fa_records:
                fa_scores = metric.score(fa_records)
                fa_report = _build_report(metric, input_file, fa_records, fa_scores)
                fa_path = output_root / f"{metric.config.name}.FA.json"
                _write_report(fa_path, fa_report)
                results[f"{metric.config.name}.FA"] = fa_path
        finally:
            metric.cleanup()

    return results
