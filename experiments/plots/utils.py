from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class SweepMetricResult:
    metric: str
    variant: str
    average_score: float | None
    n: int | None
    report_file: Path


@dataclass
class SweepRunResult:
    run_index: int
    run_name: str
    run_dir: Path
    task_id: str | None = None
    status: str = "UNKNOWN"
    src: str | None = None
    tgt: str | None = None
    lp: str | None = None
    scenario_name: str | None = None
    model: str | None = None
    metrics: list[SweepMetricResult] = field(default_factory=list)


@dataclass
class SweepResults:
    sweep_dir: Path
    runs: list[SweepRunResult]


def _safe_float(value: Any) -> float | None:
    """Safely coerce value to float.

    Args:
        value: Arbitrary value to convert.
    """
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Safely coerce value to int.

    Args:
        value: Arbitrary value to convert.
    """
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    """Load YAML file and return dict payload (or empty dict).

    Args:
        path: YAML file path to load.
    """
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    return {}


def _load_json_dict(path: Path) -> dict[str, Any]:
    """Load JSON file and return dict payload.

    Args:
        path: JSON file path to load.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    return {}


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records as a list of dictionaries.

    Args:
        path: JSONL file path to parse.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if not item:
                continue
            payload = json.loads(item)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _metric_variant_from_report(path: Path) -> tuple[str, str]:
    """Infer metric and variant from report filename.

    Args:
        path: Metric report path, for example comet_qe.json or comet_qe.FA.json.
    """
    metric = path.stem
    variant = "default"
    if metric.endswith(".FA"):
        metric = metric[: -len(".FA")]
        variant = "FA"
    return metric, variant


def _load_run_dirs(index_path: Path) -> list[Path]:
    """Load run directory list from experiment_run_dirs.txt.

    Args:
        index_path: Path to experiment_run_dirs.txt.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"Sweep run index not found: {index_path}")
    run_dirs: list[Path] = []
    for raw in index_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        run_dirs.append(Path(line).resolve())
    return run_dirs


def _parse_sweep_log(path: Path) -> dict[str, dict[str, str]]:
    """Parse sweep.log and map run_name to task_id/status metadata.

    Args:
        path: sweep.log path.
    """
    if not path.exists():
        return {}

    submit_re = re.compile(r"\[submit\]\s+(t\d+)\s+run=([^\s]+)")
    done_re = re.compile(r"\[done\]\s+(t\d+)\s+run=([^\s]+)")
    failed_re = re.compile(r"\[failed\]\s+(t\d+)\s+run=([^\s]+)")

    by_run: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        submit_match = submit_re.search(line)
        if submit_match:
            task_id, run_name = submit_match.group(1), submit_match.group(2)
            by_run.setdefault(run_name, {})
            by_run[run_name]["task_id"] = task_id
            by_run[run_name].setdefault("status", "SUBMITTED")
            continue

        done_match = done_re.search(line)
        if done_match:
            task_id, run_name = done_match.group(1), done_match.group(2)
            by_run.setdefault(run_name, {})
            by_run[run_name]["task_id"] = task_id
            by_run[run_name]["status"] = "DONE"
            continue

        failed_match = failed_re.search(line)
        if failed_match:
            task_id, run_name = failed_match.group(1), failed_match.group(2)
            by_run.setdefault(run_name, {})
            by_run[run_name]["task_id"] = task_id
            by_run[run_name]["status"] = "FAILED"
            continue
    return by_run


def _scenario_meta_for_run(run_dir: Path) -> dict[str, Any]:
    """Load resolved scenario metadata for one run when present.

    Args:
        run_dir: Run directory containing generation artifacts.
    """
    generation_dir = run_dir / "generation"
    if not generation_dir.exists():
        return {}
    scenario_paths = sorted(generation_dir.rglob("scenario_resolved.yaml"))
    if not scenario_paths:
        return {}
    return _load_yaml_dict(scenario_paths[0])


def _infer_lp(run_dir: Path, scenario_meta: dict[str, Any]) -> str | None:
    """Infer language-pair string (for example en-de).

    Args:
        run_dir: Run directory used for fallback path-based inference.
        scenario_meta: Parsed scenario_resolved.yaml dictionary.
    """
    src = scenario_meta.get("src")
    tgt = scenario_meta.get("tgt")
    if isinstance(src, str) and isinstance(tgt, str) and src and tgt:
        return f"{src}-{tgt}"
    if len(run_dir.parents) >= 3:
        candidate = run_dir.parents[2].name
        if "-" in candidate:
            return candidate
    return None


def _infer_run_status(log_status: str | None, metrics: list[SweepMetricResult]) -> str:
    """Infer run status from log status fallback and metric presence.

    Args:
        log_status: Status parsed from sweep.log if available.
        metrics: Metric summaries discovered in evaluation reports.
    """
    if log_status:
        return log_status
    if metrics:
        return "DONE"
    return "UNKNOWN"


def _load_metric_reports(run_dir: Path) -> list[SweepMetricResult]:
    """Load metric report summaries from a run's evaluation directory.

    Args:
        run_dir: Run directory that may contain evaluation/*.json reports.
    """
    evaluation_dir = run_dir / "evaluation"
    if not evaluation_dir.is_dir():
        return []

    metric_rows: list[SweepMetricResult] = []
    for report_file in sorted(evaluation_dir.glob("*.json")):
        try:
            report = _load_json_dict(report_file)
        except json.JSONDecodeError:
            continue

        metric, variant = _metric_variant_from_report(report_file)
        metric_rows.append(
            SweepMetricResult(
                metric=metric,
                variant=variant,
                average_score=_safe_float(report.get("average_score")),
                n=_safe_int(report.get("n")),
                report_file=report_file,
            )
        )
    return metric_rows


def _fallback_sequential_id(response_idx: int, scenario_meta: dict[str, Any]) -> int:
    """Fallback mapping from response index to sequential_id.

    Args:
        response_idx: Zero-based response index in the sample list.
        scenario_meta: Scenario metadata dict used to detect parallel mode.
    """
    sampling_mode = str(scenario_meta.get("sampling_mode", "")).strip().lower()
    scenario_name = str(scenario_meta.get("name", "")).strip().lower()
    if sampling_mode == "parallel" or "parallel" in scenario_name:
        return 0
    return response_idx


def _resolve_sample_file(report: dict[str, Any], run_dir: Path) -> Path | None:
    """Resolve source sample JSONL file for a metric report.

    Args:
        report: Parsed metric report dictionary.
        run_dir: Run directory for relative-path and fallback lookup.
    """
    raw_path = report.get("file")
    if isinstance(raw_path, str) and raw_path.strip():
        candidate = Path(raw_path.strip())
        if not candidate.is_absolute():
            candidate = (run_dir / candidate).resolve()
        if candidate.exists():
            return candidate

    generation_root = run_dir / "generation"
    if not generation_root.is_dir():
        return None
    candidates = sorted(generation_root.rglob("samples*.jsonl"))
    return candidates[0] if candidates else None


def _response_to_turn_map(
    sample_file: Path,
    scenario_meta: dict[str, Any],
) -> dict[tuple[str, int], int]:
    """Build (prompt_id, response_idx) -> sequential_id map from sample records.

    Args:
        sample_file: JSONL file with generation/sample records.
        scenario_meta: Scenario metadata used for fallback turn ids.
    """
    turn_by_key: dict[tuple[str, int], int] = {}
    rows = _load_jsonl_records(sample_file)
    for row in rows:
        prompt_id = str(row.get("prompt_id", "unknown"))

        sequential_ids = row.get("sequential_ids")
        if not isinstance(sequential_ids, list):
            sequential_ids = []

        candidate_outputs = row.get("solutions")
        if not isinstance(candidate_outputs, list):
            candidate_outputs = row.get("generations")
        if not isinstance(candidate_outputs, list):
            candidate_outputs = row.get("raw_generations")
        if not isinstance(candidate_outputs, list):
            candidate_outputs = []

        row_count = max(len(sequential_ids), len(candidate_outputs))
        for response_idx in range(row_count):
            sequential_id = (
                _safe_int(sequential_ids[response_idx])
                if response_idx < len(sequential_ids)
                else None
            )
            if sequential_id is None:
                sequential_id = _fallback_sequential_id(response_idx, scenario_meta)
            turn_by_key[(prompt_id, response_idx)] = sequential_id
    return turn_by_key


def load_sweep_results(sweep_dir: str | Path) -> SweepResults:
    """Load structured run-level sweep results from a sweep work directory.

    Args:
        sweep_dir: Sweep directory containing sweep.log and experiment_run_dirs.txt.
    """
    root = Path(sweep_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {root}")

    run_dirs = _load_run_dirs(root / "experiment_run_dirs.txt")
    by_run_log = _parse_sweep_log(root / "sweep.log")
    run_rows: list[SweepRunResult] = []

    for run_index, run_dir in enumerate(run_dirs):
        scenario_meta = _scenario_meta_for_run(run_dir)
        metrics = _load_metric_reports(run_dir)

        run_name = run_dir.name
        log_meta = by_run_log.get(run_name, {})

        src = scenario_meta.get("src")
        tgt = scenario_meta.get("tgt")
        if not isinstance(src, str):
            src = None
        if not isinstance(tgt, str):
            tgt = None

        model = scenario_meta.get("model")
        if not isinstance(model, str) or not model:
            model = run_dir.parent.name

        scenario_name = scenario_meta.get("name")
        if not isinstance(scenario_name, str) or not scenario_name:
            scenario_name = run_dir.parent.parent.name

        run_rows.append(
            SweepRunResult(
                run_index=run_index,
                run_name=run_name,
                run_dir=run_dir,
                task_id=log_meta.get("task_id"),
                status=_infer_run_status(log_meta.get("status"), metrics),
                src=src,
                tgt=tgt,
                lp=_infer_lp(run_dir, scenario_meta),
                scenario_name=scenario_name,
                model=model,
                metrics=metrics,
            )
        )

    return SweepResults(sweep_dir=root, runs=run_rows)


def load_sweep_dataframe(sweep_dir: str | Path) -> pd.DataFrame:
    """Build run+metric dataframe for one sweep.

    Args:
        sweep_dir: Sweep directory containing run index/log/artifacts.
    """
    results = load_sweep_results(sweep_dir)
    rows: list[dict[str, Any]] = []

    for run in results.runs:
        base_row = {
            "sweep_dir": str(results.sweep_dir),
            "run_index": run.run_index,
            "task_id": run.task_id,
            "run_name": run.run_name,
            "run_dir": str(run.run_dir),
            "status": run.status,
            "src": run.src,
            "tgt": run.tgt,
            "lp": run.lp,
            "scenario_name": run.scenario_name,
            "model": run.model,
        }
        if run.metrics:
            for metric in run.metrics:
                rows.append(
                    {
                        **base_row,
                        "metric": metric.metric,
                        "variant": metric.variant,
                        "average_score": metric.average_score,
                        "n": metric.n,
                        "report_file": str(metric.report_file),
                    }
                )
        else:
            rows.append(
                {
                    **base_row,
                    "metric": None,
                    "variant": None,
                    "average_score": None,
                    "n": None,
                    "report_file": None,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["run_index", "metric", "variant"], na_position="last").reset_index(drop=True)


def load_sweep_score_dataframe(sweep_dir: str | Path) -> pd.DataFrame:
    """Build score-level dataframe with sequential_id mapping for one sweep.

    Args:
        sweep_dir: Sweep directory containing run artifacts and reports.
    """
    results = load_sweep_results(sweep_dir)
    rows: list[dict[str, Any]] = []

    for run in results.runs:
        scenario_meta = _scenario_meta_for_run(run.run_dir)
        sample_cache: dict[Path, dict[tuple[str, int], int]] = {}

        for metric in run.metrics:
            try:
                report = _load_json_dict(metric.report_file)
            except json.JSONDecodeError:
                continue

            scores = report.get("scores")
            if not isinstance(scores, list):
                continue

            sample_file = _resolve_sample_file(report, run.run_dir)
            turn_map: dict[tuple[str, int], int] = {}
            if sample_file is not None:
                if sample_file not in sample_cache:
                    try:
                        sample_cache[sample_file] = _response_to_turn_map(sample_file, scenario_meta)
                    except (OSError, json.JSONDecodeError):
                        sample_cache[sample_file] = {}
                turn_map = sample_cache[sample_file]

            for score_item in scores:
                prompt_id = str(score_item.get("prompt_id", "unknown"))
                response_idx = _safe_int(score_item.get("response_idx"))
                quality = _safe_float(score_item.get("score"))
                if response_idx is None or quality is None:
                    continue

                sequential_id = turn_map.get((prompt_id, response_idx))
                if sequential_id is None:
                    sequential_id = _fallback_sequential_id(response_idx, scenario_meta)

                rows.append(
                    {
                        "sweep_dir": str(results.sweep_dir),
                        "run_index": run.run_index,
                        "task_id": run.task_id,
                        "run_name": run.run_name,
                        "run_dir": str(run.run_dir),
                        "status": run.status,
                        "src": run.src,
                        "tgt": run.tgt,
                        "lp": run.lp,
                        "scenario_name": run.scenario_name,
                        "model": run.model,
                        "metric": metric.metric,
                        "variant": metric.variant,
                        "prompt_id": prompt_id,
                        "response_idx": response_idx,
                        "sequential_id": sequential_id,
                        "quality": quality,
                        "report_file": str(metric.report_file),
                        "sample_file": str(sample_file) if sample_file is not None else None,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["run_index", "metric", "variant", "sequential_id", "prompt_id", "response_idx"],
        na_position="last",
    ).reset_index(drop=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for utility dataframe loader.

    Args:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Load MT sweep results into a structured pandas DataFrame."
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Sweep work directory (contains sweep.log and experiment_run_dirs.txt).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="Print first N rows.",
    )
    return parser


def main() -> None:
    """Run utility CLI: print/load dataframe and optionally save CSV.

    Args:
        None.
    """
    args = _build_arg_parser().parse_args()
    df = load_sweep_dataframe(args.sweep_dir)
    if df.empty:
        print("No rows found.")
        return

    print(df.head(max(0, args.head)).to_string(index=False))
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"saved_csv: {args.out_csv}")


if __name__ == "__main__":
    main()
