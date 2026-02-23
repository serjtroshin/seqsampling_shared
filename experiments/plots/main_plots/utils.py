from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

FATAL_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"\btraceback\b",
        r"\bruntimeerror\b",
        r"\bvalueerror\b",
        r"\bassertionerror\b",
        r"\bcuda out of memory\b",
        r"\bout of memory\b",
        r"\bslurmstepd: error\b",
        r"\berror:\b",
    ]
]


def parse_multi_args(raw_values: list[str] | None) -> list[str]:
    """Parse repeatable/comma-separated CLI values into a unique ordered list.

    Args:
        raw_values: Raw argparse values, possibly repeated and comma-separated.
    """
    if not raw_values:
        return []
    values: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for piece in raw.split(","):
            item = piece.strip()
            if item and item not in seen:
                seen.add(item)
                values.append(item)
    return values


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


def _contains(path: Path, marker: str) -> bool:
    """Check whether file text contains the given marker (case-insensitive).

    Args:
        path: File path to inspect.
        marker: Text marker to search for.
    """
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return marker.lower() in text


def _has_fatal_error(path: Path) -> bool:
    """Detect fatal-error patterns in stderr logs.

    Args:
        path: Log file path to inspect.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(pattern.search(text) for pattern in FATAL_PATTERNS)


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


def _discover_run_dirs(root_dir: Path) -> list[Path]:
    """Discover run directories under an experiment root.

    Args:
        root_dir: Root directory containing nested run folders.
    """
    run_dirs: set[Path] = set()
    for marker in root_dir.rglob("slurm_eval.out"):
        run_dir = marker.parent
        if (run_dir / "slurm_gen.out").exists():
            run_dirs.add(run_dir)
    return sorted(run_dirs)


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


def _load_metric_reports(
    run_dir: Path,
) -> list[dict[str, str | float | int | None]]:
    """Load metric report summaries from a run's evaluation directory.

    Args:
        run_dir: Run directory that may contain evaluation/*.json reports.
    """
    evaluation_dir = run_dir / "evaluation"
    if not evaluation_dir.is_dir():
        return []

    metric_rows: list[dict[str, str | float | int | None]] = []
    for report_file in sorted(evaluation_dir.glob("*.json")):
        try:
            report = _load_json_dict(report_file)
        except json.JSONDecodeError:
            continue
        metric, variant = _metric_variant_from_report(report_file)
        metric_rows.append(
            {
                "metric": metric,
                "variant": variant,
                "average_score": _safe_float(report.get("average_score")),
                "n": _safe_int(report.get("n")),
                "report_file": str(report_file),
            }
        )
    return metric_rows


def _infer_status(run_dir: Path, metric_count: int) -> str:
    """Infer run status from logs and report availability.

    Args:
        run_dir: Run directory with slurm logs.
        metric_count: Number of discovered metric reports for this run.
    """
    eval_done = _contains(run_dir / "slurm_eval.out", "[eval] done")
    gen_done = _contains(run_dir / "slurm_gen.out", "generation done")
    has_fatal = _has_fatal_error(run_dir / "slurm_gen.err") or _has_fatal_error(
        run_dir / "slurm_eval.err"
    )
    if has_fatal:
        return "FAILED"
    if eval_done and gen_done:
        return "DONE"
    if metric_count > 0:
        return "DONE"
    return "ACTIVE_OR_INCOMPLETE"


def _infer_src_tgt_lp(run_dir: Path, scenario_meta: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Infer src/tgt/language-pair from scenario metadata and directory structure.

    Args:
        run_dir: Run directory used for path-based fallback inference.
        scenario_meta: Parsed scenario_resolved.yaml dictionary.
    """
    src = scenario_meta.get("src")
    tgt = scenario_meta.get("tgt")
    src = src if isinstance(src, str) and src else None
    tgt = tgt if isinstance(tgt, str) and tgt else None
    lp: str | None = None

    if src and tgt:
        lp = f"{src}-{tgt}"
        return src, tgt, lp

    if len(run_dir.parents) >= 3:
        candidate = run_dir.parents[1].name
        if "-" in candidate:
            lp = candidate
            parts = candidate.split("-", maxsplit=1)
            if len(parts) == 2:
                src = src or parts[0]
                tgt = tgt or parts[1]
    return src, tgt, lp


def _infer_dataset_tag(run_dir: Path, scenario_meta: dict[str, Any]) -> tuple[str, str | None]:
    """Infer dataset tag for plot output partitioning.

    Args:
        run_dir: Run directory used for path/name fallback inference.
        scenario_meta: Parsed scenario_resolved.yaml dictionary.
    """
    prompts_path_raw = scenario_meta.get("prompts_path")
    prompts_path = prompts_path_raw if isinstance(prompts_path_raw, str) else None
    raw = (prompts_path or "").lower()
    run_name = run_dir.name.lower()
    run_path = str(run_dir).lower()

    is_doc = any(
        token in raw or token in run_name or token in run_path
        for token in [
            "plus_doc",
            "plus-doc",
            "wmttest2024_plus_doc",
            "/doc",
            "_doc",
        ]
    )
    if is_doc:
        return "wmt24pp_doc", prompts_path
    return "wmt24pp_par", prompts_path


def _fallback_model_name(root_dir: Path, run_dir: Path) -> str | None:
    """Infer model name from run relative path when metadata is missing.

    Args:
        root_dir: Experiment root directory.
        run_dir: Run directory path.
    """
    try:
        rel = run_dir.relative_to(root_dir)
    except ValueError:
        return None
    if len(rel.parts) > 0:
        return rel.parts[0]
    return None


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


def load_all_finished_dataframes(
    root_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all run metadata + metric and score rows from an experiment root.

    Args:
        root_dir: Experiment root to scan (for example outputs/mt/all_finished_runs).

    Returns:
      - runs_df: one informative row per metric per run (plus rows with metric=None when missing)
      - score_df: score-level rows with sequential_id for plotting curves
    """
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Experiment root not found: {root}")

    run_dirs = _discover_run_dirs(root)
    run_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for run_index, run_dir in enumerate(run_dirs):
        scenario_meta = _scenario_meta_for_run(run_dir)
        metrics = _load_metric_reports(run_dir)
        status = _infer_status(run_dir, metric_count=len(metrics))

        src, tgt, lp = _infer_src_tgt_lp(run_dir, scenario_meta)
        dataset_tag, prompts_path = _infer_dataset_tag(run_dir, scenario_meta)
        run_name = run_dir.name
        scenario_name = scenario_meta.get("name")
        if not isinstance(scenario_name, str) or not scenario_name:
            scenario_name = run_dir.parent.name

        model = scenario_meta.get("model")
        if not isinstance(model, str) or not model:
            model = _fallback_model_name(root, run_dir)

        sampling_profile = None
        try:
            rel_parts = run_dir.relative_to(root).parts
            if len(rel_parts) > 1:
                sampling_profile = rel_parts[1]
        except ValueError:
            sampling_profile = None

        base = {
            "root_dir": str(root),
            "run_index": run_index,
            "task_id": None,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "run_rel_path": str(run_dir.relative_to(root)),
            "status": status,
            "src": src,
            "tgt": tgt,
            "lp": lp,
            "scenario_name": scenario_name,
            "model": model,
            "sampling_profile": sampling_profile,
            "dataset_tag": dataset_tag,
            "prompts_path": prompts_path,
        }

        if metrics:
            for metric in metrics:
                run_rows.append({**base, **metric})
        else:
            run_rows.append(
                {
                    **base,
                    "metric": None,
                    "variant": None,
                    "average_score": None,
                    "n": None,
                    "report_file": None,
                }
            )

        sample_cache: dict[Path, dict[tuple[str, int], int]] = {}
        for metric in metrics:
            report_file_raw = metric.get("report_file")
            if not isinstance(report_file_raw, str):
                continue
            report_path = Path(report_file_raw)
            try:
                report = _load_json_dict(report_path)
            except json.JSONDecodeError:
                continue

            scores = report.get("scores")
            if not isinstance(scores, list):
                continue

            sample_file = _resolve_sample_file(report, run_dir)
            turn_map: dict[tuple[str, int], int] = {}
            if sample_file is not None:
                if sample_file not in sample_cache:
                    try:
                        sample_cache[sample_file] = _response_to_turn_map(
                            sample_file, scenario_meta
                        )
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

                score_rows.append(
                    {
                        **base,
                        "metric": metric["metric"],
                        "variant": metric["variant"],
                        "prompt_id": prompt_id,
                        "response_idx": response_idx,
                        "sequential_id": sequential_id,
                        "quality": quality,
                        "report_file": report_file_raw,
                        "sample_file": str(sample_file) if sample_file is not None else None,
                    }
                )

    runs_df = pd.DataFrame(run_rows)
    if not runs_df.empty:
        runs_df = runs_df.sort_values(
            ["run_index", "metric", "variant"], na_position="last"
        ).reset_index(drop=True)

    score_df = pd.DataFrame(score_rows)
    if not score_df.empty:
        score_df = score_df.sort_values(
            ["run_index", "metric", "variant", "sequential_id", "prompt_id", "response_idx"],
            na_position="last",
        ).reset_index(drop=True)

    return runs_df, score_df


def load_all_finished_runs_dataframe(root_dir: str | Path) -> pd.DataFrame:
    """Convenience API for the run-level informative dataframe.

    Args:
        root_dir: Experiment root to scan.
    """
    runs_df, _ = load_all_finished_dataframes(root_dir)
    return runs_df


def load_all_finished_scores_dataframe(root_dir: str | Path) -> pd.DataFrame:
    """Convenience API for the score-level dataframe used by curve plots.

    Args:
        root_dir: Experiment root to scan.
    """
    _, score_df = load_all_finished_dataframes(root_dir)
    return score_df
