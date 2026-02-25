from __future__ import annotations

import argparse
import json
import math
import os
from pprint import pprint
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import pandas as pd
import yaml

try:
    from experiments.plots.main_plots.utils import (
        load_all_finished_dataframes,
        parse_multi_args,
    )
except ModuleNotFoundError:
    from utils import load_all_finished_dataframes, parse_multi_args


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the draft parallel-vs-sequential matcher.

    Args:
        None.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Draft utility: match parallel runs with sequential runs that share "
            "the same model/lang setup."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/mt/all_finished_runs"),
        help="Root directory with merged finished runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/mt/all_finished_runs__plots/plot_parallel_vs_sequential"),
        help="Output directory for generated parallel-vs-sequential plots.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional cap on number of printed match groups.",
    )
    parser.add_argument(
        "--max-prompts-print",
        type=int,
        default=20,
        help="Draft quality label print cap per (run, metric, variant). Use negative to print all.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric filter (repeatable or comma-separated). Default: all available.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant filter (repeatable or comma-separated). Default: all available.",
    )
    parser.add_argument(
        "--parallel-iqr-tail-percent",
        type=float,
        default=5.0,
        help="Save low/high grouped samples using this tail percent on each side (default: 5).",
    )
    return parser


def _is_parallel_run(row: pd.Series) -> bool:
    """Heuristic to identify parallel runs.

    Args:
        row: Run-level metadata row.
    """
    scenario_name = str(row.get("scenario_name") or "").lower()
    run_rel_path = str(row.get("run_rel_path") or "").lower()
    run_name = str(row.get("run_name") or "").lower()
    return (
        "parallel" in scenario_name
        or "/mt_parallel/" in run_rel_path
        or "parallel" in run_name
    )


def _lang_key(row: pd.Series) -> str:
    """Return language key used for matching.

    Args:
        row: Run-level metadata row.
    """
    lp = row.get("lp")
    tgt = row.get("tgt")
    if isinstance(lp, str) and lp:
        return lp
    if isinstance(tgt, str) and tgt:
        return tgt
    return "unknown"


def _dataset_key(row: pd.Series) -> str:
    """Return dataset tag key used for matching.

    Args:
        row: Run-level metadata row.
    """
    dataset_tag = row.get("dataset_tag")
    if isinstance(dataset_tag, str) and dataset_tag:
        return dataset_tag
    return "unknown_dataset"


def _to_run_info(row: pd.Series) -> dict[str, Any]:
    """Convert dataframe row into a compact printable run dictionary.

    Args:
        row: Run-level metadata row.
    """
    return {
        "run_index": int(row.get("run_index", -1)),
        "run_name": str(row.get("run_name") or ""),
        "scenario_name": str(row.get("scenario_name") or ""),
        "status": str(row.get("status") or ""),
        "model": str(row.get("model") or ""),
        "lang": _lang_key(row),
        "dataset_tag": _dataset_key(row),
        "run_rel_path": str(row.get("run_rel_path") or ""),
    }


def _path_component(value: str, fallback: str) -> str:
    """Keep path components human-readable while avoiding nested separators.

    Args:
        value: Raw path component value.
        fallback: Fallback value when input is empty.
    """
    cleaned = value.strip()
    if not cleaned:
        cleaned = fallback
    return cleaned.replace("/", "-")


def _parallel_vs_sequential_out_dir(
    out_dir: Path,
    parallel_row: pd.Series,
    metric: str,
    variant: str,
) -> Path:
    """Build output directory for one parallel run + metric/variant plot bundle.

    Args:
        out_dir: Root output directory for this plot family.
        parallel_row: Parallel run metadata row.
        metric: Metric name.
        variant: Variant name.
    """
    dataset_tag = _path_component(_dataset_key(parallel_row), "unknown_dataset")
    lang = _path_component(_lang_key(parallel_row), "unknown")
    scenario_name = _path_component(str(parallel_row.get("scenario_name") or ""), "unknown_scenario")
    run_name = _path_component(str(parallel_row.get("run_name") or ""), "unknown_run")
    metric_name = _path_component(metric, "metric")
    if variant != "default":
        metric_name = f"{metric_name}__{_path_component(variant, 'variant')}"
    return out_dir / dataset_tag / lang / scenario_name / f"{metric_name}__{run_name}"


def _parallel_bundle_parallel_plot_path(bundle_dir: Path) -> Path:
    """Build output path for the parallel plot in a bundle directory.

    Args:
        bundle_dir: Bundle directory path for one parallel run + metric.
    """
    return bundle_dir / "parallel.png"


def _parallel_bundle_sequential_plot_path(bundle_dir: Path, seq_row: pd.Series) -> Path:
    """Build output path for one matched sequential plot in the bundle.

    Args:
        bundle_dir: Bundle directory path for one parallel run + metric.
        seq_row: Sequential run metadata row.
    """
    scenario_name = _path_component(str(seq_row.get("scenario_name") or ""), "unknown_scenario")
    run_name = _path_component(str(seq_row.get("run_name") or ""), "unknown_run")
    return bundle_dir / f"{scenario_name}__{run_name}.png"


def _read_num_generations_for_run(run_row: pd.Series) -> int:
    """Load num_generations from scenario_resolved.yaml for a run.

    Args:
        run_row: Run-level metadata row containing run_dir.
    """
    run_dir_raw = run_row.get("run_dir")
    if not isinstance(run_dir_raw, str) or not run_dir_raw:
        raise ValueError(f"Missing run_dir for run: {run_row.get('run_name')}")
    run_dir = Path(run_dir_raw)
    scenario_paths = sorted((run_dir / "generation").rglob("scenario_resolved.yaml"))
    if not scenario_paths:
        raise FileNotFoundError(f"scenario_resolved.yaml not found under: {run_dir / 'generation'}")
    payload = yaml.safe_load(scenario_paths[0].read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid scenario_resolved.yaml payload: {scenario_paths[0]}")
    raw_num = payload.get("num_generations")
    if raw_num is None:
        raise KeyError(f"num_generations is missing in: {scenario_paths[0]}")
    num_generations = int(raw_num)
    if num_generations <= 0:
        raise ValueError(f"num_generations must be > 0 in: {scenario_paths[0]}")
    return num_generations


def _compute_prompt_quality_labels(
    run_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute q1/median/q3 and IQR labels per prompt_id.

    Args:
        run_scores_df: Score rows for exactly one run and one metric/variant.
    """
    grouped = run_scores_df.groupby("prompt_id")["quality"]
    out_df = pd.DataFrame(
        {
            "q1": grouped.quantile(0.25),
            "median": grouped.quantile(0.50),
            "q3": grouped.quantile(0.75),
            "n_samples": grouped.size(),
        }
    ).reset_index()

    def _iqr(row: pd.Series) -> float:
        q1 = float(row["q1"])
        q3 = float(row["q3"])
        return q3 - q1

    out_df["iqr"] = out_df.apply(_iqr, axis=1)
    return out_df.sort_values("prompt_id").reset_index(drop=True)


def _safe_int(value: Any) -> int | None:
    """Safely coerce an arbitrary value to int.

    Args:
        value: Input value.
    """
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from a file.

    Args:
        path: JSONL path.
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


def _sample_file_from_score_rows(score_rows: pd.DataFrame) -> Path | None:
    """Resolve sample file path from score rows for one run+metric+variant.

    Args:
        score_rows: Score dataframe slice.
    """
    if "sample_file" not in score_rows.columns:
        return None
    candidates: list[str] = []
    for raw in score_rows["sample_file"].dropna().tolist():
        if isinstance(raw, str) and raw.strip():
            candidates.append(raw.strip())
    if not candidates:
        return None
    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) > 1:
        print(
            "[MATCH CHECK FAIL] multiple sample files for one score slice "
            f"using_first={unique_candidates[0]} all={unique_candidates[:5]}"
        )
    return Path(unique_candidates[0])


def _sample_outputs_from_row(row: dict[str, Any]) -> list[str]:
    """Extract a best-effort ordered list of outputs from one sample record.

    Args:
        row: One JSON object from samples*.jsonl.
    """
    for key in ("solutions", "generations", "raw_generations"):
        values = row.get(key)
        if isinstance(values, list):
            return [str(v) for v in values]
    return []


def _build_prompt_sample_index(sample_file: Path) -> dict[str, dict[str, Any]]:
    """Build prompt_id -> sample payload index from samples JSONL.

    Args:
        sample_file: Source JSONL with generation rows.
    """
    prompt_index: dict[str, dict[str, Any]] = {}
    rows = _load_jsonl_records(sample_file)
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id:
            continue

        outputs = _sample_outputs_from_row(row)
        seq_ids_raw = row.get("sequential_ids")
        seq_ids = seq_ids_raw if isinstance(seq_ids_raw, list) else []
        samples: list[dict[str, Any]] = []
        for response_idx, text in enumerate(outputs):
            seq_id = seq_ids[response_idx] if response_idx < len(seq_ids) else None
            samples.append(
                {
                    "response_idx": response_idx,
                    "sequential_id": _safe_int(seq_id),
                    "text": text,
                }
            )

        prompt_index[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt": row.get("prompt"),
            "source": row.get("source"),
            "target": row.get("target"),
            "lp": row.get("lp"),
            "domain": row.get("domain"),
            "document_id": row.get("document_id"),
            "segment_id": row.get("segment_id"),
            "samples": samples,
        }
    return prompt_index


def _write_grouped_samples_jsonl(
    *,
    out_path: Path,
    bucket_name: str,
    prompt_ids: list[str],
    parallel_stats: pd.DataFrame,
    sequential_stats: pd.DataFrame,
    parallel_prompt_index: dict[str, dict[str, Any]],
    sequential_prompt_index: dict[str, dict[str, Any]],
    parallel_row: pd.Series,
    sequential_row: pd.Series,
    metric: str,
    variant: str,
    tail_percent: float,
    tail_count: int,
    total_prompts: int,
) -> None:
    """Write grouped prompt sample comparisons for one bucket to JSONL.

    Args:
        out_path: Output JSONL path.
        bucket_name: Bucket identifier, for example parallel_iqr_low.
        prompt_ids: Ordered prompt IDs to export.
        parallel_stats: Prompt-indexed parallel stats dataframe.
        sequential_stats: Prompt-indexed sequential stats dataframe.
        parallel_prompt_index: prompt_id -> sample payload for parallel run.
        sequential_prompt_index: prompt_id -> sample payload for sequential run.
        parallel_row: Parallel run metadata row.
        sequential_row: Sequential run metadata row.
        metric: Metric name.
        variant: Metric variant.
        tail_percent: Percentile size used for low/high tails.
        tail_count: Number of prompts selected in this tail.
        total_prompts: Total aligned prompt count.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path = out_path.with_suffix(".txt")
    with out_path.open("w", encoding="utf-8") as handle:
        txt_blocks: list[str] = []
        for prompt_id in prompt_ids:
            if prompt_id not in parallel_stats.index or prompt_id not in sequential_stats.index:
                continue
            p_stats = parallel_stats.loc[prompt_id]
            s_stats = sequential_stats.loc[prompt_id]
            parallel_median = float(p_stats["median"])

            p_payload = parallel_prompt_index.get(prompt_id, {})
            s_payload = sequential_prompt_index.get(prompt_id, {})
            prompt_text = p_payload.get("prompt") or s_payload.get("prompt")
            source_text = p_payload.get("source") or s_payload.get("source")
            target_text = p_payload.get("target") or s_payload.get("target")

            row = {
                "bucket": bucket_name,
                "prompt_id": prompt_id,
                "metric": metric,
                "variant": variant,
                "parallel_iqr_tail_percent": tail_percent,
                "parallel_iqr_tail_count": tail_count,
                "parallel_iqr_total_prompts": total_prompts,
                "parallel_run_name": str(parallel_row.get("run_name") or ""),
                "parallel_scenario_name": str(parallel_row.get("scenario_name") or ""),
                "sequential_run_name": str(sequential_row.get("run_name") or ""),
                "sequential_scenario_name": str(sequential_row.get("scenario_name") or ""),
                "prompt": prompt_text,
                "source": source_text,
                "target": target_text,
                "parallel_stats": {
                    "q1": float(p_stats["q1"]),
                    "median": parallel_median,
                    "q3": float(p_stats["q3"]),
                    "iqr": float(p_stats["iqr"]),
                    "n_samples": int(p_stats["n_samples"]),
                },
                "sequential_stats": {
                    "q1": float(s_stats["q1"]),
                    "median": float(s_stats["median"]),
                    "q3": float(s_stats["q3"]),
                    "iqr": float(s_stats["iqr"]),
                    "n_samples": int(s_stats["n_samples"]),
                    "q1_minus_parallel_median": float(s_stats["q1"]) - parallel_median,
                    "q3_minus_parallel_median": float(s_stats["q3"]) - parallel_median,
                },
                "parallel_samples": p_payload.get("samples", []),
                "sequential_samples": s_payload.get("samples", []),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            parallel_sample_texts = [
                str(item.get("text", ""))
                for item in p_payload.get("samples", [])
                if isinstance(item, dict)
            ]
            sequential_sample_texts = [
                str(item.get("text", ""))
                for item in s_payload.get("samples", [])
                if isinstance(item, dict)
            ]
            prompt_lines: list[str] = [
                f"prompt_id={prompt_id}",
                f"source={source_text}",
                f"====parallel outputs q1={float(p_stats['q1']):.6f} q3={float(p_stats['q3']):.6f} ====",
                *parallel_sample_texts,
                f"====sequential outputs q1={float(s_stats['q1']):.6f} q3={float(s_stats['q3']):.6f} ====",
                *sequential_sample_texts,
                "===============================",
            ]
            txt_blocks.append("\n".join(prompt_lines))
    txt_path.write_text("\n\n".join(txt_blocks), encoding="utf-8")
    print(f"saved_jsonl: {out_path} prompts={len(prompt_ids)}")
    print(f"saved_txt: {txt_path} prompts={len(txt_blocks)}")


def _save_parallel_iqr_grouped_samples(
    *,
    bundle_dir: Path,
    metric: str,
    variant: str,
    parallel_row: pd.Series,
    sequential_row: pd.Series,
    parallel_labels_df: pd.DataFrame,
    seq_aligned_df: pd.DataFrame,
    parallel_prompt_index: dict[str, dict[str, Any]],
    sequential_prompt_index: dict[str, dict[str, Any]],
    tail_percent: float,
) -> None:
    """Save low/high parallel-IQR prompt buckets with parallel+sequential samples.

    Args:
        bundle_dir: Output bundle directory for one parallel run and metric.
        metric: Metric name.
        variant: Metric variant.
        parallel_row: Parallel run metadata row.
        sequential_row: Sequential run metadata row.
        parallel_labels_df: Prompt stats for parallel run (already sorted by IQR).
        seq_aligned_df: Sequential prompt stats aligned to parallel prompt order.
        parallel_prompt_index: prompt_id -> parallel sample payload.
        sequential_prompt_index: prompt_id -> sequential sample payload.
        tail_percent: Percentile size used for low/high tails.
    """
    parallel_stats = parallel_labels_df.copy()
    parallel_stats["prompt_id"] = parallel_stats["prompt_id"].astype(str)
    parallel_stats = parallel_stats.set_index("prompt_id")

    sequential_stats = seq_aligned_df.copy()
    sequential_stats["prompt_id"] = sequential_stats["prompt_id"].astype(str)
    sequential_stats = sequential_stats.set_index("prompt_id")

    prompt_order = [str(pid) for pid in parallel_labels_df["prompt_id"].tolist()]
    if not prompt_order:
        return
    tail_count = max(1, math.ceil(len(prompt_order) * (tail_percent / 100.0)))
    tail_count = min(len(prompt_order), tail_count)
    high_prompt_ids = prompt_order[:tail_count]
    low_prompt_ids = prompt_order[-tail_count:]

    scenario_name = _path_component(str(sequential_row.get("scenario_name") or ""), "unknown_scenario")
    run_name = _path_component(str(sequential_row.get("run_name") or ""), "unknown_run")
    prefix = f"samples__{scenario_name}__{run_name}"
    low_path = bundle_dir / f"{prefix}__parallel_iqr_low.jsonl"
    high_path = bundle_dir / f"{prefix}__parallel_iqr_high.jsonl"

    _write_grouped_samples_jsonl(
        out_path=low_path,
        bucket_name="parallel_iqr_low",
        prompt_ids=low_prompt_ids,
        parallel_stats=parallel_stats,
        sequential_stats=sequential_stats,
        parallel_prompt_index=parallel_prompt_index,
        sequential_prompt_index=sequential_prompt_index,
        parallel_row=parallel_row,
        sequential_row=sequential_row,
        metric=metric,
        variant=variant,
        tail_percent=tail_percent,
        tail_count=tail_count,
        total_prompts=len(prompt_order),
    )
    _write_grouped_samples_jsonl(
        out_path=high_path,
        bucket_name="parallel_iqr_high",
        prompt_ids=high_prompt_ids,
        parallel_stats=parallel_stats,
        sequential_stats=sequential_stats,
        parallel_prompt_index=parallel_prompt_index,
        sequential_prompt_index=sequential_prompt_index,
        parallel_row=parallel_row,
        sequential_row=sequential_row,
        metric=metric,
        variant=variant,
        tail_percent=tail_percent,
        tail_count=tail_count,
        total_prompts=len(prompt_order),
    )


def _draw_quantile_iqr_panel(
    ax1: plt.Axes,
    labels_df: pd.DataFrame,
    *,
    iqr_values: list[float] | None = None,
    iqr_label: str = "iqr (q3-q1)",
    iqr_color: str = "#d62728",
    iqr_values_secondary: list[float] | None = None,
    iqr_label_secondary: str = "secondary iqr (q3-q1)",
    iqr_color_secondary: str = "#9467bd",
    baseline_median_values: list[float] | None = None,
    plot_median: bool = True,
    context_name: str = "plot",
) -> plt.Axes:
    """Draw q1/q3 (+optional median) and one/two IQR overlays on given axes.

    Args:
        ax1: Left-axis matplotlib Axes object.
        labels_df: Prompt-level q1/median/q3/iqr dataframe.
        iqr_values: Optional explicit primary IQR values.
        iqr_label: Legend label for primary IQR.
        iqr_color: Color for primary IQR.
        iqr_values_secondary: Optional secondary IQR values.
        iqr_label_secondary: Legend label for secondary IQR.
        iqr_color_secondary: Color for secondary IQR.
        baseline_median_values: Optional baseline median for q1/q3 centering.
        plot_median: Whether to draw median series.
        context_name: Debug label used in validation error messages.
    """
    x = list(range(len(labels_df)))
    q1 = labels_df["q1"].astype(float).tolist()
    median = labels_df["median"].astype(float).tolist()
    q3 = labels_df["q3"].astype(float).tolist()

    if baseline_median_values is not None:
        baseline = [float(v) for v in baseline_median_values]
        if len(baseline) != len(labels_df):
            raise ValueError(
                "baseline_median_values length mismatch: "
                f"{len(baseline)} != {len(labels_df)} for {context_name}"
            )
        q1 = [v - b for v, b in zip(q1, baseline)]
        q3 = [v - b for v, b in zip(q3, baseline)]
        median = [v - b for v, b in zip(median, baseline)]

    if iqr_values is None:
        iqr_vals = labels_df["iqr"].astype(float).tolist()
    else:
        iqr_vals = [float(v) for v in iqr_values]
        if len(iqr_vals) != len(labels_df):
            raise ValueError(
                "iqr_values length mismatch: "
                f"{len(iqr_vals)} != {len(labels_df)} for {context_name}"
            )

    if iqr_values_secondary is not None:
        iqr_vals_secondary = [float(v) for v in iqr_values_secondary]
        if len(iqr_vals_secondary) != len(labels_df):
            raise ValueError(
                "iqr_values_secondary length mismatch: "
                f"{len(iqr_vals_secondary)} != {len(labels_df)} for {context_name}"
            )
    else:
        iqr_vals_secondary = None

    ax1.scatter(x, q1, s=10, alpha=0.75, label="q1", color="#1f77b4")
    if plot_median:
        ax1.scatter(x, median, s=10, alpha=0.75, label="median", color="#2ca02c")
    ax1.scatter(x, q3, s=10, alpha=0.75, label="q3", color="#ff7f0e")
    ax1.set_xlabel("prompt rank (sorted by iqr desc)")
    ax1.set_ylabel(
        "quality quantiles" if baseline_median_values is None else "delta vs parallel median"
    )
    ax1.grid(alpha=0.25, linewidth=0.7)

    ax2 = ax1.twinx()
    ax2.scatter(x, iqr_vals, s=9, alpha=0.45, label=iqr_label, color=iqr_color)
    if iqr_vals_secondary is not None:
        ax2.scatter(
            x,
            iqr_vals_secondary,
            s=9,
            alpha=0.45,
            label=iqr_label_secondary,
            color=iqr_color_secondary,
        )
    ax2.set_ylabel("iqr")
    ax2.set_ylim(0.0, 1.0)
    return ax2


def _plot_prompt_stats_scatter(
    labels_df: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    iqr_values: list[float] | None = None,
    iqr_label: str = "iqr (q3-q1)",
    iqr_color: str = "#d62728",
    iqr_values_secondary: list[float] | None = None,
    iqr_label_secondary: str = "secondary iqr (q3-q1)",
    iqr_color_secondary: str = "#9467bd",
    baseline_median_values: list[float] | None = None,
    plot_median: bool = True,
) -> None:
    """Plot q1/median/q3 and IQR over prompt rank.

    Args:
        labels_df: Per-prompt stats dataframe sorted by desired prompt order.
        title: Plot title.
        out_path: Target PNG path.
        iqr_values: Optional explicit IQR values to plot on red axis.
        iqr_label: Legend label for red IQR series.
        iqr_color: Color for primary IQR series.
        iqr_values_secondary: Optional secondary IQR series to overlay.
        iqr_label_secondary: Legend label for secondary IQR series.
        iqr_color_secondary: Color for secondary IQR series.
        baseline_median_values: Optional baseline median values used to center q1/q3.
        plot_median: Whether to plot the median series on the left axis.
    """
    if labels_df.empty:
        return

    fig_w = max(9.0, min(18.0, 0.03 * len(labels_df) + 8.0))
    fig, ax1 = plt.subplots(figsize=(fig_w, 5.2))
    ax2 = _draw_quantile_iqr_panel(
        ax1,
        labels_df,
        iqr_values=iqr_values,
        iqr_label=iqr_label,
        iqr_color=iqr_color,
        iqr_values_secondary=iqr_values_secondary,
        iqr_label_secondary=iqr_label_secondary,
        iqr_color_secondary=iqr_color_secondary,
        baseline_median_values=baseline_median_values,
        plot_median=plot_median,
        context_name=str(out_path),
    )

    ax1.set_title(title)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot: {out_path}")


def _build_turn_delta_dataframe(
    *,
    seq_scores_df: pd.DataFrame,
    parallel_labels_df: pd.DataFrame,
    prompt_order: list[str],
) -> pd.DataFrame:
    """Build prompt-level diagnostics for turn-based delta scatter panels.

    Args:
        seq_scores_df: Sequential score rows for one run and one metric/variant.
        parallel_labels_df: Parallel prompt stats dataframe (includes IQR).
        prompt_order: Prompt order inherited from parallel run.
    """
    if seq_scores_df.empty:
        return pd.DataFrame()

    turn_col = "sequential_id" if "sequential_id" in seq_scores_df.columns else "response_idx"
    local = seq_scores_df.copy()
    local["prompt_id"] = local["prompt_id"].astype(str)
    local["turn_id"] = local[turn_col].map(_safe_int)
    turn_means = (
        local.dropna(subset=["prompt_id", "quality", "turn_id"])
        .groupby(["prompt_id", "turn_id"], as_index=False)["quality"]
        .mean()
    )
    if turn_means.empty:
        return pd.DataFrame()

    pivot = turn_means.pivot(index="prompt_id", columns="turn_id", values="quality")
    if 0 not in pivot.columns or 1 not in pivot.columns:
        return pd.DataFrame()
    last_turn_rows = turn_means.sort_values("turn_id").groupby("prompt_id").tail(1)
    last_turn_map = {
        str(row["prompt_id"]): float(row["quality"])
        for _, row in last_turn_rows.iterrows()
    }

    parallel_stats_by_prompt = (
        parallel_labels_df.assign(prompt_id=parallel_labels_df["prompt_id"].astype(str))
        .set_index("prompt_id")[["iqr", "median"]]
        .astype(float)
    )

    diag = pd.DataFrame({"prompt_id": [str(pid) for pid in prompt_order]})
    turn0_map = pivot[0].astype(float).to_dict()
    turn1_map = pivot[1].astype(float).to_dict()
    diag["turn0_quality"] = diag["prompt_id"].map(lambda pid: turn0_map.get(pid))
    diag["turn1_quality"] = diag["prompt_id"].map(lambda pid: turn1_map.get(pid))
    diag["last_turn_quality"] = diag["prompt_id"].map(lambda pid: last_turn_map.get(pid))
    diag["turn1_minus_turn0"] = diag["turn1_quality"] - diag["turn0_quality"]
    diag["last_turn_minus_turn0"] = diag["last_turn_quality"] - diag["turn0_quality"]
    diag["parallel_iqr"] = diag["prompt_id"].map(
        lambda pid: parallel_stats_by_prompt["iqr"].get(pid)
    )
    diag["parallel_median"] = diag["prompt_id"].map(
        lambda pid: parallel_stats_by_prompt["median"].get(pid)
    )
    diag["parallel_median_minus_turn0"] = diag["parallel_median"] - diag["turn0_quality"]
    return diag.dropna(
        subset=[
            "turn0_quality",
            "turn1_quality",
            "turn1_minus_turn0",
            "last_turn_quality",
            "last_turn_minus_turn0",
            "parallel_iqr",
            "parallel_median",
            "parallel_median_minus_turn0",
        ]
    ).reset_index(drop=True)


def _sample_turn_text_map(sample_payload: dict[str, Any]) -> dict[int, str]:
    """Build turn_id -> output text map from one prompt sample payload.

    Args:
        sample_payload: One prompt payload from _build_prompt_sample_index.
    """
    out: dict[int, str] = {}
    samples = sample_payload.get("samples")
    if not isinstance(samples, list):
        return out
    for item in samples:
        if not isinstance(item, dict):
            continue
        turn_id = _safe_int(item.get("sequential_id"))
        if turn_id is None:
            turn_id = _safe_int(item.get("response_idx"))
        if turn_id is None:
            continue
        if turn_id in out:
            continue
        out[turn_id] = str(item.get("text", ""))
    return out


def _build_turn01_diff_binned_dataframe(
    *,
    diag_df: pd.DataFrame,
    sequential_prompt_index: dict[str, dict[str, Any]],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin turn0 quality and estimate P(turn1 output != turn0 output) per bin.

    Args:
        diag_df: Prompt-level diagnostics with at least prompt_id and turn0_quality.
        sequential_prompt_index: prompt_id -> prompt sample payload for sequential run.
        n_bins: Number of bins for turn0 quality.
    """
    if diag_df.empty or not sequential_prompt_index:
        return pd.DataFrame()

    turn0_quality_map = (
        diag_df.assign(prompt_id=diag_df["prompt_id"].astype(str))
        .set_index("prompt_id")["turn0_quality"]
        .astype(float)
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for prompt_id, prompt_payload in sequential_prompt_index.items():
        if prompt_id not in turn0_quality_map:
            continue
        turn_map = _sample_turn_text_map(prompt_payload)
        if 0 not in turn_map or 1 not in turn_map:
            continue
        t0 = turn_map[0].strip()
        t1 = turn_map[1].strip()
        rows.append(
            {
                "prompt_id": str(prompt_id),
                "turn0_quality": float(turn0_quality_map[prompt_id]),
                "turn1_diff_from_turn0": 1.0 if t1 != t0 else 0.0,
            }
        )

    diff_df = pd.DataFrame(rows)
    if diff_df.empty:
        return pd.DataFrame()

    unique_turn0 = int(diff_df["turn0_quality"].nunique())
    if unique_turn0 <= 1:
        return pd.DataFrame(
            [
                {
                    "bin_center": float(diff_df["turn0_quality"].iloc[0]),
                    "prob_diff": float(diff_df["turn1_diff_from_turn0"].mean()),
                    "count": int(len(diff_df)),
                    "bin_label": "all",
                }
            ]
        )

    bins = max(2, min(int(n_bins), unique_turn0))
    # Use quantile bins so each bin has roughly equal number of prompts.
    diff_df["bin"] = pd.qcut(
        diff_df["turn0_quality"],
        q=bins,
        duplicates="drop",
    )
    grouped = (
        diff_df.dropna(subset=["bin"])
        .groupby("bin", as_index=False, observed=False)
        .agg(
            prob_diff=("turn1_diff_from_turn0", "mean"),
            count=("turn1_diff_from_turn0", "size"),
        )
    )
    if grouped.empty:
        return pd.DataFrame()

    grouped["bin_center"] = grouped["bin"].apply(
        lambda interval: float((interval.left + interval.right) / 2.0)
    )
    grouped["bin_label"] = grouped["bin"].astype(str)
    return grouped.sort_values("bin_center").reset_index(drop=True)


def _plot_sequential_prompt_stats_subplots(
    *,
    seq_aligned_df: pd.DataFrame,
    seq_scores_df: pd.DataFrame,
    sequential_prompt_index: dict[str, dict[str, Any]],
    parallel_labels_df: pd.DataFrame,
    prompt_order: list[str],
    title: str,
    out_path: Path,
    iqr_values: list[float],
    iqr_label: str,
    iqr_values_secondary: list[float],
    iqr_label_secondary: str,
    baseline_median_values: list[float],
) -> None:
    """Create 6-row sequential diagnostic plot with turn-delta scatter panels.

    Args:
        seq_aligned_df: Sequential q1/median/q3/iqr aligned to parallel prompt order.
        seq_scores_df: Raw sequential score rows for one metric/variant.
        sequential_prompt_index: prompt_id -> sequential sample payload with outputs.
        parallel_labels_df: Parallel prompt stats for this run+metric+variant.
        prompt_order: Prompt order from parallel run.
        title: Figure title.
        out_path: Target PNG path.
        iqr_values: Parallel IQR values in prompt_order.
        iqr_label: Label for parallel IQR series.
        iqr_values_secondary: Sequential IQR values in prompt_order.
        iqr_label_secondary: Label for sequential IQR series.
        baseline_median_values: Parallel median values used to center q1/q3.
    """
    if seq_aligned_df.empty:
        return

    diag_df = _build_turn_delta_dataframe(
        seq_scores_df=seq_scores_df,
        parallel_labels_df=parallel_labels_df,
        prompt_order=prompt_order,
    )
    turn01_diff_binned_df = _build_turn01_diff_binned_dataframe(
        diag_df=diag_df,
        sequential_prompt_index=sequential_prompt_index,
        n_bins=10,
    )

    fig_w = max(9.0, min(18.0, 0.03 * len(seq_aligned_df) + 8.0))
    fig, axes = plt.subplots(6, 1, figsize=(fig_w, 25.2))

    ax1 = axes[0]
    ax1_twin = _draw_quantile_iqr_panel(
        ax1,
        seq_aligned_df,
        iqr_values=iqr_values,
        iqr_label=iqr_label,
        iqr_values_secondary=iqr_values_secondary,
        iqr_label_secondary=iqr_label_secondary,
        baseline_median_values=baseline_median_values,
        plot_median=False,
        context_name=str(out_path),
    )
    ax1.set_title(title)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)

    ax2 = axes[1]
    if diag_df.empty:
        ax2.text(
            0.5, 0.5, "no turn0/turn1 overlap for diagnostics", ha="center", va="center"
        )
        ax2.set_axis_off()
    else:
        sc1 = ax2.scatter(
            diag_df["turn0_quality"],
            diag_df["turn1_minus_turn0"],
            c=diag_df["parallel_iqr"],
            cmap="coolwarm",
            s=12,
            alpha=0.75,
        )
        ax2.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.7)
        ax2.set_xlabel("sequential quality (turn0)")
        ax2.set_ylabel("sequential quality (turn1 - turn0)")
        ax2.set_title("turn0 vs turn1-turn0 | color: parallel IQR")
        ax2.grid(alpha=0.25, linewidth=0.7)
        cbar1 = fig.colorbar(sc1, ax=ax2, pad=0.01)
        cbar1.set_label("parallel IQR (q3-q1)")

    ax3 = axes[2]
    if diag_df.empty:
        ax3.text(
            0.5, 0.5, "no turn0/turn1 overlap for diagnostics", ha="center", va="center"
        )
        ax3.set_axis_off()
    else:
        sc2 = ax3.scatter(
            diag_df["parallel_iqr"],
            diag_df["turn1_minus_turn0"],
            c=diag_df["turn0_quality"],
            cmap="coolwarm",
            s=12,
            alpha=0.75,
        )
        ax3.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.7)
        ax3.set_xlabel("parallel IQR (q3-q1)")
        ax3.set_ylabel("sequential quality (turn1 - turn0)")
        ax3.set_title("parallel IQR vs turn1-turn0 | color: sequential turn0 quality")
        ax3.grid(alpha=0.25, linewidth=0.7)
        cbar2 = fig.colorbar(sc2, ax=ax3, pad=0.01)
        cbar2.set_label("sequential quality (turn0)")

    ax4 = axes[3]
    if diag_df.empty:
        ax4.text(
            0.5, 0.5, "no turn0/turn1 overlap for diagnostics", ha="center", va="center"
        )
        ax4.set_axis_off()
    else:
        sc3 = ax4.scatter(
            diag_df["turn1_minus_turn0"],
            diag_df["parallel_median_minus_turn0"],
            c=diag_df["parallel_iqr"],
            cmap="coolwarm",
            s=12,
            alpha=0.75,
        )
        ax4.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.7)
        ax4.axvline(0.0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.7)
        ax4.set_xlabel("sequential quality (turn1 - turn0)")
        ax4.set_ylabel("parallel median - sequential turn0")
        ax4.set_title(
            "turn1-turn0 vs (parallel median - turn0) | color: parallel IQR"
        )
        ax4.grid(alpha=0.25, linewidth=0.7)
        cbar3 = fig.colorbar(sc3, ax=ax4, pad=0.01)
        cbar3.set_label("parallel IQR (q3-q1)")

    ax5 = axes[4]
    if diag_df.empty:
        ax5.text(
            0.5, 0.5, "no turn0/turn1 overlap for diagnostics", ha="center", va="center"
        )
        ax5.set_axis_off()
    else:
        sc4 = ax5.scatter(
            diag_df["last_turn_minus_turn0"],
            diag_df["parallel_median_minus_turn0"],
            c=diag_df["parallel_iqr"],
            cmap="coolwarm",
            s=12,
            alpha=0.75,
        )
        ax5.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.7)
        ax5.axvline(0.0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.7)
        ax5.set_xlabel("sequential quality (last turn - turn0)")
        ax5.set_ylabel("parallel median - sequential turn0")
        ax5.set_title(
            "last_turn-turn0 vs (parallel median - turn0) | color: parallel IQR"
        )
        ax5.grid(alpha=0.25, linewidth=0.7)
        cbar4 = fig.colorbar(sc4, ax=ax5, pad=0.01)
        cbar4.set_label("parallel IQR (q3-q1)")

    ax6 = axes[5]
    if turn01_diff_binned_df.empty:
        ax6.text(
            0.5,
            0.5,
            "no turn0/turn1 string pairs for bin probability plot",
            ha="center",
            va="center",
        )
        ax6.set_axis_off()
    else:
        ax6.plot(
            turn01_diff_binned_df["bin_center"],
            turn01_diff_binned_df["prob_diff"],
            marker="o",
            linewidth=1.2,
            markersize=4.0,
            color="#1f77b4",
        )
        ax6.set_ylim(-0.02, 1.02)
        ax6.set_xlabel("sequential quality (turn0) bin center")
        ax6.set_ylabel("P(turn1 output != turn0 output)")
        ax6.set_title(
            "binned probability of turn1 string differing from turn0 string"
        )
        ax6.grid(alpha=0.25, linewidth=0.7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot: {out_path}")


def _resolve_metric_variant_pairs(
    run_scores: pd.DataFrame,
    metrics: list[str] | None,
    variants: list[str] | None,
) -> list[tuple[str, str]]:
    """Resolve metric/variant pairs for one run after optional filters.

    Args:
        run_scores: Score dataframe for one run.
        metrics: Optional allowed metric names.
        variants: Optional allowed variant names.
    """
    pairs = (
        run_scores[["metric", "variant"]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    out: list[tuple[str, str]] = []
    for metric, variant in pairs:
        metric = str(metric)
        variant = str(variant)
        if metrics is not None and metric not in metrics:
            continue
        if variants is not None and variant not in variants:
            continue
        out.append((metric, variant))
    return out


def _build_prompt_id_index(score_df: pd.DataFrame) -> dict[str, set[str]]:
    """Build run_name -> unique prompt_id set index from score dataframe.

    Args:
        score_df: Score-level dataframe returned by load_all_finished_dataframes.
    """
    prompt_ids_by_run: dict[str, set[str]] = {}
    if score_df.empty:
        return prompt_ids_by_run
    if "run_name" not in score_df.columns or "prompt_id" not in score_df.columns:
        return prompt_ids_by_run

    local_df = score_df[["run_name", "prompt_id"]].dropna(subset=["run_name", "prompt_id"])
    for run_name, group in local_df.groupby("run_name"):
        prompt_ids_by_run[str(run_name)] = {str(pid) for pid in group["prompt_id"].tolist()}
    return prompt_ids_by_run


def _is_prompt_alignment_ok(
    parallel_row: pd.Series,
    sequential_row: pd.Series,
    prompt_ids_by_run: dict[str, set[str]],
) -> bool:
    """Check whether prompt IDs are alignable between two runs.

    Args:
        parallel_row: Parallel run metadata row.
        sequential_row: Sequential run metadata row.
        prompt_ids_by_run: Mapping from run_name to prompt-id set.
    """
    parallel_name = str(parallel_row.get("run_name") or "")
    sequential_name = str(sequential_row.get("run_name") or "")
    par_ids = prompt_ids_by_run.get(parallel_name)
    seq_ids = prompt_ids_by_run.get(sequential_name)

    if not par_ids or not seq_ids:
        print(
            "[MATCH CHECK FAIL] missing prompt ids "
            f"parallel={parallel_name} sequential={sequential_name} "
            f"parallel_count={0 if not par_ids else len(par_ids)} "
            f"sequential_count={0 if not seq_ids else len(seq_ids)}"
        )
        return False

    if par_ids == seq_ids:
        return True

    only_parallel = sorted(par_ids - seq_ids)
    only_sequential = sorted(seq_ids - par_ids)
    preview_parallel = only_parallel[:5]
    preview_sequential = only_sequential[:5]
    print(
        "[MATCH CHECK FAIL] prompt-id mismatch "
        f"parallel={parallel_name} sequential={sequential_name} "
        f"parallel_count={len(par_ids)} sequential_count={len(seq_ids)} "
        f"only_parallel_sample={preview_parallel} "
        f"only_sequential_sample={preview_sequential}"
    )
    return False


def match_parallel_and_sequential_runs(
    runs_df: pd.DataFrame,
    score_df: pd.DataFrame | None = None,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Match each parallel run with sequential runs under same model/lang/dataset.

    Args:
        runs_df: Informative run dataframe from load_all_finished_runs_dataframe.
        score_df: Optional score-level dataframe used for prompt-id alignment checks.
    """
    if runs_df.empty:
        return []

    base_df = (
        runs_df.sort_values("run_index")
        .drop_duplicates(subset=["run_name"], keep="first")
        .copy()
    )

    base_df["lang_key"] = base_df.apply(_lang_key, axis=1)
    base_df["is_parallel"] = base_df.apply(_is_parallel_run, axis=1)

    parallel_df = base_df[base_df["is_parallel"]].copy()
    sequential_df = base_df[~base_df["is_parallel"]].copy()
    prompt_ids_by_run = _build_prompt_id_index(score_df) if score_df is not None else {}

    seq_index: dict[tuple[str, str, str], list[pd.Series]] = {}
    for _, row in sequential_df.iterrows():
        key = (
            str(row.get("model") or ""),
            str(row.get("lang_key") or ""),
            _dataset_key(row),
        )
        seq_index.setdefault(key, []).append(row)

    matches: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for _, row in parallel_df.iterrows():
        key = (
            str(row.get("model") or ""),
            str(row.get("lang_key") or ""),
            _dataset_key(row),
        )
        parallel_info = _to_run_info(row)
        seq_rows = seq_index.get(key, [])
        seq_matches: list[dict[str, Any]] = []
        for seq_row in seq_rows:
            if score_df is not None:
                if not _is_prompt_alignment_ok(
                    parallel_row=row,
                    sequential_row=seq_row,
                    prompt_ids_by_run=prompt_ids_by_run,
                ):
                    continue
            seq_matches.append(_to_run_info(seq_row))
        seq_matches.sort(key=lambda item: int(item.get("run_index", -1)))
        matches.append((parallel_info, seq_matches))

    matches.sort(key=lambda pair: int(pair[0].get("run_index", -1)))
    return matches


def plot_parallel_vs_sequential(
    *,
    runs_df: pd.DataFrame,
    score_df: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    metrics: list[str] | None = None,
    variants: list[str] | None = None,
    parallel_iqr_tail_percent: float = 5.0,
    max_groups: int | None = None,
    max_prompts_print: int = 20,
) -> int:
    """Plot per-prompt parallel quality stats for aligned parallel/sequential matches.

    Args:
        runs_df: Informative run dataframe from load_all_finished_runs_dataframe.
        score_df: Optional score-level dataframe for prompt-id match checks.
        out_dir: Output directory root for generated plots.
        metrics: Optional metric filter list.
        variants: Optional variant filter list.
        parallel_iqr_tail_percent: Percentile size used for low/high grouped samples.
        max_groups: Optional cap on printed groups.
        max_prompts_print: Max prompt rows printed per run+metric+variant (negative => all).
    """
    if score_df is None:
        raise ValueError("score_df is required for parallel-vs-sequential plotting.")
    if out_dir is None:
        raise ValueError("out_dir is required for parallel-vs-sequential plotting.")
    if parallel_iqr_tail_percent <= 0.0 or parallel_iqr_tail_percent > 50.0:
        raise ValueError(
            "parallel_iqr_tail_percent must be in (0, 50]. "
            f"Got {parallel_iqr_tail_percent}."
        )

    matches = match_parallel_and_sequential_runs(runs_df, score_df=score_df)
    if max_groups is not None and max_groups >= 0:
        matches = matches[:max_groups]

    printable: list[list[Any]] = []
    for parallel_info, seq_infos in matches:
        printable.append([parallel_info, seq_infos])

    print("draft_parallel_vs_sequential_matches:")
    pprint(printable, sort_dicts=False, width=140)
    print(f"draft_parallel_count={len(matches)}")
    print(
        "draft_total_sequential_links="
        f"{sum(len(seq_infos) for _, seq_infos in matches)}"
    )

    base_df = (
        runs_df.sort_values("run_index")
        .drop_duplicates(subset=["run_name"], keep="first")
        .copy()
    )
    by_run_name = {str(row["run_name"]): row for _, row in base_df.iterrows()}
    sample_prompt_cache: dict[Path, dict[str, dict[str, Any]]] = {}

    saved = 0
    for parallel_info, seq_infos in matches:
        parallel_name = str(parallel_info.get("run_name") or "")
        run_row = by_run_name.get(parallel_name)
        if run_row is None:
            continue
        if not seq_infos:
            print(
                "[MATCH CHECK FAIL] no aligned sequential runs "
                f"for parallel={parallel_name}"
            )
            continue

        run_scores = score_df[score_df["run_name"] == parallel_name].copy()
        if run_scores.empty:
            print(f"[DRAFT PARALLEL LABELS] no scores for run={parallel_name}")
            continue
        num_generations = _read_num_generations_for_run(run_row)

        pairs = _resolve_metric_variant_pairs(run_scores, metrics=metrics, variants=variants)
        for metric, variant in pairs:
            current = run_scores[
                (run_scores["metric"] == metric) & (run_scores["variant"] == variant)
            ].copy()
            if current.empty:
                continue

            counts = current.groupby("prompt_id")["quality"].size().astype(int)
            bad = counts[counts != num_generations]
            assert bad.empty, (
                "Prompt sample count mismatch for parallel run "
                f"{parallel_name} metric={metric} variant={variant}. "
                f"Expected n_generations={num_generations}; "
                f"bad_counts={bad.head(10).to_dict()}"
            )

            labels_df = _compute_prompt_quality_labels(current)
            labels_df = labels_df.sort_values("iqr", ascending=False).reset_index(drop=True)
            prompt_order = labels_df["prompt_id"].tolist()
            parallel_iqr_values = labels_df["iqr"].astype(float).tolist()
            parallel_median_values = labels_df["median"].astype(float).tolist()
            parallel_sample_file = _sample_file_from_score_rows(current)
            parallel_prompt_index: dict[str, dict[str, Any]] = {}
            if parallel_sample_file is None:
                print(
                    "[MATCH CHECK FAIL] no sample file for parallel run "
                    f"run={parallel_name} metric={metric} variant={variant}"
                )
            elif not parallel_sample_file.exists():
                print(
                    "[MATCH CHECK FAIL] sample file missing for parallel run "
                    f"path={parallel_sample_file}"
                )
            else:
                if parallel_sample_file not in sample_prompt_cache:
                    sample_prompt_cache[parallel_sample_file] = _build_prompt_sample_index(
                        parallel_sample_file
                    )
                parallel_prompt_index = sample_prompt_cache[parallel_sample_file]

            print(
                "[DRAFT PARALLEL LABELS] "
                f"run={parallel_name} metric={metric} variant={variant} "
                f"prompts={len(labels_df)} n_generations={num_generations} "
                f"aligned_sequential={len(seq_infos)}"
            )
            if max_prompts_print < 0:
                to_print = labels_df
            else:
                to_print = labels_df.head(max_prompts_print)
            print(
                to_print[
                    ["prompt_id", "n_samples", "q1", "median", "q3", "iqr"]
                ].to_string(index=False)
            )

            bundle_dir = _parallel_vs_sequential_out_dir(
                out_dir=out_dir,
                parallel_row=run_row,
                metric=metric,
                variant=variant,
            )
            parallel_title = (
                f"parallel={parallel_name} "
                f"| lang={_lang_key(run_row)} "
                f"| dataset={_dataset_key(run_row)} "
                f"| metric={metric} ({variant}) "
                f"| aligned_seq={len(seq_infos)}"
            )
            _plot_prompt_stats_scatter(
                labels_df,
                title=parallel_title,
                out_path=_parallel_bundle_parallel_plot_path(bundle_dir),
            )
            saved += 1

            for seq_info in seq_infos:
                sequential_name = str(seq_info.get("run_name") or "")
                seq_row = by_run_name.get(sequential_name)
                if seq_row is None:
                    print(
                        "[MATCH CHECK FAIL] missing sequential run row "
                        f"parallel={parallel_name} sequential={sequential_name}"
                    )
                    continue

                seq_scores = score_df[score_df["run_name"] == sequential_name].copy()
                if seq_scores.empty:
                    print(
                        "[MATCH CHECK FAIL] no scores for sequential run "
                        f"parallel={parallel_name} sequential={sequential_name}"
                    )
                    continue

                seq_current = seq_scores[
                    (seq_scores["metric"] == metric) & (seq_scores["variant"] == variant)
                ].copy()
                if seq_current.empty:
                    print(
                        "[MATCH CHECK FAIL] no metric/variant scores for sequential run "
                        f"parallel={parallel_name} sequential={sequential_name} "
                        f"metric={metric} variant={variant}"
                    )
                    continue

                seq_labels = _compute_prompt_quality_labels(seq_current)
                seq_aligned = (
                    seq_labels.set_index("prompt_id")
                    .reindex(prompt_order)
                    .reset_index()
                )
                if seq_aligned[["q1", "median", "q3"]].isna().any().any():
                    print(
                        "[MATCH CHECK FAIL] unable to align sequential prompt stats "
                        f"parallel={parallel_name} sequential={sequential_name} "
                        f"metric={metric} variant={variant}"
                    )
                    continue
                if seq_aligned["prompt_id"].tolist() != prompt_order:
                    print(
                        "[MATCH CHECK FAIL] sequential prompt order differs from parallel order "
                        f"parallel={parallel_name} sequential={sequential_name} "
                        f"metric={metric} variant={variant}"
                    )
                    continue
                sequential_iqr_values = seq_aligned["iqr"].astype(float).tolist()
                seq_sample_file = _sample_file_from_score_rows(seq_current)
                seq_prompt_index: dict[str, dict[str, Any]] = {}
                if seq_sample_file is None:
                    print(
                        "[MATCH CHECK FAIL] no sample file for sequential run "
                        f"run={sequential_name} metric={metric} variant={variant}"
                    )
                elif not seq_sample_file.exists():
                    print(
                        "[MATCH CHECK FAIL] sample file missing for sequential run "
                        f"path={seq_sample_file}"
                    )
                else:
                    if seq_sample_file not in sample_prompt_cache:
                        sample_prompt_cache[seq_sample_file] = _build_prompt_sample_index(
                            seq_sample_file
                        )
                    seq_prompt_index = sample_prompt_cache[seq_sample_file]

                seq_title = (
                    f"parallel={parallel_name} "
                    f"| sequential={sequential_name} "
                    f"| seq_scenario={str(seq_row.get('scenario_name') or '')} "
                    f"| lang={_lang_key(run_row)} "
                    f"| dataset={_dataset_key(run_row)} "
                    f"| metric={metric} ({variant})"
                )
                _plot_sequential_prompt_stats_subplots(
                    seq_aligned_df=seq_aligned,
                    seq_scores_df=seq_current,
                    sequential_prompt_index=seq_prompt_index,
                    parallel_labels_df=labels_df,
                    prompt_order=[str(pid) for pid in prompt_order],
                    title=seq_title,
                    out_path=_parallel_bundle_sequential_plot_path(bundle_dir, seq_row),
                    iqr_values=parallel_iqr_values,
                    iqr_label=f"parallel iqr (q3-q1): {parallel_name}",
                    iqr_values_secondary=sequential_iqr_values,
                    iqr_label_secondary=f"sequential iqr (q3-q1): {sequential_name}",
                    baseline_median_values=parallel_median_values,
                )
                saved += 1
                if parallel_prompt_index and seq_prompt_index:
                    _save_parallel_iqr_grouped_samples(
                        bundle_dir=bundle_dir,
                        metric=metric,
                        variant=variant,
                        parallel_row=run_row,
                        sequential_row=seq_row,
                        parallel_labels_df=labels_df,
                        seq_aligned_df=seq_aligned,
                        parallel_prompt_index=parallel_prompt_index,
                        sequential_prompt_index=seq_prompt_index,
                        tail_percent=parallel_iqr_tail_percent,
                    )

    return saved


def main() -> None:
    """Run draft parallel-vs-sequential matching and print groups.

    Args:
        None.
    """
    args = _build_arg_parser().parse_args()
    runs_df, score_df = load_all_finished_dataframes(args.runs_dir)
    if runs_df.empty:
        raise ValueError(f"No run rows found in: {args.runs_dir}")
    metrics = parse_multi_args(args.metric)
    variants = parse_multi_args(args.variant)

    saved = plot_parallel_vs_sequential(
        runs_df=runs_df,
        score_df=score_df,
        out_dir=args.out_dir,
        metrics=metrics if metrics else None,
        variants=variants if variants else None,
        parallel_iqr_tail_percent=args.parallel_iqr_tail_percent,
        max_groups=args.max_groups,
        max_prompts_print=args.max_prompts_print,
    )
    if saved == 0:
        raise ValueError("No parallel-vs-sequential plots were generated.")


if __name__ == "__main__":
    main()
