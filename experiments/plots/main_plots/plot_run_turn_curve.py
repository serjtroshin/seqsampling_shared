from __future__ import annotations

import os
from pathlib import Path

# Avoid matplotlib cache permission issues on shared systems.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import pandas as pd


def _path_component(value: str, fallback: str) -> str:
    """Keep human-readable path components while avoiding nested separators.

    Args:
        value: Raw directory/file component.
        fallback: Fallback string when value is empty.
    """
    cleaned = value.strip()
    if not cleaned:
        cleaned = fallback
    # Keep names readable; only normalize path separators.
    return cleaned.replace("/", "-")


def _run_turn_out_path(
    out_dir: Path,
    run_row: pd.Series,
    metric: str,
    variant: str,
) -> Path:
    """Build output path for one run's turn-curve plot.

    Args:
        out_dir: Root output directory for this plot family.
        run_row: Run-level dataframe row with run metadata.
        metric: Metric name.
        variant: Metric variant.
    """
    dataset_tag = _path_component(str(run_row.get("dataset_tag") or ""), "unknown_dataset")
    lang = _path_component(str(run_row.get("tgt") or run_row.get("lp") or ""), "unknown")
    scenario_name = _path_component(str(run_row.get("scenario_name") or ""), "unknown_scenario")
    metric_name = _path_component(str(metric), "metric")
    if variant != "default":
        metric_name = f"{metric_name}__{_path_component(str(variant), 'variant')}"
    file_name = f"{metric_name}.png"
    return (
        out_dir
        / "turn_curves"
        / dataset_tag
        / lang
        / scenario_name
        / file_name
    )


def _plot_run_turn_curve(
    run_row: pd.Series,
    run_score_df: pd.DataFrame,
    metric: str,
    variant: str,
    sweep_name: str,
    out_path: Path,
) -> bool:
    """Render one run-level mean-quality-vs-turn curve.

    Args:
        run_row: Run-level metadata row (run_index, task_id, tgt, status).
        run_score_df: Score-level rows for a single run+metric+variant.
        metric: Metric name used in title/filename.
        variant: Metric variant used in title/filename.
        sweep_name: Group name shown in the plot title.
        out_path: Target PNG path.
    """
    turn_df = run_score_df.copy()
    turn_df["sequential_id"] = pd.to_numeric(turn_df["sequential_id"], errors="coerce")
    turn_df["quality"] = pd.to_numeric(turn_df["quality"], errors="coerce")
    turn_df = turn_df.dropna(subset=["sequential_id", "quality"])
    if turn_df.empty:
        return False

    turn_df["sequential_id"] = turn_df["sequential_id"].astype(int)
    curve = (
        turn_df.groupby("sequential_id", as_index=False)
        .agg(mean_quality=("quality", "mean"), count=("quality", "size"))
        .sort_values("sequential_id")
    )
    if curve.empty:
        return False

    x = curve["sequential_id"].tolist()
    y = curve["mean_quality"].tolist()
    counts = curve["count"].tolist()

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(x, y, marker="o", linewidth=2.0, color="#1f77b4")
    for xi, yi, cnt in zip(x, y, counts):
        ax.text(xi, yi + 0.002, f"{yi:.3f}\n(n={cnt})", ha="center", va="bottom", fontsize=8)

    task = str(run_row.get("task_id") or f"r{int(run_row['run_index']):03d}")
    tgt = str(run_row.get("tgt") or "unknown")
    status = str(run_row.get("status") or "UNKNOWN")
    ax.set_title(f"{sweep_name} | {task}/{tgt} | {metric} ({variant}) | {status}")
    ax.set_xlabel("sequential_id (turn_id)")
    ax.set_ylabel("mean quality")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_xticks(x)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot: {out_path}")
    return True


def plot_run_turn_curves(
    *,
    base_runs_df: pd.DataFrame,
    score_df: pd.DataFrame,
    metrics: list[str],
    variants: list[str],
    sweep_name: str,
    out_dir: Path,
) -> int:
    """Generate run-level turn curves for all selected metric/variant pairs.

    Args:
        base_runs_df: One row per run with identifying metadata.
        score_df: Score-level dataframe with sequential_id and quality columns.
        metrics: Metric names to include.
        variants: Variant names to include.
        sweep_name: Group label shown in generated titles.
        out_dir: Output directory for this plot function family.
    """
    if score_df.empty:
        print("warn_no_turn_plots: score dataframe is empty.")
        return 0

    saved_turn = 0
    for metric in metrics:
        for variant in variants:
            current_scores = score_df[
                (score_df["metric"] == metric) & (score_df["variant"] == variant)
            ].copy()
            if current_scores.empty:
                continue
            for _, run_row in base_runs_df.iterrows():
                run_name = str(run_row["run_name"])
                run_scores = current_scores[current_scores["run_name"] == run_name].copy()
                if run_scores.empty:
                    continue
                did_save = _plot_run_turn_curve(
                    run_row=run_row,
                    run_score_df=run_scores,
                    metric=metric,
                    variant=variant,
                    sweep_name=sweep_name,
                    out_path=_run_turn_out_path(out_dir, run_row, metric, variant),
                )
                if did_save:
                    saved_turn += 1
    return saved_turn
