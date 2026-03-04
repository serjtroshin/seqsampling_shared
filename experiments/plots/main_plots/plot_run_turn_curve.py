from __future__ import annotations

import os
from pathlib import Path

# Avoid matplotlib cache permission issues on shared systems.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import pandas as pd

try:
    from experiments.plots.main_plots.utils import metric_path_component, path_component
except ModuleNotFoundError:
    from utils import metric_path_component, path_component


def _run_turn_out_path(
    out_dir: Path,
    run_row: pd.Series,
    metric: str,
    variant: str,
) -> Path:
    dataset_tag = path_component(str(run_row.get("dataset_tag") or ""), "unknown_dataset")
    lang = path_component(str(run_row.get("tgt") or run_row.get("lp") or ""), "unknown")
    scenario_name = path_component(str(run_row.get("scenario_name") or ""), "unknown_scenario")
    metric_name = metric_path_component(str(metric), str(variant))
    file_name = f"{metric_name}.png"
    return out_dir / "turn_curves" / dataset_tag / lang / scenario_name / file_name


def _plot_run_turn_curve(
    run_row: pd.Series,
    run_score_df: pd.DataFrame,
    metric: str,
    variant: str,
    sweep_name: str,
    out_path: Path,
) -> bool:
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
