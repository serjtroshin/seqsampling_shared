from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

# Avoid matplotlib cache permission issues on shared systems.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

try:
    from experiments.plots.utils import load_sweep_dataframe, load_sweep_score_dataframe
except ModuleNotFoundError:
    from utils import load_sweep_dataframe, load_sweep_score_dataframe


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    return slug.strip("_") or "metric"


def _parse_multi_args(raw_values: list[str] | None) -> list[str]:
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


def _run_label(row: pd.Series) -> str:
    parts: list[str] = []
    task_id = row.get("task_id")
    tgt = row.get("tgt")
    if isinstance(task_id, str) and task_id:
        parts.append(task_id)
    if isinstance(tgt, str) and tgt:
        parts.append(tgt)
    if not parts:
        parts.append(str(row.get("run_index", "?")))
    return "/".join(parts)


def _metric_out_path(
    out_dir: Path,
    metric: str,
    variant: str,
) -> Path:
    variant_suffix = "" if variant == "default" else f"_{_slugify(variant)}"
    return out_dir / f"sweep_scores_{_slugify(metric)}{variant_suffix}.png"


def _run_turn_out_path(
    out_dir: Path,
    run_row: pd.Series,
    metric: str,
    variant: str,
) -> Path:
    run_idx = int(run_row["run_index"])
    task = str(run_row.get("task_id") or f"r{run_idx:03d}")
    tgt = str(run_row.get("tgt") or "unknown")
    variant_suffix = "" if variant == "default" else f"_{_slugify(variant)}"
    name = f"turn_quality_run{run_idx:03d}_{_slugify(task)}_{_slugify(tgt)}_{_slugify(metric)}{variant_suffix}.png"
    return out_dir / "turn_curves" / name


def _plot_metric_variant(
    base_runs_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    metric: str,
    variant: str,
    sweep_name: str,
    out_path: Path,
) -> None:
    metric_cols = ["run_name", "average_score"]
    merged = base_runs_df.merge(metric_df[metric_cols], on="run_name", how="left")
    merged = merged.sort_values("run_index").reset_index(drop=True)
    merged["run_label"] = merged.apply(_run_label, axis=1)

    targets = sorted([t for t in merged["tgt"].dropna().unique().tolist() if t])
    palette = list(plt.get_cmap("tab10").colors)
    tgt_color = {tgt: palette[i % len(palette)] for i, tgt in enumerate(targets)}

    default_color = (0.5, 0.5, 0.5)
    edge_color_by_status = {
        "FAILED": "#d62728",
        "DONE": "#222222",
        "SUBMITTED": "#ff7f0e",
        "UNKNOWN": "#666666",
    }

    values: list[float | None] = []
    for raw in merged["average_score"].tolist():
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            values.append(None)
        else:
            values.append(float(raw))

    present_values = [v for v in values if v is not None]
    if not present_values:
        print(f"skip_plot_no_values: metric={metric} variant={variant}")
        return

    y_min = min(present_values)
    y_max = max(present_values)
    span = max(y_max - y_min, 0.02)
    missing_y = y_min - 0.12 * span

    x = list(range(len(merged)))
    bar_heights = [v if v is not None else missing_y for v in values]
    bar_colors = [tgt_color.get(tgt, default_color) for tgt in merged["tgt"].tolist()]
    edge_colors = [
        edge_color_by_status.get(str(status).upper(), "#222222")
        for status in merged["status"].tolist()
    ]
    hatches = ["" if v is not None else "//" for v in values]

    fig_w = max(8.0, 0.85 * len(merged))
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    bars = ax.bar(x, bar_heights, color=bar_colors, edgecolor=edge_colors, linewidth=1.2)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    for i, value in enumerate(values):
        if value is None:
            ax.text(i, missing_y, "NA", ha="center", va="bottom", fontsize=8, color="#333333")
        else:
            ax.text(
                i,
                value + 0.02 * span,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111111",
            )

    ax.set_xticks(x, merged["run_label"].tolist(), rotation=35, ha="right")
    ax.set_ylabel("average_score")
    ax.set_xlabel("run (task_id/tgt)")
    ax.set_title(f"{sweep_name} | {metric} ({variant})")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_ylim(missing_y - 0.06 * span, y_max + 0.20 * span)

    legend_items: list[Patch] = []
    for tgt in targets:
        legend_items.append(Patch(facecolor=tgt_color[tgt], edgecolor="#222222", label=f"tgt={tgt}"))
    legend_items.append(Patch(facecolor="white", edgecolor="#d62728", label="FAILED edge"))
    legend_items.append(Patch(facecolor="white", edgecolor="#222222", hatch="//", label="metric missing"))
    ax.legend(handles=legend_items, loc="best", fontsize=8, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot: {out_path}")


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
    print(f"saved_plot: {out_path}")
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot sweep run results (average_score) from MT evaluation reports."
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Sweep work directory (contains sweep.log and experiment_run_dirs.txt).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for generated plots (default: <sweep-dir>/plots).",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric(s) to plot (repeat or pass comma-separated values). Default: all discovered metrics.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant(s) to plot, e.g. default or FA. Default: all discovered variants.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    df = load_sweep_dataframe(args.sweep_dir)
    score_df = load_sweep_score_dataframe(args.sweep_dir)
    if df.empty:
        raise ValueError(f"No sweep rows found in: {args.sweep_dir}")

    metric_df = df[df["metric"].notna()].copy()
    if metric_df.empty:
        raise ValueError(f"No evaluation metric rows found in: {args.sweep_dir}")

    out_dir = args.out_dir if args.out_dir is not None else args.sweep_dir / "plots"

    selected_metrics = _parse_multi_args(args.metric)
    available_metrics = sorted(metric_df["metric"].dropna().astype(str).unique().tolist())
    if selected_metrics:
        missing_metrics = [m for m in selected_metrics if m not in available_metrics]
        if missing_metrics:
            raise ValueError(f"Unknown metric(s): {missing_metrics}. Available: {available_metrics}")
        metrics = selected_metrics
    else:
        metrics = available_metrics

    selected_variants = _parse_multi_args(args.variant)
    available_variants = sorted(metric_df["variant"].dropna().astype(str).unique().tolist())
    if selected_variants:
        missing_variants = [v for v in selected_variants if v not in available_variants]
        if missing_variants:
            raise ValueError(f"Unknown variant(s): {missing_variants}. Available: {available_variants}")
        variants = selected_variants
    else:
        variants = available_variants

    base_runs_df = (
        df.sort_values("run_index")
        .drop_duplicates(subset=["run_name"], keep="first")
        [
            [
                "run_index",
                "task_id",
                "run_name",
                "status",
                "tgt",
            ]
        ]
    )

    sweep_name = Path(args.sweep_dir).resolve().name
    saved = 0
    saved_turn = 0
    for metric in metrics:
        for variant in variants:
            current = metric_df[
                (metric_df["metric"] == metric) & (metric_df["variant"] == variant)
            ].copy()
            if current.empty:
                continue
            _plot_metric_variant(
                base_runs_df=base_runs_df,
                metric_df=current,
                metric=metric,
                variant=variant,
                sweep_name=sweep_name,
                out_path=_metric_out_path(out_dir, metric, variant),
            )
            saved += 1

            if not score_df.empty:
                current_scores = score_df[
                    (score_df["metric"] == metric) & (score_df["variant"] == variant)
                ].copy()
                if not current_scores.empty:
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

    if saved == 0:
        raise ValueError(
            "No plots were generated with the selected metric/variant filters."
        )
    if saved_turn == 0:
        print("warn_no_turn_plots: no per-run sequential_id curves were generated.")


if __name__ == "__main__":
    main()
