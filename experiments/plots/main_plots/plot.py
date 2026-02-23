from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from experiments.plots.main_plots.plot_parallel_vs_sequential import (
        plot_parallel_vs_sequential,
    )
    from experiments.plots.main_plots.plot_run_turn_curve import plot_run_turn_curves
    from experiments.plots.main_plots.utils import (
        load_all_finished_dataframes,
        parse_multi_args,
    )
except ModuleNotFoundError:
    from plot_parallel_vs_sequential import plot_parallel_vs_sequential
    from plot_run_turn_curve import plot_run_turn_curves
    from utils import load_all_finished_dataframes, parse_multi_args


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the main all-finished-runs plotting entrypoint.

    Args:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Run main plots over outputs/mt/all_finished_runs."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/mt/all_finished_runs"),
        help="Root directory with merged finished runs.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/mt/all_finished_runs__plots"),
        help="All outputs are saved under <out-root>/<plot_function_name>/...",
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
        "--out-dataframe-csv",
        type=Path,
        default=None,
        help="Optional path to save the informative run dataframe as CSV.",
    )
    return parser


def _resolve_metric_variant_filters(
    runs_df: pd.DataFrame,
    selected_metrics: list[str],
    selected_variants: list[str],
) -> tuple[list[str], list[str]]:
    """Validate and resolve requested metric/variant filters.

    Args:
        runs_df: Informative run dataframe returned by loader utilities.
        selected_metrics: User-requested metric names from CLI.
        selected_variants: User-requested variant names from CLI.
    """
    metric_df = runs_df[runs_df["metric"].notna()].copy()
    if metric_df.empty:
        raise ValueError("No evaluation metric rows were discovered in runs_dir.")

    available_metrics = sorted(metric_df["metric"].dropna().astype(str).unique().tolist())
    if selected_metrics:
        missing_metrics = [m for m in selected_metrics if m not in available_metrics]
        if missing_metrics:
            raise ValueError(f"Unknown metric(s): {missing_metrics}. Available: {available_metrics}")
        metrics = selected_metrics
    else:
        metrics = available_metrics

    available_variants = sorted(metric_df["variant"].dropna().astype(str).unique().tolist())
    if selected_variants:
        missing_variants = [v for v in selected_variants if v not in available_variants]
        if missing_variants:
            raise ValueError(f"Unknown variant(s): {missing_variants}. Available: {available_variants}")
        variants = selected_variants
    else:
        variants = available_variants
    return metrics, variants


def main() -> None:
    """Run enabled plot jobs over merged finished runs.

    Args:
        None.
    """
    args = _build_arg_parser().parse_args()
    runs_df, score_df = load_all_finished_dataframes(args.runs_dir)
    if runs_df.empty:
        raise ValueError(f"No run rows found in: {args.runs_dir}")

    if args.out_dataframe_csv is not None:
        args.out_dataframe_csv.parent.mkdir(parents=True, exist_ok=True)
        runs_df.to_csv(args.out_dataframe_csv, index=False)
        print(f"saved_dataframe_csv: {args.out_dataframe_csv}")

    metrics, variants = _resolve_metric_variant_filters(
        runs_df,
        selected_metrics=parse_multi_args(args.metric),
        selected_variants=parse_multi_args(args.variant),
    )
    sweep_name = Path(args.runs_dir).resolve().name

    base_runs_df = (
        runs_df.sort_values("run_index")
        .drop_duplicates(subset=["run_name"], keep="first")
        [
            [
                "run_index",
                "task_id",
                "run_name",
                "status",
                "tgt",
                "lp",
                "scenario_name",
                "dataset_tag",
            ]
        ]
    )

    # Comment out entries in this list to disable specific plot families.
    plot_functions: list[tuple[str, Callable[[Path], int]]] = [
        # (
        #     "plot_run_turn_curve",
        #     lambda out_dir: plot_run_turn_curves(
        #         base_runs_df=base_runs_df,
        #         score_df=score_df,
        #         metrics=metrics,
        #         variants=variants,
        #         sweep_name=sweep_name,
        #         out_dir=out_dir,
        #     ),
        # ),
        (
            "plot_parallel_vs_sequential",
            lambda out_dir: plot_parallel_vs_sequential(
                runs_df=runs_df,
                score_df=score_df,
                out_dir=out_dir,
                metrics=metrics,
                variants=variants,
            ),
        ),
        # ("plot_other", lambda out_dir: plot_other(..., out_dir=out_dir)),
    ]

    total_saved = 0
    for plot_name, plot_fn in plot_functions:
        saved = int(plot_fn(args.out_root / plot_name))
        print(f"plot_job_done: {plot_name} saved={saved}")
        total_saved += saved

    if total_saved == 0:
        raise ValueError("No plots were generated by enabled plot jobs.")


if __name__ == "__main__":
    main()
