from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from experiments.plots.main_plots.plot_greedy import plot_greedy
    from experiments.plots.main_plots.plot_parallel_vs_sequential import (
        plot_parallel_vs_sequential,
    )
    from experiments.plots.main_plots.plot_run_turn_curve import plot_run_turn_curves
    from experiments.plots.main_plots.utils import (
        load_all_finished_dataframes,
        parse_multi_args,
    )
except ModuleNotFoundError:
    from plot_greedy import plot_greedy
    from plot_parallel_vs_sequential import plot_parallel_vs_sequential
    from plot_run_turn_curve import plot_run_turn_curves
    from utils import load_all_finished_dataframes, parse_multi_args


@dataclass
class PlotJob:
    """Serializable specification for one plot family job."""

    plot_name: str
    out_dir: Path
    metrics: tuple[str, ...]
    variants: tuple[str, ...]
    runs_df: pd.DataFrame | None = None
    base_runs_df: pd.DataFrame | None = None
    score_df: pd.DataFrame | None = None
    sweep_name: str | None = None


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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of plot families to run in parallel. Default: all enabled plot families.",
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


def _run_plot_job(job: PlotJob) -> tuple[str, int]:
    """Execute one plot job in a worker process."""
    if job.plot_name == "plot_run_turn_curve":
        if job.base_runs_df is None or job.score_df is None or job.sweep_name is None:
            raise ValueError("plot_run_turn_curve job is missing required inputs.")
        saved = plot_run_turn_curves(
            base_runs_df=job.base_runs_df,
            score_df=job.score_df,
            metrics=list(job.metrics),
            variants=list(job.variants),
            sweep_name=job.sweep_name,
            out_dir=job.out_dir,
        )
    elif job.plot_name == "plot_parallel_vs_sequential":
        if job.runs_df is None or job.score_df is None:
            raise ValueError("plot_parallel_vs_sequential job is missing required inputs.")
        saved = plot_parallel_vs_sequential(
            runs_df=job.runs_df,
            score_df=job.score_df,
            out_dir=job.out_dir,
            metrics=list(job.metrics),
            variants=list(job.variants),
        )
    elif job.plot_name == "plot_greedy":
        if job.base_runs_df is None or job.score_df is None:
            raise ValueError("plot_greedy job is missing required inputs.")
        saved = plot_greedy(
            base_runs_df=job.base_runs_df,
            score_df=job.score_df,
            metrics=list(job.metrics),
            variants=list(job.variants),
            out_root=job.out_dir,
        )
    else:
        raise ValueError(f"Unknown plot job: {job.plot_name}")

    return job.plot_name, int(saved)


def _run_plot_jobs(plot_jobs: list[PlotJob], max_workers: int | None) -> int:
    """Run configured plot jobs and return the total number of saved plots."""
    if not plot_jobs:
        return 0

    worker_count = len(plot_jobs) if max_workers is None else max_workers
    if worker_count < 1:
        raise ValueError(f"max_workers must be >= 1, got {worker_count}")
    worker_count = min(worker_count, len(plot_jobs))

    if worker_count == 1:
        total_saved = 0
        for job in plot_jobs:
            plot_name, saved = _run_plot_job(job)
            print(f"plot_job_done: {plot_name} saved={saved}")
            total_saved += saved
        return total_saved

    total_saved = 0
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
        future_to_job = {executor.submit(_run_plot_job, job): job for job in plot_jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                plot_name, saved = future.result()
            except Exception as exc:
                raise RuntimeError(f"Plot job failed: {job.plot_name}") from exc
            print(f"plot_job_done: {plot_name} saved={saved}")
            total_saved += saved

    return total_saved


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
    metric_tuple = tuple(metrics)
    variant_tuple = tuple(variants)
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
    plot_jobs: list[PlotJob] = [
        PlotJob(
            plot_name="plot_run_turn_curve",
            out_dir=args.out_root / "plot_run_turn_curve",
            base_runs_df=base_runs_df,
            score_df=score_df,
            metrics=metric_tuple,
            variants=variant_tuple,
            sweep_name=sweep_name,
        ),
        PlotJob(
            plot_name="plot_parallel_vs_sequential",
            out_dir=args.out_root / "plot_parallel_vs_sequential",
            runs_df=runs_df,
            score_df=score_df,
            metrics=metric_tuple,
            variants=variant_tuple,
        ),
        PlotJob(
            plot_name="plot_greedy",
            out_dir=args.out_root / "plot_greedy",
            base_runs_df=base_runs_df,
            score_df=score_df,
            metrics=metric_tuple,
            variants=variant_tuple,
        ),
        # PlotJob(plot_name="plot_other", out_dir=args.out_root / "plot_other", ...),
    ]

    total_saved = _run_plot_jobs(plot_jobs, max_workers=args.max_workers)
    if total_saved == 0:
        raise ValueError("No plots were generated by enabled plot jobs.")


if __name__ == "__main__":
    main()
