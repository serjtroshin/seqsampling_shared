from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from experiments.plots.main_plots.plot_greedy import plot_multiturn
    from experiments.plots.main_plots.plot_parallel_vs_sequential import (
        plot_parallel_vs_sequential,
    )
    from experiments.plots.main_plots.plot_run_turn_curve import plot_run_turn_curves
    from experiments.plots.main_plots.utils import (
        load_all_finished_dataframes,
        parse_multi_args,
    )
except ModuleNotFoundError:
    from plot_greedy import plot_multiturn
    from plot_parallel_vs_sequential import plot_parallel_vs_sequential
    from plot_run_turn_curve import plot_run_turn_curves
    from utils import load_all_finished_dataframes, parse_multi_args


@dataclass
class PlotJob:
    plot_name: str
    out_dir: Path
    metrics: tuple[str, ...]
    variants: tuple[str, ...]
    runs_df: pd.DataFrame | None = None
    base_runs_df: pd.DataFrame | None = None
    score_df: pd.DataFrame | None = None
    sweep_name: str | None = None


def _build_arg_parser() -> argparse.ArgumentParser:
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


def _resolve_multiturn_run_coords(row: pd.Series) -> tuple[str, str] | None:
    source_root = Path("outputs/mt/all_finished_runs").resolve()
    raw_run_dir = row.get("run_dir")
    if isinstance(raw_run_dir, str) and raw_run_dir:
        try:
            rel_parts = Path(raw_run_dir).resolve().relative_to(source_root).parts
        except ValueError:
            rel_parts = ()
        if len(rel_parts) >= 2 and rel_parts[0] and rel_parts[1]:
            return str(rel_parts[0]), str(rel_parts[1])

    model_name = str(row.get("model") or "")
    sampling_profile = str(row.get("sampling_profile") or "")
    if model_name and sampling_profile:
        return model_name, sampling_profile
    return None


def _build_multiturn_plot_job(
    *,
    runs_df: pd.DataFrame,
    base_runs_df: pd.DataFrame,
    score_df: pd.DataFrame,
    out_root: Path,
    metrics: tuple[str, ...],
    variants: tuple[str, ...],
) -> PlotJob | None:
    sweep_coords = _resolve_multiturn_sweep_coords(runs_df)
    if sweep_coords is None:
        return None
    model_name, sampling_profile = sweep_coords

    multiturn_out_root = out_root / model_name / sampling_profile

    return PlotJob(
        plot_name="multiturn_plot",
        out_dir=multiturn_out_root,
        base_runs_df=base_runs_df,
        score_df=score_df,
        metrics=metrics,
        variants=variants,
    )


def _resolve_multiturn_sweep_coords(runs_df: pd.DataFrame) -> tuple[str, str] | None:
    coords = {
        sweep_coords
        for _, row in runs_df.iterrows()
        for sweep_coords in [_resolve_multiturn_run_coords(row)]
        if sweep_coords is not None
    }
    if len(coords) != 1:
        return None
    return next(iter(coords))


def _build_multiturn_plot_jobs(
    *,
    runs_df: pd.DataFrame,
    base_runs_df: pd.DataFrame,
    score_df: pd.DataFrame,
    out_root: Path,
    metrics: tuple[str, ...],
    variants: tuple[str, ...],
) -> list[PlotJob]:
    valid_runs = runs_df[
        runs_df["sampling_profile"].notna()
        & runs_df["run_name"].notna()
    ].copy()
    if valid_runs.empty:
        return []

    valid_runs["sweep_coords"] = valid_runs.apply(_resolve_multiturn_run_coords, axis=1)
    valid_runs = valid_runs[valid_runs["sweep_coords"].notna()].copy()
    if valid_runs.empty:
        return []

    plot_jobs: list[PlotJob] = []
    for sweep_coords, group_df in valid_runs.groupby("sweep_coords", sort=True):
        group_df = group_df.copy()
        run_names = group_df["run_name"].astype(str).drop_duplicates().tolist()
        if not run_names:
            continue
        group_base_runs_df = base_runs_df[base_runs_df["run_name"].isin(run_names)].copy()
        group_score_df = score_df[score_df["run_name"].isin(run_names)].copy()
        if group_base_runs_df.empty or group_score_df.empty:
            continue
        plot_job = _build_multiturn_plot_job(
            runs_df=group_df,
            base_runs_df=group_base_runs_df,
            score_df=group_score_df,
            out_root=out_root,
            metrics=metrics,
            variants=variants,
        )
        if plot_job is not None:
            plot_jobs.append(plot_job)

    return plot_jobs


def _run_plot_job(job: PlotJob) -> tuple[str, int]:
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
    elif job.plot_name == "multiturn_plot":
        if job.base_runs_df is None or job.score_df is None:
            raise ValueError("multiturn_plot job is missing required inputs.")
        saved = plot_multiturn(
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

    # Keep the default main run focused on always-on summaries.
    # plot_parallel_vs_sequential stays available as a standalone script if needed.
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
        # PlotJob(plot_name="plot_other", out_dir=args.out_root / "plot_other", ...),
    ]
    plot_jobs.extend(
        _build_multiturn_plot_jobs(
            runs_df=runs_df,
            base_runs_df=base_runs_df,
            score_df=score_df,
            out_root=args.out_root,
            metrics=metric_tuple,
            variants=variant_tuple,
        )
    )

    total_saved = _run_plot_jobs(plot_jobs, max_workers=args.max_workers)
    if total_saved == 0:
        raise ValueError("No plots were generated by enabled plot jobs.")


if __name__ == "__main__":
    main()
