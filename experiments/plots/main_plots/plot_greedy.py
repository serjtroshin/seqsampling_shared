from __future__ import annotations

import argparse
import os
from pathlib import Path

# Avoid matplotlib cache permission issues on shared systems.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from experiments.plots.main_plots.plot_parallel_vs_sequential import (
        _build_prompt_sample_index,
        _build_turn01_diff_binned_dataframe,
        _sample_file_from_score_rows,
        _sample_turn_text_map,
        _safe_int,
    )
    from experiments.plots.main_plots.plot_run_turn_curve import _plot_run_turn_curve
    from experiments.plots.main_plots.utils import (
        load_all_finished_dataframes,
        parse_multi_args,
    )
except ModuleNotFoundError:
    from plot_parallel_vs_sequential import (
        _build_prompt_sample_index,
        _build_turn01_diff_binned_dataframe,
        _sample_file_from_score_rows,
        _sample_turn_text_map,
        _safe_int,
    )
    from plot_run_turn_curve import _plot_run_turn_curve
    from utils import load_all_finished_dataframes, parse_multi_args


DEFAULT_RUNS_DIR = Path("outputs/mt/all_finished_runs/qwen3-32b-instruct/temp_0.0.p1.0.k1")
DEFAULT_OUT_ROOT = Path("outputs/mt/all_finished_runs__plots/plot_greedy")


def _path_component(value: str, fallback: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        cleaned = fallback
    return cleaned.replace("/", "-")


def _sort_prompt_ids(prompt_ids: list[str]) -> list[str]:
    def key(prompt_id: str) -> tuple[int, int | str]:
        try:
            return (0, int(prompt_id))
        except ValueError:
            return (1, prompt_id)

    return sorted(prompt_ids, key=key)


def _select_latest_runs(base_runs_df: pd.DataFrame) -> pd.DataFrame:
    selected = (
        base_runs_df.sort_values("run_index")
        .drop_duplicates(subset=["dataset_tag", "tgt", "scenario_name"], keep="last")
        .copy()
    )
    return selected.sort_values(["dataset_tag", "tgt", "run_index"]).reset_index(drop=True)


def _aggregated_out_path(out_root: Path, dataset_tag: str, metric: str, variant: str) -> Path:
    metric_name = _path_component(metric, "metric")
    if variant != "default":
        metric_name = f"{metric_name}__{_path_component(variant, 'variant')}"
    return out_root / dataset_tag / f"reranker_best_worst__{metric_name}.png"


def _turn_curve_out_path(
    out_root: Path,
    run_row: pd.Series,
    metric: str,
    variant: str,
) -> Path:
    dataset_tag = _path_component(str(run_row.get("dataset_tag") or ""), "unknown_dataset")
    lang = _path_component(str(run_row.get("tgt") or run_row.get("lp") or ""), "unknown_lang")
    scenario = _path_component(str(run_row.get("scenario_name") or ""), "unknown_scenario")
    metric_name = _path_component(metric, "metric")
    if variant != "default":
        metric_name = f"{metric_name}__{_path_component(variant, 'variant')}"
    return out_root / "turn_curves" / dataset_tag / lang / scenario / f"{metric_name}.png"


def _binned_prob_out_path(
    out_root: Path,
    run_row: pd.Series,
    metric: str,
    variant: str,
) -> Path:
    dataset_tag = _path_component(str(run_row.get("dataset_tag") or ""), "unknown_dataset")
    lang = _path_component(str(run_row.get("tgt") or run_row.get("lp") or ""), "unknown_lang")
    scenario = _path_component(str(run_row.get("scenario_name") or ""), "unknown_scenario")
    metric_name = _path_component(metric, "metric")
    if variant != "default":
        metric_name = f"{metric_name}__{_path_component(variant, 'variant')}"
    return out_root / "binned_prob_of_change" / dataset_tag / lang / scenario / f"{metric_name}.png"


def _binned_prob_23_out_path(
    out_root: Path,
    run_row: pd.Series,
    metric: str,
    variant: str,
) -> Path:
    dataset_tag = _path_component(str(run_row.get("dataset_tag") or ""), "unknown_dataset")
    lang = _path_component(str(run_row.get("tgt") or run_row.get("lp") or ""), "unknown_lang")
    scenario = _path_component(str(run_row.get("scenario_name") or ""), "unknown_scenario")
    metric_name = _path_component(metric, "metric")
    if variant != "default":
        metric_name = f"{metric_name}__{_path_component(variant, 'variant')}"
    return (
        out_root
        / "binned_prob_of_change_from_2_to_3_turn"
        / dataset_tag
        / lang
        / scenario
        / f"{metric_name}.png"
    )


def _build_greedy_turn_pair_diag_dataframe(
    run_scores: pd.DataFrame,
    *,
    start_turn_id: int,
    end_turn_id: int,
) -> pd.DataFrame:
    if run_scores.empty:
        return pd.DataFrame()

    turn_col = "sequential_id" if "sequential_id" in run_scores.columns else "response_idx"
    local = run_scores.copy()
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
    if start_turn_id not in pivot.columns or end_turn_id not in pivot.columns:
        return pd.DataFrame()

    diag_df = (
        pivot[[start_turn_id, end_turn_id]]
        .reset_index()
        .rename(
            columns={
                start_turn_id: "start_turn_quality",
                end_turn_id: "end_turn_quality",
            }
        )
    )
    diag_df["prompt_id"] = diag_df["prompt_id"].astype(str)
    return diag_df.dropna(subset=["start_turn_quality", "end_turn_quality"]).reset_index(drop=True)


def _build_turn_pair_diff_binned_dataframe(
    *,
    diag_df: pd.DataFrame,
    prompt_index: dict[str, dict[str, object]],
    start_turn_id: int,
    end_turn_id: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    if diag_df.empty or not prompt_index:
        return pd.DataFrame()

    start_quality_map = (
        diag_df.assign(prompt_id=diag_df["prompt_id"].astype(str))
        .set_index("prompt_id")["start_turn_quality"]
        .astype(float)
        .to_dict()
    )

    rows: list[dict[str, float | str]] = []
    for prompt_id, prompt_payload in prompt_index.items():
        if prompt_id not in start_quality_map:
            continue
        turn_map = _sample_turn_text_map(prompt_payload)
        if start_turn_id not in turn_map or end_turn_id not in turn_map:
            continue
        start_text = turn_map[start_turn_id].strip()
        end_text = turn_map[end_turn_id].strip()
        rows.append(
            {
                "prompt_id": str(prompt_id),
                "start_turn_quality": float(start_quality_map[prompt_id]),
                "changed": 1.0 if end_text != start_text else 0.0,
            }
        )

    diff_df = pd.DataFrame(rows)
    if diff_df.empty:
        return pd.DataFrame()

    unique_start = int(diff_df["start_turn_quality"].nunique())
    if unique_start <= 1:
        return pd.DataFrame(
            [
                {
                    "bin_center": float(diff_df["start_turn_quality"].iloc[0]),
                    "prob_diff": float(diff_df["changed"].mean()),
                    "count": int(len(diff_df)),
                    "bin_label": "all",
                }
            ]
        )

    bins = max(2, min(int(n_bins), unique_start))
    diff_df["bin"] = pd.qcut(
        diff_df["start_turn_quality"],
        q=bins,
        duplicates="drop",
    )
    grouped = (
        diff_df.dropna(subset=["bin"])
        .groupby("bin", as_index=False, observed=False)
        .agg(
            prob_diff=("changed", "mean"),
            count=("changed", "size"),
        )
    )
    if grouped.empty:
        return pd.DataFrame()

    grouped["bin_center"] = grouped["bin"].apply(
        lambda interval: float((interval.left + interval.right) / 2.0)
    )
    grouped["bin_label"] = grouped["bin"].astype(str)
    return grouped.sort_values("bin_center").reset_index(drop=True)


def _plot_run_binned_prob_of_change(
    *,
    run_row: pd.Series,
    run_score_df: pd.DataFrame,
    metric: str,
    variant: str,
    out_path: Path,
) -> bool:
    sample_file = _sample_file_from_score_rows(run_score_df)
    if sample_file is None or not sample_file.exists():
        return False

    diag_df = _build_greedy_turn_pair_diag_dataframe(
        run_score_df,
        start_turn_id=0,
        end_turn_id=1,
    )
    if diag_df.empty:
        return False

    diag_df = diag_df.rename(columns={"start_turn_quality": "turn0_quality"})
    prompt_index = _build_prompt_sample_index(sample_file)
    turn01_diff_binned_df = _build_turn01_diff_binned_dataframe(
        diag_df=diag_df,
        sequential_prompt_index=prompt_index,
        n_bins=10,
    )
    if turn01_diff_binned_df.empty:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        turn01_diff_binned_df["bin_center"],
        turn01_diff_binned_df["prob_diff"],
        marker="o",
        linewidth=1.4,
        markersize=4.4,
        color="#1f77b4",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("greedy quality (turn0) bin center")
    ax.set_ylabel("P(turn1 output != turn0 output)")
    ax.set_title(
        "binned probability of turn1 string differing from turn0 string\n"
        f"{run_row.get('dataset_tag', 'unknown_dataset')} | "
        f"{run_row.get('tgt') or run_row.get('lp') or 'unknown_lang'} | "
        f"{run_row.get('scenario_name', 'unknown_scenario')} | "
        f"{metric} ({variant})"
    )
    ax.grid(alpha=0.25, linewidth=0.7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot: {out_path}")
    return True


def _plot_run_binned_prob_of_change_from_2_to_3_turn(
    *,
    run_row: pd.Series,
    run_score_df: pd.DataFrame,
    metric: str,
    variant: str,
    out_path: Path,
) -> bool:
    sample_file = _sample_file_from_score_rows(run_score_df)
    if sample_file is None or not sample_file.exists():
        return False

    diag_df = _build_greedy_turn_pair_diag_dataframe(
        run_score_df,
        start_turn_id=1,
        end_turn_id=2,
    )
    if diag_df.empty:
        return False

    prompt_index = _build_prompt_sample_index(sample_file)
    turn23_diff_binned_df = _build_turn_pair_diff_binned_dataframe(
        diag_df=diag_df,
        prompt_index=prompt_index,
        start_turn_id=1,
        end_turn_id=2,
        n_bins=10,
    )
    if turn23_diff_binned_df.empty:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        turn23_diff_binned_df["bin_center"],
        turn23_diff_binned_df["prob_diff"],
        marker="o",
        linewidth=1.4,
        markersize=4.4,
        color="#2c7fb8",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("greedy quality (turn2) bin center")
    ax.set_ylabel("P(turn3 output != turn2 output)")
    ax.set_title(
        "binned probability of turn3 string differing from turn2 string\n"
        f"{run_row.get('dataset_tag', 'unknown_dataset')} | "
        f"{run_row.get('tgt') or run_row.get('lp') or 'unknown_lang'} | "
        f"{run_row.get('scenario_name', 'unknown_scenario')} | "
        f"{metric} ({variant})"
    )
    ax.grid(alpha=0.25, linewidth=0.7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot: {out_path}")
    return True


def _plot_dataset_best_worst(
    current_scores: pd.DataFrame,
    run_rows: pd.DataFrame,
    *,
    dataset_tag: str,
    metric: str,
    variant: str,
    out_path: Path,
) -> bool:
    if current_scores.empty or run_rows.empty:
        return False

    prompt_summary = (
        current_scores.groupby(["tgt", "prompt_id"], as_index=False)
        .agg(best_quality=("quality", "max"), worst_quality=("quality", "min"))
        .copy()
    )
    if prompt_summary.empty:
        return False
    prompt_summary["gap"] = prompt_summary["best_quality"] - prompt_summary["worst_quality"]

    langs = [str(item) for item in run_rows["tgt"].dropna().astype(str).tolist()]
    if not langs:
        return False

    turn_summary = (
        current_scores[current_scores["sequential_id"].isin([0, 1, 4])]
        .groupby(["tgt", "sequential_id"], as_index=False)["quality"]
        .mean()
        .copy()
    )
    turn_label_map = {0: "turn1", 1: "turn2", 4: "turn5"}
    turn_summary["kind"] = turn_summary["sequential_id"].map(turn_label_map)

    aggregate = (
        pd.concat(
            [
                prompt_summary.groupby("tgt", as_index=False)["worst_quality"]
                .mean()
                .rename(columns={"worst_quality": "quality"})
                .assign(kind="worst"),
                turn_summary[["tgt", "kind", "quality"]],
                prompt_summary.groupby("tgt", as_index=False)["best_quality"]
                .mean()
                .rename(columns={"best_quality": "quality"})
                .assign(kind="best"),
            ],
            ignore_index=True,
        )
        .copy()
    )
    kind_order = ["worst", "turn1", "turn2", "turn5", "best"]
    aggregate["kind"] = pd.Categorical(aggregate["kind"], categories=kind_order, ordered=True)
    aggregate = aggregate.sort_values(["kind", "tgt"]).reset_index(drop=True)

    fig_height = 4.2 + 2.8 * len(langs)
    fig, axes = plt.subplots(
        nrows=1 + len(langs),
        ncols=1,
        figsize=(16.0, fig_height),
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    bar_ax = axes[0]
    x = np.arange(len(kind_order), dtype=float)
    group_width = 0.82
    lang_width = group_width / max(len(langs), 1)
    bar_width = lang_width / 1.35
    lang_colors = {
        lang: color
        for lang, color in zip(langs, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"])
    }

    for lang_idx, lang in enumerate(langs):
        lang_df = (
            aggregate[aggregate["tgt"] == lang][["kind", "quality"]]
            .set_index("kind")
            .reindex(kind_order)
        )
        offset = -group_width / 2 + lang_idx * lang_width + lang_width / 2
        bar_ax.bar(
            x + offset,
            lang_df["quality"].to_numpy(dtype=float),
            width=bar_width,
            label=lang,
            color=lang_colors[lang],
            alpha=0.88,
        )

    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(kind_order)
    bar_ax.set_ylabel("mean quality")
    bar_ax.set_title(
        f"greedy reranker summary | {dataset_tag} | {metric} ({variant})\n"
        "average over prompts"
    )
    bar_ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    bar_ax.legend(ncol=max(1, min(3, len(langs))), fontsize=9, frameon=False, loc="upper right")

    for ax_idx, lang in enumerate(langs, start=1):
        ax = axes[ax_idx]
        lang_df = prompt_summary[prompt_summary["tgt"] == lang].copy()
        if lang_df.empty:
            ax.set_visible(False)
            continue
        lang_df["prompt_id"] = lang_df["prompt_id"].astype(str)
        lang_df = lang_df.sort_values(["gap", "prompt_id"], ascending=[False, True]).reset_index(drop=True)
        prompt_rank = np.arange(len(lang_df), dtype=float)

        ax.plot(
            prompt_rank,
            lang_df["worst_quality"].to_numpy(dtype=float),
            linewidth=1.7,
            color="#d95f02",
            label="worst",
        )
        ax.plot(
            prompt_rank,
            lang_df["best_quality"].to_numpy(dtype=float),
            linewidth=1.7,
            color="#1b9e77",
            label="best",
        )
        ax.fill_between(
            prompt_rank,
            lang_df["worst_quality"].to_numpy(dtype=float),
            lang_df["best_quality"].to_numpy(dtype=float),
            color="#9ecae1",
            alpha=0.22,
        )

        tick_step = max(1, len(lang_df) // 24)
        tick_positions = prompt_rank[::tick_step]
        tick_labels = lang_df["prompt_id"].iloc[::tick_step].tolist()
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_xlabel("prompt_id (sorted by best-worst gap)")
        ax.set_ylabel("quality")
        ax.set_title(f"{lang} | prompt-level best/worst")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.legend(fontsize=8, frameon=False, loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot: {out_path}")
    return True


def plot_greedy(
    *,
    base_runs_df: pd.DataFrame,
    score_df: pd.DataFrame,
    metrics: list[str],
    variants: list[str],
    out_root: Path,
) -> int:
    selected_runs_df = _select_latest_runs(base_runs_df)
    if selected_runs_df.empty or score_df.empty:
        return 0

    saved = 0

    for metric in metrics:
        for variant in variants:
            current_scores = score_df[
                (score_df["metric"] == metric) & (score_df["variant"] == variant)
            ].copy()
            if current_scores.empty:
                continue

            for dataset_tag, dataset_runs in selected_runs_df.groupby("dataset_tag", sort=True):
                run_names = dataset_runs["run_name"].astype(str).tolist()
                dataset_scores = current_scores[current_scores["run_name"].isin(run_names)].copy()
                if dataset_scores.empty:
                    continue
                if _plot_dataset_best_worst(
                    dataset_scores,
                    dataset_runs,
                    dataset_tag=str(dataset_tag),
                    metric=metric,
                    variant=variant,
                    out_path=_aggregated_out_path(out_root, str(dataset_tag), metric, variant),
                ):
                    saved += 1

            for _, run_row in selected_runs_df.iterrows():
                run_name = str(run_row["run_name"])
                run_scores = current_scores[current_scores["run_name"] == run_name].copy()
                if run_scores.empty:
                    continue
                if _plot_run_binned_prob_of_change(
                    run_row=run_row,
                    run_score_df=run_scores,
                    metric=metric,
                    variant=variant,
                    out_path=_binned_prob_out_path(out_root, run_row, metric, variant),
                ):
                    saved += 1
                if _plot_run_binned_prob_of_change_from_2_to_3_turn(
                    run_row=run_row,
                    run_score_df=run_scores,
                    metric=metric,
                    variant=variant,
                    out_path=_binned_prob_23_out_path(out_root, run_row, metric, variant),
                ):
                    saved += 1
                if _plot_run_turn_curve(
                    run_row=run_row,
                    run_score_df=run_scores,
                    metric=metric,
                    variant=variant,
                    sweep_name="greedy",
                    out_path=_turn_curve_out_path(out_root, run_row, metric, variant),
                ):
                    saved += 1

    return saved


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Greedy multi-turn plots over all finished greedy runs.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Root directory with finished greedy runs.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Output root for greedy plots.",
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
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    runs_df, score_df = load_all_finished_dataframes(args.runs_dir)
    if runs_df.empty:
        raise ValueError(f"No runs discovered in: {args.runs_dir}")

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

    metrics = parse_multi_args(args.metric)
    variants = parse_multi_args(args.variant)
    if not metrics:
        metrics = sorted(score_df["metric"].dropna().astype(str).unique().tolist())
    if not variants:
        variants = sorted(score_df["variant"].dropna().astype(str).unique().tolist())

    saved = plot_greedy(
        base_runs_df=base_runs_df,
        score_df=score_df,
        metrics=metrics,
        variants=variants,
        out_root=args.out_root,
    )
    if saved == 0:
        raise ValueError("No greedy plots were generated.")
    print(f"plot_job_done: plot_greedy saved={saved}")


if __name__ == "__main__":
    main()
