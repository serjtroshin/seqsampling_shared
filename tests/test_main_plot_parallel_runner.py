from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.plots.main_plots import plot as plot_module

pgg_module = importlib.import_module("experiments.plots.main_plots.plot_greedy")
pvs_module = importlib.import_module("experiments.plots.main_plots.plot_parallel_vs_sequential")


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame({"run_index": [0], "run_name": ["run-0"]})


def test_run_plot_job_dispatches_to_expected_plot_function(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_turn_curves(**kwargs):
        calls.append(("plot_run_turn_curve", kwargs["out_dir"]))
        return 2

    def fake_parallel_vs_sequential(**kwargs):
        calls.append(("plot_parallel_vs_sequential", kwargs["out_dir"]))
        return 3

    def fake_greedy(**kwargs):
        calls.append(("plot_greedy", kwargs["out_root"]))
        return 4

    monkeypatch.setattr(plot_module, "plot_run_turn_curves", fake_turn_curves)
    monkeypatch.setattr(plot_module, "plot_parallel_vs_sequential", fake_parallel_vs_sequential)
    monkeypatch.setattr(plot_module, "plot_greedy", fake_greedy)

    base_runs_df = _sample_frame()
    runs_df = _sample_frame()
    score_df = _sample_frame()

    jobs = [
        plot_module.PlotJob(
            plot_name="plot_run_turn_curve",
            out_dir=tmp_path / "turn",
            base_runs_df=base_runs_df,
            score_df=score_df,
            metrics=("comet_qe",),
            variants=("default",),
            sweep_name="demo",
        ),
        plot_module.PlotJob(
            plot_name="plot_parallel_vs_sequential",
            out_dir=tmp_path / "parallel",
            runs_df=runs_df,
            score_df=score_df,
            metrics=("comet_qe",),
            variants=("default",),
        ),
        plot_module.PlotJob(
            plot_name="plot_greedy",
            out_dir=tmp_path / "greedy",
            base_runs_df=base_runs_df,
            score_df=score_df,
            metrics=("comet_qe",),
            variants=("default",),
        ),
    ]

    results = [plot_module._run_plot_job(job) for job in jobs]

    assert results == [
        ("plot_run_turn_curve", 2),
        ("plot_parallel_vs_sequential", 3),
        ("plot_greedy", 4),
    ]
    assert calls == [
        ("plot_run_turn_curve", tmp_path / "turn"),
        ("plot_parallel_vs_sequential", tmp_path / "parallel"),
        ("plot_greedy", tmp_path / "greedy"),
    ]


def test_run_plot_jobs_uses_executor_for_parallel_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    executor_config: dict[str, object] = {}

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeExecutor:
        def __init__(self, *, max_workers, mp_context):
            executor_config["max_workers"] = max_workers
            executor_config["mp_context"] = mp_context

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, job):
            calls.append(job.plot_name)
            return FakeFuture(fn(job))

    monkeypatch.setattr(plot_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(plot_module, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        plot_module.multiprocessing,
        "get_context",
        lambda method: f"context:{method}",
    )
    monkeypatch.setattr(
        plot_module,
        "_run_plot_job",
        lambda job: (job.plot_name, len(job.plot_name)),
    )

    plot_jobs = [
        plot_module.PlotJob(
            plot_name="plot_run_turn_curve",
            out_dir=tmp_path / "turn",
            base_runs_df=_sample_frame(),
            score_df=_sample_frame(),
            metrics=("comet_qe",),
            variants=("default",),
            sweep_name="demo",
        ),
        plot_module.PlotJob(
            plot_name="plot_parallel_vs_sequential",
            out_dir=tmp_path / "parallel",
            runs_df=_sample_frame(),
            score_df=_sample_frame(),
            metrics=("comet_qe",),
            variants=("default",),
        ),
    ]

    total_saved = plot_module._run_plot_jobs(plot_jobs, max_workers=8)

    assert total_saved == len("plot_run_turn_curve") + len("plot_parallel_vs_sequential")
    assert calls == ["plot_run_turn_curve", "plot_parallel_vs_sequential"]
    assert executor_config == {
        "max_workers": 2,
        "mp_context": "context:spawn",
    }


def test_match_parallel_and_sequential_runs_requires_matching_sampling_profile() -> None:
    runs_df = pd.DataFrame(
        [
            {
                "run_index": 0,
                "run_name": "parallel-run",
                "scenario_name": "mt_parallel",
                "run_rel_path": "outputs/en-de/mt_parallel/parallel-run",
                "model": "qwen",
                "lp": "en-de",
                "tgt": "de",
                "dataset_tag": "wmt24pp_par",
                "sampling_profile": "temp_0.9.p0.95.k5",
            },
            {
                "run_index": 1,
                "run_name": "sequential-same-sampling",
                "scenario_name": "mt_multi_turn_please_translate_again",
                "run_rel_path": "outputs/en-de/mt_multi_turn/sequential-same-sampling",
                "model": "qwen",
                "lp": "en-de",
                "tgt": "de",
                "dataset_tag": "wmt24pp_par",
                "sampling_profile": "temp_0.9.p0.95.k5",
            },
            {
                "run_index": 2,
                "run_name": "sequential-different-sampling",
                "scenario_name": "mt_multi_turn_please_translate_again",
                "run_rel_path": "outputs/en-de/mt_multi_turn/sequential-different-sampling",
                "model": "qwen",
                "lp": "en-de",
                "tgt": "de",
                "dataset_tag": "wmt24pp_par",
                "sampling_profile": "temp_0.0.p1.0.k1",
            },
        ]
    )

    matches = pvs_module.match_parallel_and_sequential_runs(runs_df)

    assert len(matches) == 1
    parallel_info, sequential_infos = matches[0]
    assert parallel_info["run_name"] == "parallel-run"
    assert [item["run_name"] for item in sequential_infos] == ["sequential-same-sampling"]


def test_plot_greedy_includes_later_turn_pair_binned_probability_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pair_calls: list[tuple[int, int, str]] = []

    monkeypatch.setattr(pgg_module, "_plot_dataset_best_worst", lambda *args, **kwargs: False)
    monkeypatch.setattr(pgg_module, "_plot_run_binned_prob_of_change", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        pgg_module,
        "_plot_run_binned_prob_of_change_for_turn_pair",
        lambda *, start_turn_id, end_turn_id, out_path, **kwargs: pair_calls.append(
            (start_turn_id, end_turn_id, str(out_path))
        )
        or True,
    )
    monkeypatch.setattr(pgg_module, "_plot_run_turn_curve", lambda *args, **kwargs: False)

    base_runs_df = pd.DataFrame(
        [
            {
                "run_index": 0,
                "task_id": None,
                "run_name": "run-0",
                "status": "DONE",
                "tgt": "de",
                "lp": "en-de",
                "scenario_name": "mt_multi_turn_please_translate_again",
                "dataset_tag": "wmt24pp_par",
            }
        ]
    )
    score_df = pd.DataFrame(
        [
            {
                "run_name": "run-0",
                "metric": "comet_qe",
                "variant": "default",
                "sequential_id": 0,
                "prompt_id": "p1",
                "quality": 0.1,
            }
        ]
    )

    saved = pgg_module.plot_greedy(
        base_runs_df=base_runs_df,
        score_df=score_df,
        metrics=["comet_qe"],
        variants=["default"],
        out_root=tmp_path,
    )

    assert saved == 3
    assert [(start, end) for start, end, _ in pair_calls] == [(1, 2), (2, 3), (3, 4)]
    assert any("binned_prob_of_change_from_3_to_4_turn" in out_path for _, _, out_path in pair_calls)
    assert any("binned_prob_of_change_from_4_to_5_turn" in out_path for _, _, out_path in pair_calls)


def test_plot_greedy_splits_reranker_best_worst_by_scenario(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    aggregate_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        pgg_module,
        "_plot_dataset_best_worst",
        lambda current_scores, run_rows, *, dataset_tag, scenario_name, out_path, **kwargs: aggregate_calls.append(
            (scenario_name, str(out_path))
        )
        or True,
    )
    monkeypatch.setattr(pgg_module, "_plot_run_binned_prob_of_change", lambda *args, **kwargs: False)
    monkeypatch.setattr(pgg_module, "_plot_run_binned_prob_of_change_for_turn_pair", lambda *args, **kwargs: False)
    monkeypatch.setattr(pgg_module, "_save_turn_pair_change_summary", lambda *args, **kwargs: False)
    monkeypatch.setattr(pgg_module, "_plot_run_turn_curve", lambda *args, **kwargs: False)
    monkeypatch.setattr(pgg_module, "_plot_probability_of_change_grid", lambda *args, **kwargs: False)
    monkeypatch.setattr(pgg_module, "_plot_probability_of_change_bars", lambda *args, **kwargs: False)
    monkeypatch.setattr(pgg_module, "_plot_turn_curve_bars", lambda *args, **kwargs: False)

    base_runs_df = pd.DataFrame(
        [
            {
                "run_index": 0,
                "task_id": None,
                "run_name": "again-run",
                "status": "DONE",
                "tgt": "ru",
                "lp": "en-ru",
                "scenario_name": "mt_multi_turn_please_translate_again",
                "dataset_tag": "wmt24pp_doc",
            },
            {
                "run_index": 1,
                "task_id": None,
                "run_name": "better-run",
                "status": "DONE",
                "tgt": "ru",
                "lp": "en-ru",
                "scenario_name": "mt_multi_turn_please_translate_again_for_a_better",
                "dataset_tag": "wmt24pp_doc",
            },
        ]
    )
    score_df = pd.DataFrame(
        [
            {"run_name": "again-run", "metric": "comet_qe", "variant": "default", "prompt_id": "p1", "quality": 0.1},
            {"run_name": "better-run", "metric": "comet_qe", "variant": "default", "prompt_id": "p1", "quality": 0.2},
        ]
    )

    saved = pgg_module.plot_greedy(
        base_runs_df=base_runs_df,
        score_df=score_df,
        metrics=["comet_qe"],
        variants=["default"],
        out_root=tmp_path,
    )

    assert saved == 2
    assert [scenario_name for scenario_name, _ in aggregate_calls] == [
        "mt_multi_turn_please_translate_again",
        "mt_multi_turn_please_translate_again_for_a_better",
    ]
    assert any(
        "wmt24pp_doc/mt_multi_turn_please_translate_again/reranker_best_worst__comet_qe.png" in out_path
        for _, out_path in aggregate_calls
    )
    assert any(
        "wmt24pp_doc/mt_multi_turn_please_translate_again_for_a_better/reranker_best_worst__comet_qe.png" in out_path
        for _, out_path in aggregate_calls
    )


def test_plot_greedy_builds_probability_of_change_grid(tmp_path: Path) -> None:
    image = np.zeros((8, 8, 3), dtype=float)
    image[..., 1] = 0.5

    for dataset_tag in pgg_module.GRID_DATASET_TAGS:
        for scenario_name in pgg_module.GRID_SCENARIOS:
            for _, source_dir in pgg_module.GRID_COLUMNS:
                source_path = pgg_module._probability_of_change_source_path(
                    tmp_path,
                    dataset_tag=dataset_tag,
                    lang=pgg_module.GRID_LANG,
                    scenario_name=scenario_name,
                    metric="comet_qe",
                    variant="default",
                    source_dir=source_dir,
                )
                source_path.parent.mkdir(parents=True, exist_ok=True)
                plt.imsave(source_path, image)

        saved = pgg_module._plot_probability_of_change_grid(
            out_root=tmp_path,
            dataset_tag=dataset_tag,
            lang=pgg_module.GRID_LANG,
            scenarios=pgg_module.GRID_SCENARIOS,
            columns=pgg_module.GRID_COLUMNS,
            metric="comet_qe",
            variant="default",
        )

        assert saved is True
        assert pgg_module._probability_of_change_grid_out_path(
            tmp_path,
            dataset_tag,
            pgg_module.GRID_LANG,
            "comet_qe",
            "default",
        ).exists()


def test_save_turn_pair_change_summary_writes_json_next_to_plot(tmp_path: Path) -> None:
    sample_file = tmp_path / "samples.jsonl"
    sample_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_id": "p1",
                        "source": "hello",
                        "solutions": ["a", "b"],
                        "sequential_ids": [0, 1],
                    }
                ),
                json.dumps(
                    {
                        "prompt_id": "p2",
                        "source": "world",
                        "solutions": ["x", "x"],
                        "sequential_ids": [0, 1],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_score_df = pd.DataFrame(
        [
            {"prompt_id": "p1", "sequential_id": 0, "quality": 0.1, "sample_file": str(sample_file)},
            {"prompt_id": "p1", "sequential_id": 1, "quality": 0.2, "sample_file": str(sample_file)},
            {"prompt_id": "p2", "sequential_id": 0, "quality": 0.3, "sample_file": str(sample_file)},
            {"prompt_id": "p2", "sequential_id": 1, "quality": 0.4, "sample_file": str(sample_file)},
        ]
    )
    run_row = pd.Series(
        {
            "dataset_tag": "wmt24pp_doc",
            "tgt": "ru",
            "scenario_name": "mt_multi_turn_please_translate_again",
        }
    )
    out_path = tmp_path / "binned_prob_of_change" / "plot.png"

    saved = pgg_module._save_turn_pair_change_summary(
        run_row=run_row,
        run_score_df=run_score_df,
        metric="comet_qe",
        variant="default",
        out_path=out_path,
        start_turn_id=0,
        end_turn_id=1,
    )

    assert saved is True
    payload = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["turn_transition"] == "0->1"
    assert payload["changed_count"] == 1
    assert payload["total_count"] == 2
    assert payload["probability_change"] == pytest.approx(0.5)


def test_plot_probability_of_change_bars_reads_saved_jsons(tmp_path: Path) -> None:
    for scenario_name in pgg_module.GRID_SCENARIOS:
        for column_idx, (_, source_dir) in enumerate(pgg_module.GRID_COLUMNS):
            plot_path = pgg_module._probability_of_change_source_path(
                tmp_path,
                dataset_tag="wmt24pp_doc",
                lang=pgg_module.GRID_LANG,
                scenario_name=scenario_name,
                metric="comet_qe",
                variant="default",
                source_dir=source_dir,
            )
            json_path = plot_path.with_suffix(".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(
                    {
                        "scenario_name": scenario_name,
                        "probability_change": 0.1 + 0.1 * column_idx,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    saved = pgg_module._plot_probability_of_change_bars(
        out_root=tmp_path,
        dataset_tag="wmt24pp_doc",
        lang=pgg_module.GRID_LANG,
        scenarios=pgg_module.GRID_SCENARIOS,
        columns=pgg_module.GRID_COLUMNS,
        metric="comet_qe",
        variant="default",
    )

    assert saved is True
    assert pgg_module._probability_of_change_bar_out_path(
        tmp_path,
        "wmt24pp_doc",
        pgg_module.GRID_LANG,
        "comet_qe",
        "default",
    ).exists()


def test_plot_turn_curve_bars_writes_aggregate_chart(tmp_path: Path) -> None:
    selected_runs_df = pd.DataFrame(
        [
            {
                "run_name": "again-run",
                "dataset_tag": "wmt24pp_doc",
                "tgt": "ru",
                "scenario_name": "mt_multi_turn_please_translate_again",
            },
            {
                "run_name": "better-run",
                "dataset_tag": "wmt24pp_doc",
                "tgt": "ru",
                "scenario_name": "mt_multi_turn_please_translate_again_for_a_better",
            },
            {
                "run_name": "decide-run",
                "dataset_tag": "wmt24pp_doc",
                "tgt": "ru",
                "scenario_name": "mt_multi_turn_please_translate_again_for_a_better_deside_if_change",
            },
        ]
    )
    current_scores = pd.DataFrame(
        [
            {"run_name": "again-run", "sequential_id": 0, "quality": 0.2},
            {"run_name": "again-run", "sequential_id": 1, "quality": 0.3},
            {"run_name": "better-run", "sequential_id": 0, "quality": 0.4},
            {"run_name": "better-run", "sequential_id": 1, "quality": 0.5},
            {"run_name": "decide-run", "sequential_id": 0, "quality": 0.6},
            {"run_name": "decide-run", "sequential_id": 1, "quality": 0.7},
        ]
    )

    saved = pgg_module._plot_turn_curve_bars(
        out_root=tmp_path,
        selected_runs_df=selected_runs_df,
        current_scores=current_scores,
        dataset_tag="wmt24pp_doc",
        lang="ru",
        scenarios=pgg_module.GRID_SCENARIOS,
        metric="comet_qe",
        variant="default",
    )

    assert saved is True
    assert pgg_module._turn_curve_bar_out_path(
        tmp_path,
        "wmt24pp_doc",
        "ru",
        "comet_qe",
        "default",
    ).exists()
