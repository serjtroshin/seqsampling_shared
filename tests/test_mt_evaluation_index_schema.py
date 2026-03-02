from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.plots.main_plots.utils import load_all_finished_dataframes
from experiments.plots.utils import load_sweep_score_dataframe
from experiments.plots.index_compat import (
    build_response_to_turn_map,
    build_turn_parallel_to_response_map,
    resolve_score_item_indices,
)


EVAL_SRC = ROOT / "evaluation" / "mt" / "src"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))

from mt_evaluation.metrics.base import EvaluationMetric, MetricConfig
from mt_evaluation.pipeline import _build_report
from mt_evaluation.records import SampleRecord, extract_records_with_ids


@dataclass
class DummyMetricConfig(MetricConfig):
    @property
    def metric_type(self) -> str:
        return "dummy"


class DummyMetric(EvaluationMetric):
    def score(self, records: list[SampleRecord]) -> list[float]:
        return [0.0 for _ in records]

    def report_metadata(self) -> dict[str, str]:
        return {"metric_type": "dummy"}


def _create_run_layout(root_dir: Path) -> tuple[Path, Path]:
    run_dir = (
        root_dir
        / "qwen3-32b-instruct"
        / "temp_0.0.p1.0.k1"
        / "en-de"
        / "mt_multi_turn_please_translate_again"
        / "20260302-140903_mt-greedy_000"
    )
    generation_dir = run_dir / "generation" / "mt_multi_turn_please_translate_again"
    evaluation_dir = run_dir / "evaluation"
    generation_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)
    (run_dir / "slurm_gen.out").write_text("", encoding="utf-8")
    (run_dir / "slurm_eval.out").write_text("", encoding="utf-8")
    (generation_dir / "scenario_resolved.yaml").write_text(
        "\n".join(
            [
                "name: mt_multi_turn_please_translate_again",
                "sampling_mode: multi_turn",
                "src: en",
                "tgt: de",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sample_file = generation_dir / "samples.jsonl"
    sample_file.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "source": "hello",
                "solutions": ["a", "b", "c"],
                "parallel_ids": [0, 1, 0],
                "sequential_ids": [0, 0, 1],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir, sample_file


def _write_report(report_file: Path, sample_file: Path, score_item: dict[str, object]) -> None:
    report_file.write_text(
        json.dumps(
            {
                "file": str(sample_file),
                "n": 1,
                "average_score": 0.7,
                "scores": [score_item],
            }
        ),
        encoding="utf-8",
    )


def _assert_loaded_score_row(score_df, *, expected_parallel_idx, expected_sequential_id: int, expected_response_idx: int) -> None:
    assert len(score_df) == 1
    row = score_df.iloc[0]
    assert row["prompt_id"] == "p1"
    if expected_parallel_idx is None:
        assert pd.isna(row["parallel_idx"])
    else:
        assert row["parallel_idx"] == expected_parallel_idx
    assert row["sequential_id"] == expected_sequential_id
    assert row["response_idx"] == expected_response_idx


def test_extract_records_with_ids_reads_parallel_and_sequential_indices(tmp_path: Path) -> None:
    sample_file = tmp_path / "samples.jsonl"
    sample_file.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "source": "hello",
                "solutions": ["a", "b", "c"],
                "parallel_ids": [0, 1, 0],
                "sequential_ids": [0, 0, 1],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = extract_records_with_ids(sample_file, generation_keys=("solutions",))

    assert [(record.response_idx, record.parallel_idx, record.sequential_idx) for record in records] == [
        (0, 0, 0),
        (1, 1, 0),
        (2, 0, 1),
    ]


def test_build_report_writes_new_index_schema_when_available() -> None:
    metric = DummyMetric(DummyMetricConfig(name="dummy"))
    record = SampleRecord(
        prompt_id="p1",
        response_idx=7,
        parallel_idx=3,
        sequential_idx=2,
        src="src",
        mt="mt",
    )

    report = _build_report(metric, Path("samples.jsonl"), [record], [0.75])

    assert report["scores"] == [
        {
            "prompt_id": "p1",
            "parallel_idx": 3,
            "sequential_idx": 2,
            "score": 0.75,
        }
    ]


def test_build_report_keeps_legacy_response_idx_fallback() -> None:
    metric = DummyMetric(DummyMetricConfig(name="dummy"))
    record = SampleRecord(
        prompt_id="p1",
        response_idx=4,
        parallel_idx=None,
        sequential_idx=None,
        src="src",
        mt="mt",
    )

    report = _build_report(metric, Path("samples.jsonl"), [record], [0.25])

    assert report["scores"] == [
        {
            "prompt_id": "p1",
            "response_idx": 4,
            "score": 0.25,
        }
    ]


def test_index_compat_resolves_new_and_legacy_score_entries() -> None:
    rows = [
        {
            "prompt_id": "p1",
            "solutions": ["a", "b", "c"],
            "parallel_ids": [0, 1, 0],
            "sequential_ids": [0, 0, 1],
        }
    ]
    scenario_meta = {"sampling_mode": "multi_turn", "name": "mt_multi_turn"}
    turn_map = build_response_to_turn_map(rows, scenario_meta)
    response_map = build_turn_parallel_to_response_map(rows, scenario_meta)

    new_style = resolve_score_item_indices(
        {"prompt_id": "p1", "parallel_idx": 0, "sequential_idx": 1, "score": 0.9},
        turn_map=turn_map,
        response_map=response_map,
        scenario_meta=scenario_meta,
    )
    assert new_style is not None
    assert (new_style.prompt_id, new_style.response_idx, new_style.parallel_idx, new_style.sequential_id) == (
        "p1",
        2,
        0,
        1,
    )

    legacy = resolve_score_item_indices(
        {"prompt_id": "p1", "response_idx": 1, "score": 0.6},
        turn_map=turn_map,
        response_map=response_map,
        scenario_meta=scenario_meta,
    )
    assert legacy is not None
    assert (legacy.prompt_id, legacy.response_idx, legacy.parallel_idx, legacy.sequential_id) == (
        "p1",
        1,
        None,
        0,
    )


def test_load_all_finished_dataframes_supports_new_score_index_format(tmp_path: Path) -> None:
    root_dir = tmp_path / "all_finished"
    run_dir, sample_file = _create_run_layout(root_dir)
    _write_report(
        run_dir / "evaluation" / "comet_qe.json",
        sample_file,
        {
            "prompt_id": "p1",
            "parallel_idx": 0,
            "sequential_idx": 1,
            "score": 0.7,
        },
    )

    _, score_df = load_all_finished_dataframes(root_dir)
    _assert_loaded_score_row(
        score_df,
        expected_parallel_idx=0,
        expected_sequential_id=1,
        expected_response_idx=2,
    )


def test_load_all_finished_dataframes_supports_legacy_score_index_format(tmp_path: Path) -> None:
    root_dir = tmp_path / "all_finished"
    run_dir, sample_file = _create_run_layout(root_dir)
    _write_report(
        run_dir / "evaluation" / "comet_qe.json",
        sample_file,
        {
            "prompt_id": "p1",
            "response_idx": 2,
            "score": 0.7,
        },
    )

    _, score_df = load_all_finished_dataframes(root_dir)
    _assert_loaded_score_row(
        score_df,
        expected_parallel_idx=None,
        expected_sequential_id=1,
        expected_response_idx=2,
    )


def test_load_sweep_score_dataframe_supports_new_score_index_format(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    run_dir, sample_file = _create_run_layout(sweep_dir / "runs")
    _write_report(
        run_dir / "evaluation" / "comet_qe.json",
        sample_file,
        {
            "prompt_id": "p1",
            "parallel_idx": 0,
            "sequential_idx": 1,
            "score": 0.7,
        },
    )
    (sweep_dir / "experiment_run_dirs.txt").write_text(f"{run_dir}\n", encoding="utf-8")
    (sweep_dir / "sweep.log").write_text("", encoding="utf-8")

    score_df = load_sweep_score_dataframe(sweep_dir)
    _assert_loaded_score_row(
        score_df,
        expected_parallel_idx=0,
        expected_sequential_id=1,
        expected_response_idx=2,
    )


def test_load_sweep_score_dataframe_supports_legacy_score_index_format(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    run_dir, sample_file = _create_run_layout(sweep_dir / "runs")
    _write_report(
        run_dir / "evaluation" / "comet_qe.json",
        sample_file,
        {
            "prompt_id": "p1",
            "response_idx": 2,
            "score": 0.7,
        },
    )
    (sweep_dir / "experiment_run_dirs.txt").write_text(f"{run_dir}\n", encoding="utf-8")
    (sweep_dir / "sweep.log").write_text("", encoding="utf-8")

    score_df = load_sweep_score_dataframe(sweep_dir)
    _assert_loaded_score_row(
        score_df,
        expected_parallel_idx=None,
        expected_sequential_id=1,
        expected_response_idx=2,
    )
