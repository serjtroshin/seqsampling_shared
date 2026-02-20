from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..records import SampleRecord
from .base import EvaluationMetric, MetricConfig


@dataclass
class MetricX24Config(MetricConfig):
    model_name: str = "google/metricx-24-hybrid-xl-v2p6"
    tokenizer_name: str = "google/mt5-xl"
    max_input_length: int = 1536
    qe: bool = False
    metricx_python: str = "evaluation/mt/metricx/.venv/bin/python"
    metricx_repo_dir: str = "evaluation/mt/metricx"

    @property
    def metric_type(self) -> str:
        return "metricx24"


class MetricX24Metric(EvaluationMetric):
    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[5]

    @classmethod
    def _absolute_from_root(cls, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (cls._project_root() / path).absolute()

    @classmethod
    def _resolve_from_root(cls, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path.resolve()
        return (cls._project_root() / path).resolve()

    def _build_payload(self, records: Sequence[SampleRecord]) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = []
        for record in records:
            item: dict[str, str] = {
                "source": record.src,
                "hypothesis": record.mt,
                "reference": "",
            }
            if not self.config.qe:
                if not record.ref:
                    raise ValueError("MetricX reference-based mode requires references for all records.")
                item["reference"] = record.ref
            payload.append(item)
        return payload

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_predictions(path: Path) -> list[float]:
        scores: list[float] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "prediction" not in row:
                    raise RuntimeError("MetricX output row missing 'prediction' field.")
                scores.append(float(row["prediction"]))
        return scores

    def score(self, records: Sequence[SampleRecord]) -> list[float]:
        # Keep symlink path to preserve virtualenv context.
        metricx_python = self._absolute_from_root(self.config.metricx_python)
        metricx_repo_dir = self._resolve_from_root(self.config.metricx_repo_dir)

        if not metricx_python.exists():
            raise FileNotFoundError(
                f"MetricX python interpreter not found: {metricx_python}. "
                "Create it with: cd evaluation/mt/metricx && uv venv .venv && uv sync"
            )
        if not metricx_repo_dir.exists():
            raise FileNotFoundError(f"MetricX repo directory not found: {metricx_repo_dir}")

        payload = self._build_payload(records)
        with tempfile.TemporaryDirectory(prefix="metricx24_eval_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            input_file = tmp_root / "input.jsonl"
            output_file = tmp_root / "output.jsonl"
            self._write_jsonl(input_file, payload)

            cmd = [
                str(metricx_python),
                "-m",
                "metricx24.predict",
                "--tokenizer",
                self.config.tokenizer_name,
                "--model_name_or_path",
                self.config.model_name,
                "--max_input_length",
                str(self.config.max_input_length),
                "--batch_size",
                str(self.config.batch_size),
                "--input_file",
                str(input_file),
                "--output_file",
                str(output_file),
            ]
            if self.config.qe:
                cmd.append("--qe")

            result = subprocess.run(
                cmd,
                cwd=str(metricx_repo_dir),
                text=True,
                capture_output=True,
            )
            if result.returncode != 0:
                debug_cmd = [
                    str(metricx_python),
                    "-c",
                    (
                        "import sys, importlib.util; "
                        "print('sys.executable=', sys.executable); "
                        "print('sys.prefix=', sys.prefix); "
                        "print('datasets_spec=', importlib.util.find_spec('datasets')); "
                        "print('pyarrow_spec=', importlib.util.find_spec('pyarrow'))"
                    ),
                ]
                debug = subprocess.run(
                    debug_cmd,
                    cwd=str(metricx_repo_dir),
                    text=True,
                    capture_output=True,
                )
                raise RuntimeError(
                    "MetricX inference failed with return code "
                    f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
                    f"debug_stdout:\n{debug.stdout}\ndebug_stderr:\n{debug.stderr}"
                )

            scores = self._read_predictions(output_file)
            if len(scores) != len(records):
                raise RuntimeError(
                    f"MetricX returned {len(scores)} predictions for {len(records)} input records."
                )
            return scores

    def report_metadata(self) -> dict[str, Any]:
        return {
            "metric": self.config.metric_type,
            "model": self.config.model_name,
            "tokenizer": self.config.tokenizer_name,
            "batch_size": self.config.batch_size,
            "max_input_length": self.config.max_input_length,
            "qe": self.config.qe,
            "direction": "lower_is_better",
        }
