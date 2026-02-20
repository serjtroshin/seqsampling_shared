from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Sequence

import torch

try:
    from comet import download_model, load_from_checkpoint
except ImportError as exc:  # pragma: no cover - exercised in runtime environments without COMET
    download_model = None  # type: ignore[assignment]
    load_from_checkpoint = None  # type: ignore[assignment]
    _COMET_IMPORT_ERROR = exc
else:
    _COMET_IMPORT_ERROR = None

from ..records import SampleRecord, as_comet_qe_payload, as_comet_ref_payload
from .base import EvaluationMetric, MetricConfig


@dataclass
class CometQEConfig(MetricConfig):
    model_name: str = "Unbabel/wmt23-cometkiwi-da-xl"
    use_gpu_if_available: bool = True

    @property
    def metric_type(self) -> str:
        return "comet_qe"


@dataclass
class CometReferenceConfig(MetricConfig):
    model_name: str = "Unbabel/wmt22-comet-da"
    use_gpu_if_available: bool = True

    @property
    def metric_type(self) -> str:
        return "comet_ref"


class _CometMetric(EvaluationMetric):
    def __init__(self, config: MetricConfig) -> None:
        super().__init__(config)
        self._model = None
        self._last_prediction: Any = None

    def _load_model(self):
        if download_model is None or load_from_checkpoint is None:
            raise RuntimeError(
                "COMET dependency is not available. Install `unbabel-comet` to use COMET metrics."
            ) from _COMET_IMPORT_ERROR
        if self._model is None:
            checkpoint = download_model(self.config.model_name)  # type: ignore[attr-defined]
            self._model = load_from_checkpoint(checkpoint)
        return self._model

    @staticmethod
    def _extract_field(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        if hasattr(obj, key):
            return getattr(obj, key)
        try:
            return obj[key]
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): _CometMetric._to_serializable(v) for k, v in value.items()}
        if hasattr(value, "items"):
            return {
                str(k): _CometMetric._to_serializable(v)
                for k, v in value.items()  # type: ignore[attr-defined]
            }
        if isinstance(value, (list, tuple)):
            return [_CometMetric._to_serializable(v) for v in value]
        return str(value)

    def _extract_scores(self, outputs: Any) -> list[float]:
        raw_scores = self._extract_field(outputs, "scores")
        if raw_scores is None:
            raise RuntimeError("Unexpected COMET predict output format; expected 'scores'.")

        serializable_scores = self._to_serializable(raw_scores)
        if not isinstance(serializable_scores, list):
            raise RuntimeError("Unexpected COMET scores format; expected a list.")
        return [float(score) for score in serializable_scores]

    def _predict(self, payload: list[dict[str, str]]) -> list[float]:
        model = self._load_model()
        use_gpu = bool(getattr(self.config, "use_gpu_if_available", True))
        gpus = 1 if (use_gpu and torch.cuda.is_available()) else 0
        outputs = model.predict(
            payload,
            batch_size=self.config.batch_size,
            gpus=gpus,
            progress_bar=False,
        )
        self._last_prediction = outputs
        return self._extract_scores(outputs)

    def report_metadata(self) -> dict[str, Any]:
        return {
            "metric": self.config.metric_type,
            "model": self.config.model_name,  # type: ignore[attr-defined]
            "batch_size": self.config.batch_size,
        }

    def cleanup(self) -> None:
        self._last_prediction = None
        if self._model is not None:
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CometQEMetric(_CometMetric):
    def __init__(self, config: CometQEConfig) -> None:
        super().__init__(config)

    def score(self, records: Sequence[SampleRecord]) -> list[float]:
        payload = as_comet_qe_payload(records)
        return self._predict(payload)


class CometReferenceMetric(_CometMetric):
    def __init__(self, config: CometReferenceConfig) -> None:
        super().__init__(config)

    def score(self, records: Sequence[SampleRecord]) -> list[float]:
        payload = as_comet_ref_payload(records)
        return self._predict(payload)


class XCometXXLMetric(CometReferenceMetric):
    def augment_report(self, report: dict[str, Any], records: Sequence[SampleRecord]) -> None:
        del records
        system_score = self._extract_field(self._last_prediction, "system_score")
        if system_score is not None:
            report["system_score"] = float(system_score)

        metadata = self._extract_field(self._last_prediction, "metadata")
        if metadata is None:
            return

        error_spans = self._extract_field(metadata, "error_spans")
        if error_spans is None:
            return

        spans = self._to_serializable(error_spans)
        report["error_spans"] = spans

        scored = report.get("scores")
        if not isinstance(scored, list) or not isinstance(spans, list) or len(scored) != len(spans):
            return
        for score_entry, span_entry in zip(scored, spans):
            if isinstance(score_entry, dict):
                score_entry["error_spans"] = span_entry
