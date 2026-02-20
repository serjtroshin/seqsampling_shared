from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .metrics.base import EvaluationMetric
from .metrics.comet import (
    CometQEConfig,
    CometQEMetric,
    CometReferenceConfig,
    CometReferenceMetric,
    XCometXXLMetric,
)
try:
    from .metrics.metricx import MetricX24Config, MetricX24Metric
except ImportError as exc:  # pragma: no cover - optional dependency path
    MetricX24Config = None  # type: ignore[assignment]
    MetricX24Metric = None  # type: ignore[assignment]
    _METRICX_IMPORT_ERROR = exc
else:
    _METRICX_IMPORT_ERROR = None

COMET_REF_MODEL = "Unbabel/wmt22-comet-da"
COMET_KIWI_MODEL = "Unbabel/wmt23-cometkiwi-da-xl"
XCOMET_XXL_MODEL = "Unbabel/XCOMET-XXL"
METRICX24_MODEL = "google/metricx-24-hybrid-xl-v2p6"
METRICX24_TOKENIZER = "google/mt5-xl"


@dataclass
class PipelineConfig:
    metrics: list[EvaluationMetric]
    generation_keys: tuple[str, ...] = ("generations", "solutions")
    include_final_answers: bool = True


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"Expected bool, got {type(value)!r}")


def _build_metric(spec: dict[str, Any], default_batch_size: int = 8) -> EvaluationMetric:
    metric_type = str(spec.get("type", "")).strip()
    if not metric_type:
        raise ValueError("Each metric config entry must include a non-empty 'type'.")

    metric_name = str(spec.get("name") or metric_type).strip()
    batch_size = int(spec.get("batch_size", default_batch_size))
    model_name = str(spec.get("model", "")).strip() or None
    use_gpu = _as_bool(spec.get("use_gpu_if_available"), True)

    if metric_type == "comet_qe":
        config = CometQEConfig(
            name=metric_name,
            batch_size=batch_size,
            model_name=model_name or COMET_KIWI_MODEL,
            use_gpu_if_available=use_gpu,
        )
        return CometQEMetric(config)

    if metric_type == "comet_ref":
        resolved_model = model_name or COMET_REF_MODEL
        config = CometReferenceConfig(
            name=metric_name,
            batch_size=batch_size,
            model_name=resolved_model,
            use_gpu_if_available=use_gpu,
        )
        if metric_name == "xcomet_xxl" or resolved_model == XCOMET_XXL_MODEL:
            return XCometXXLMetric(config)
        return CometReferenceMetric(config)

    if metric_type == "xcomet_xxl":
        config = CometReferenceConfig(
            name=metric_name or "xcomet_xxl",
            batch_size=batch_size,
            model_name=model_name or XCOMET_XXL_MODEL,
            use_gpu_if_available=use_gpu,
        )
        return XCometXXLMetric(config)

    if metric_type == "metricx24":
        if MetricX24Config is None or MetricX24Metric is None:
            raise RuntimeError(
                "MetricX integration is not available in this environment."
            ) from _METRICX_IMPORT_ERROR
        tokenizer_name = str(spec.get("tokenizer", "")).strip() or METRICX24_TOKENIZER
        max_input_length = int(spec.get("max_input_length", 1536))
        qe = _as_bool(spec.get("qe"), False)
        metricx_python = str(spec.get("metricx_python", "evaluation/mt/metricx/.venv/bin/python")).strip()
        metricx_repo_dir = str(spec.get("metricx_repo_dir", "evaluation/mt/metricx")).strip()
        config = MetricX24Config(
            name=metric_name,
            batch_size=batch_size,
            model_name=model_name or METRICX24_MODEL,
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            qe=qe,
            metricx_python=metricx_python,
            metricx_repo_dir=metricx_repo_dir,
        )
        return MetricX24Metric(config)

    raise ValueError(f"Unsupported metric type: {metric_type!r}")


def default_metrics(default_batch_size: int = 8, model_name: Optional[str] = None) -> list[EvaluationMetric]:
    return [
        CometQEMetric(
            CometQEConfig(
                name="comet_kiwi_qe",
                batch_size=default_batch_size,
                model_name=model_name or COMET_KIWI_MODEL,
            )
        )
    ]


def preset_metric(name: str, batch_size: int = 8, model_name: Optional[str] = None) -> EvaluationMetric:
    metric_name = name.strip()
    if not metric_name:
        raise ValueError("Metric preset name cannot be empty.")

    if metric_name == "comet_qe":
        return _build_metric(
            {
                "type": "comet_ref",
                "name": metric_name,
                "batch_size": batch_size,
                "model": model_name or COMET_REF_MODEL,
            },
            default_batch_size=batch_size,
        )

    if metric_name == "comet_kiwi_qe":
        return _build_metric(
            {
                "type": "comet_qe",
                "name": metric_name,
                "batch_size": batch_size,
                "model": model_name or COMET_KIWI_MODEL,
            },
            default_batch_size=batch_size,
        )

    if metric_name == "comet_ref":
        return _build_metric(
            {
                "type": "comet_ref",
                "name": metric_name,
                "batch_size": batch_size,
                "model": model_name or COMET_REF_MODEL,
            },
            default_batch_size=batch_size,
        )

    if metric_name == "xcomet_xxl":
        return _build_metric(
            {
                "type": "comet_ref",
                "name": metric_name,
                "batch_size": batch_size,
                "model": model_name or XCOMET_XXL_MODEL,
            },
            default_batch_size=batch_size,
        )

    if metric_name == "metricx24_ref":
        return _build_metric(
            {
                "type": "metricx24",
                "name": metric_name,
                "batch_size": batch_size,
                "model": model_name or METRICX24_MODEL,
                "tokenizer": METRICX24_TOKENIZER,
                "max_input_length": 1536,
                "qe": False,
            },
            default_batch_size=batch_size,
        )

    if metric_name == "metricx24_qe":
        return _build_metric(
            {
                "type": "metricx24",
                "name": metric_name,
                "batch_size": batch_size,
                "model": model_name or METRICX24_MODEL,
                "tokenizer": METRICX24_TOKENIZER,
                "max_input_length": 1536,
                "qe": True,
            },
            default_batch_size=batch_size,
        )

    raise ValueError(f"Unknown metric preset: {metric_name!r}")


def load_pipeline_config(path: Path) -> PipelineConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics_raw = data.get("metrics")
    if not isinstance(metrics_raw, list) or not metrics_raw:
        raise ValueError("Config must contain a non-empty 'metrics' list.")

    metrics = [_build_metric(item) for item in metrics_raw]

    generation_keys_raw = data.get("generation_keys", ["generations", "solutions"])
    if not isinstance(generation_keys_raw, list) or not generation_keys_raw:
        raise ValueError("'generation_keys' must be a non-empty list of field names.")
    generation_keys = tuple(str(key) for key in generation_keys_raw)

    include_final_answers = _as_bool(data.get("include_final_answers"), True)
    return PipelineConfig(
        metrics=metrics,
        generation_keys=generation_keys,
        include_final_answers=include_final_answers,
    )
