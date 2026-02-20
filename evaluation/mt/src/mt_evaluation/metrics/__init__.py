from .base import EvaluationMetric, MetricConfig
from .comet import (
    CometQEConfig,
    CometQEMetric,
    CometReferenceConfig,
    CometReferenceMetric,
    XCometXXLMetric,
)
try:
    from .metricx import MetricX24Config, MetricX24Metric
except ImportError:  # pragma: no cover - optional dependency path
    MetricX24Config = None  # type: ignore[assignment]
    MetricX24Metric = None  # type: ignore[assignment]

__all__ = [
    "EvaluationMetric",
    "MetricConfig",
    "CometQEConfig",
    "CometQEMetric",
    "CometReferenceConfig",
    "CometReferenceMetric",
    "XCometXXLMetric",
]

if MetricX24Config is not None and MetricX24Metric is not None:
    __all__.extend(["MetricX24Config", "MetricX24Metric"])
