from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from ..records import SampleRecord


@dataclass
class MetricConfig(ABC):
    name: str
    batch_size: int = 8

    @property
    @abstractmethod
    def metric_type(self) -> str:
        raise NotImplementedError


class EvaluationMetric(ABC):
    def __init__(self, config: MetricConfig) -> None:
        self.config = config

    @abstractmethod
    def score(self, records: Sequence[SampleRecord]) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def report_metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def augment_report(self, report: dict[str, Any], records: Sequence[SampleRecord]) -> None:
        """Optionally attach metric-specific details to a report."""
        del report, records

    def cleanup(self) -> None:
        """Optionally release heavy resources (e.g., model weights/GPU memory)."""
