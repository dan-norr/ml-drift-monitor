"""
domain/entities.py — Core business objects.

Pure dataclasses with no external dependencies.  These are the nouns of the
system: what a training run produces, what a weekly monitoring cycle records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlertType(str, Enum):
    DATA_DRIFT = "DATA_DRIFT"
    DATASET_DRIFT = "DATASET_DRIFT"


@dataclass(frozen=True)
class DriftAlert:
    alert_type: AlertType
    message: str


@dataclass(frozen=True)
class ClassificationResult:
    f1: float
    precision: float
    recall: float
    accuracy: float = 0.0


@dataclass(frozen=True)
class DatasetDriftResult:
    share_drifted_features: float
    n_drifted_features: int
    dataset_drift: bool


@dataclass(frozen=True)
class WeeklyMetrics:
    week: int
    classification: ClassificationResult
    dataset_drift: DatasetDriftResult
    feature_drift_scores: dict[str, float]
    alerts: list[DriftAlert] = field(default_factory=list)

    @property
    def has_alert(self) -> bool:
        return len(self.alerts) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "week": self.week,
            "classification": {
                "f1": self.classification.f1,
                "precision": self.classification.precision,
                "recall": self.classification.recall,
                "accuracy": self.classification.accuracy,
            },
            "dataset_drift": {
                "share_drifted_features": self.dataset_drift.share_drifted_features,
                "n_drifted_features": self.dataset_drift.n_drifted_features,
                "dataset_drift": self.dataset_drift.dataset_drift,
            },
            "feature_drift_scores": self.feature_drift_scores,
            "alerts": [a.message for a in self.alerts],
            "has_alert": self.has_alert,
        }


@dataclass(frozen=True)
class TrainingResult:
    f1: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    threshold: float
