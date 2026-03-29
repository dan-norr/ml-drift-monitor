"""adapters/model_repo.py — Pickle-backed model repository."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from ..domain.entities import TrainingResult


class PickleModelRepository:
    def __init__(self, model_path: str, feature_importance_path: str) -> None:
        self._model_path = Path(model_path)
        self._fi_path = Path(feature_importance_path)

    def save(
        self,
        model: Any,
        feature_importance: dict[str, float],
        metrics: TrainingResult,
    ) -> None:
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as fh:
            pickle.dump(model, fh)

        self._fi_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_importance": feature_importance,
            "train_metrics": {
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc,
                "pr_auc": metrics.pr_auc,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "threshold": metrics.threshold,
            },
        }
        with open(self._fi_path, "w") as fh:
            json.dump(payload, fh, indent=2)

    def load_model(self) -> Any:
        with open(self._model_path, "rb") as fh:
            return pickle.load(fh)

    def load_threshold(self) -> float:
        with open(self._fi_path) as fh:
            data = json.load(fh)
        return float(data["train_metrics"]["threshold"])
