"""
domain/ports.py — Abstract I/O boundaries (Dependency Inversion).

Protocol classes define what the use cases need without knowing *how* it is
done.  Adapters implement these protocols; use cases depend only on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .entities import TrainingResult, WeeklyMetrics


class IConfigRepository(Protocol):
    def load(self) -> dict[str, Any]: ...


class IDataRepository(Protocol):
    def load_raw(self, path: str) -> pd.DataFrame: ...
    def save_reference(self, df: pd.DataFrame) -> None: ...
    def load_reference(self) -> pd.DataFrame: ...
    def save_batch(self, df: pd.DataFrame, week: int) -> None: ...
    def load_batch(self, week: int) -> pd.DataFrame: ...
    def batch_paths(self) -> list[Path]: ...


class IModelRepository(Protocol):
    def save(self, model: Any, feature_importance: dict[str, float], metrics: TrainingResult) -> None: ...
    def load_model(self) -> Any: ...
    def load_threshold(self) -> float: ...


class IMetricsRepository(Protocol):
    def save(self, metrics: WeeklyMetrics) -> None: ...
    def save_report_html(self, html: str, week: int) -> None: ...
    def load_all(self) -> list[dict[str, Any]]: ...
    def load_report_html(self, week: int) -> str: ...
    def report_exists(self, week: int) -> bool: ...


class IDriftAnalyser(Protocol):
    """Analyses a weekly batch against the reference and returns metrics + HTML report."""

    def analyse(
        self,
        reference: pd.DataFrame,
        batch: pd.DataFrame,
        week: int,
        thresholds: dict[str, float],
    ) -> tuple[WeeklyMetrics, str]: ...
