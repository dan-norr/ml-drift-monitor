"""
use_cases/monitor_drift.py — Run drift analysis on all weekly batches.

Iterates over every simulated batch, adds model predictions, delegates
drift analysis to IDriftAnalyser, and persists results via IMetricsRepository.
All framework-specific code lives in the adapter layer.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..domain.ports import IDataRepository, IDriftAnalyser, IMetricsRepository, IModelRepository

logger = logging.getLogger(__name__)

_NON_FEATURE_COLS = frozenset({"target", "prediction", "prediction_proba", "week"})


class MonitorDriftUseCase:
    def __init__(
        self,
        data_repo: IDataRepository,
        model_repo: IModelRepository,
        metrics_repo: IMetricsRepository,
        drift_analyser: IDriftAnalyser,
        cfg: dict[str, Any],
    ) -> None:
        self._data = data_repo
        self._model = model_repo
        self._metrics = metrics_repo
        self._analyser = drift_analyser
        self._cfg = cfg

    def execute(self) -> None:
        reference = self._data.load_reference()
        model = self._model.load_model()
        thresholds = self._cfg["monitoring"]["alert_thresholds"]

        feature_cols = [c for c in reference.columns if c not in _NON_FEATURE_COLS]
        ref_aligned = reference.drop(columns=["week"], errors="ignore")

        batch_paths = self._data.batch_paths()
        if not batch_paths:
            logger.error("No batch files found. Run `make simulate` first.")
            return

        alert_weeks: list[int] = []

        for batch_path in batch_paths:
            week = int(batch_path.stem.replace("week_", ""))
            logger.info("Processing week %d …", week)

            batch = self._data.load_batch(week)
            if "prediction" not in batch.columns:
                batch = self._predict(batch, model, feature_cols)
            batch_aligned = batch.drop(columns=["week"], errors="ignore")

            weekly_metrics, html = self._analyser.analyse(
                ref_aligned, batch_aligned, week, thresholds
            )
            self._metrics.save(weekly_metrics)
            self._metrics.save_report_html(html, week)

            if weekly_metrics.has_alert:
                alert_weeks.append(week)
                for alert in weekly_metrics.alerts:
                    logger.warning("ALERT week %d: %s", week, alert.message)
            else:
                logger.info(
                    "Week %d OK — F1=%.4f, share_drifted=%.1f%%",
                    week,
                    weekly_metrics.classification.f1,
                    weekly_metrics.dataset_drift.share_drifted_features * 100,
                )

        logger.info(
            "Monitoring complete. %d/%d weeks with alerts: %s",
            len(alert_weeks), len(batch_paths), alert_weeks or "none",
        )

    @staticmethod
    def _predict(batch: pd.DataFrame, model: Any, feature_cols: list[str]) -> pd.DataFrame:
        result = batch.copy()
        X = result[feature_cols]
        result["prediction"] = model.predict(X)
        result["prediction_proba"] = model.predict_proba(X)[:, 1]
        return result
