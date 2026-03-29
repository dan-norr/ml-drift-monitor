"""
adapters/drift_analyser.py — Evidently-backed drift analyser.

Implements IDriftAnalyser by wrapping the Evidently legacy API.
The use case layer has no knowledge of Evidently — it only sees the protocol.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
from evidently.legacy.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

from ..domain.entities import (
    AlertType,
    ClassificationResult,
    DatasetDriftResult,
    DriftAlert,
    WeeklyMetrics,
)

_NON_FEATURE_COLS = frozenset({"target", "prediction", "prediction_proba", "week"})


class EvidentlyDriftAnalyser:
    def analyse(
        self,
        reference: pd.DataFrame,
        batch: pd.DataFrame,
        week: int,
        thresholds: dict[str, float],
    ) -> tuple[WeeklyMetrics, str]:
        col_map = self._build_column_mapping(reference)
        report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
        report.run(reference_data=reference, current_data=batch, column_mapping=col_map)

        metrics = self._extract(report, week, thresholds)
        html = self._get_html(report)
        return metrics, html

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_column_mapping(reference: pd.DataFrame) -> ColumnMapping:
        feature_cols = [c for c in reference.columns if c not in _NON_FEATURE_COLS]
        return ColumnMapping(
            target="target",
            prediction="prediction",
            numerical_features=feature_cols,
        )

    @staticmethod
    def _get_html(report: Report) -> str:
        """Extract HTML string from an Evidently report via a temp file."""
        fd, tmp_path = tempfile.mkstemp(suffix=".html")
        os.close(fd)
        try:
            report.save_html(tmp_path)
            with open(tmp_path, encoding="utf-8") as fh:
                return fh.read()
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _extract(
        report: Report, week: int, thresholds: dict[str, float]
    ) -> WeeklyMetrics:
        metrics_raw = report.as_dict().get("metrics", [])

        feature_psi: dict[str, float] = {}
        dataset_drift = DatasetDriftResult(0.0, 0, False)
        classification = ClassificationResult(0.0, 0.0, 0.0, 0.0)

        for m in metrics_raw:
            metric_id: str = m.get("metric", "")
            result = m.get("result", {})

            if "DataDriftTable" in metric_id or metric_id == "DataDriftTable":
                for feat, info in result.get("drift_by_columns", {}).items():
                    feature_psi[feat] = float(info.get("drift_score", 0.0))

            if "DatasetDriftMetric" in metric_id or metric_id == "DatasetDriftMetric":
                dataset_drift = DatasetDriftResult(
                    share_drifted_features=float(result.get("share_of_drifted_columns", 0.0)),
                    n_drifted_features=int(result.get("number_of_drifted_columns", 0)),
                    dataset_drift=bool(result.get("dataset_drift", False)),
                )

            if "ClassificationQualityMetric" in metric_id:
                cur = result.get("current", {})
                classification = ClassificationResult(
                    f1=float(cur.get("f1", 0.0)),
                    precision=float(cur.get("precision", 0.0)),
                    recall=float(cur.get("recall", 0.0)),
                    accuracy=float(cur.get("accuracy", 0.0)),
                )

        alerts: list[DriftAlert] = []
        psi_threshold = float(thresholds["psi"])
        for feat, score in feature_psi.items():
            if score > psi_threshold:
                alerts.append(DriftAlert(
                    alert_type=AlertType.DATA_DRIFT,
                    message=f"DATA_DRIFT: {feat} drift_score={score:.4f} > {psi_threshold}",
                ))

        if dataset_drift.share_drifted_features > float(thresholds["share_drifted"]):
            alerts.append(DriftAlert(
                alert_type=AlertType.DATASET_DRIFT,
                message=(
                    f"DATASET_DRIFT: {dataset_drift.share_drifted_features:.1%} features drifted"
                    f" > {thresholds['share_drifted']:.1%} threshold"
                ),
            ))

        return WeeklyMetrics(
            week=week,
            classification=classification,
            dataset_drift=dataset_drift,
            feature_drift_scores=feature_psi,
            alerts=alerts,
        )
