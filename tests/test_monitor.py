"""
Unit tests for EvidentlyDriftAnalyser and JSONMetricsRepository.

Tests the metric extraction and alert logic without requiring Evidently
or real data — mocked report dicts are injected directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.adapters.drift_analyser import EvidentlyDriftAnalyser
from src.adapters.metrics_repo import JSONMetricsRepository
from src.domain.entities import (
    AlertType,
    ClassificationResult,
    DatasetDriftResult,
    WeeklyMetrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_reference() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        "V4":               rng.normal(0, 1, n),
        "V11":              rng.normal(0, 1, n),
        "Amount":           rng.exponential(50, n),
        "target":           rng.integers(0, 2, n),
        "prediction":       rng.integers(0, 2, n),
        "prediction_proba": rng.uniform(0, 1, n),
    })


@pytest.fixture
def thresholds() -> dict[str, float]:
    return {"psi": 0.2, "f1_drop": 0.1, "share_drifted": 0.3}


@pytest.fixture
def report_dict_no_drift() -> dict[str, Any]:
    return {
        "metrics": [
            {
                "metric": "DataDriftTable",
                "result": {
                    "drift_by_columns": {
                        "V4":     {"drift_score": 0.05},
                        "V11":    {"drift_score": 0.08},
                        "Amount": {"drift_score": 0.10},
                    }
                },
            },
            {
                "metric": "DatasetDriftMetric",
                "result": {
                    "share_of_drifted_columns": 0.0,
                    "dataset_drift": False,
                    "number_of_drifted_columns": 0,
                },
            },
            {
                "metric": "ClassificationQualityMetric",
                "result": {
                    "current": {
                        "f1": 0.91, "precision": 0.89,
                        "recall": 0.93, "accuracy": 0.995,
                    }
                },
            },
        ]
    }


@pytest.fixture
def report_dict_with_drift() -> dict[str, Any]:
    return {
        "metrics": [
            {
                "metric": "DataDriftTable",
                "result": {
                    "drift_by_columns": {
                        "V4":     {"drift_score": 0.35},
                        "V11":    {"drift_score": 0.08},
                        "Amount": {"drift_score": 0.42},
                    }
                },
            },
            {
                "metric": "DatasetDriftMetric",
                "result": {
                    "share_of_drifted_columns": 0.45,
                    "dataset_drift": True,
                    "number_of_drifted_columns": 2,
                },
            },
            {
                "metric": "ClassificationQualityMetric",
                "result": {
                    "current": {
                        "f1": 0.72, "precision": 0.75,
                        "recall": 0.69, "accuracy": 0.991,
                    }
                },
            },
        ]
    }


# ---------------------------------------------------------------------------
# _build_column_mapping
# ---------------------------------------------------------------------------


def test_build_column_mapping_sets_target_and_prediction(sample_reference: pd.DataFrame) -> None:
    analyser = EvidentlyDriftAnalyser()
    mapping = analyser._build_column_mapping(sample_reference)
    assert mapping.target == "target"
    assert mapping.prediction == "prediction"


def test_build_column_mapping_excludes_metadata(sample_reference: pd.DataFrame) -> None:
    analyser = EvidentlyDriftAnalyser()
    mapping = analyser._build_column_mapping(sample_reference)
    excluded = {"target", "prediction", "prediction_proba", "week"}
    if mapping.numerical_features:
        for col in excluded:
            assert col not in mapping.numerical_features


# ---------------------------------------------------------------------------
# _extract — no drift
# ---------------------------------------------------------------------------


def test_extract_no_alerts(report_dict_no_drift: dict[str, Any], thresholds: dict[str, float]) -> None:
    mock_report = MagicMock()
    mock_report.as_dict.return_value = report_dict_no_drift

    result = EvidentlyDriftAnalyser._extract(mock_report, week=1, thresholds=thresholds)

    assert result.week == 1
    assert not result.has_alert
    assert len(result.alerts) == 0
    assert abs(result.classification.f1 - 0.91) < 1e-6


def test_extract_classification_values(report_dict_no_drift: dict[str, Any], thresholds: dict[str, float]) -> None:
    mock_report = MagicMock()
    mock_report.as_dict.return_value = report_dict_no_drift

    result = EvidentlyDriftAnalyser._extract(mock_report, week=2, thresholds=thresholds)

    assert abs(result.classification.precision - 0.89) < 1e-6
    assert abs(result.classification.recall - 0.93) < 1e-6


# ---------------------------------------------------------------------------
# _extract — with drift
# ---------------------------------------------------------------------------


def test_extract_detects_feature_alerts(report_dict_with_drift: dict[str, Any], thresholds: dict[str, float]) -> None:
    mock_report = MagicMock()
    mock_report.as_dict.return_value = report_dict_with_drift

    result = EvidentlyDriftAnalyser._extract(mock_report, week=9, thresholds=thresholds)

    assert result.has_alert
    messages = " ".join(a.message for a in result.alerts)
    assert "V4" in messages
    assert "Amount" in messages


def test_extract_detects_dataset_drift_alert(report_dict_with_drift: dict[str, Any], thresholds: dict[str, float]) -> None:
    mock_report = MagicMock()
    mock_report.as_dict.return_value = report_dict_with_drift

    result = EvidentlyDriftAnalyser._extract(mock_report, week=10, thresholds=thresholds)

    types = [a.alert_type for a in result.alerts]
    assert AlertType.DATASET_DRIFT in types


# ---------------------------------------------------------------------------
# JSONMetricsRepository
# ---------------------------------------------------------------------------


def test_save_metrics_writes_valid_json(tmp_path: Path) -> None:
    repo = JSONMetricsRepository(
        metrics_dir=str(tmp_path / "metrics"),
        reports_dir=str(tmp_path / "reports"),
    )
    metrics = WeeklyMetrics(
        week=1,
        classification=ClassificationResult(f1=0.90, precision=0.88, recall=0.92),
        dataset_drift=DatasetDriftResult(share_drifted_features=0.0, n_drifted_features=0, dataset_drift=False),
        feature_drift_scores={"V4": 0.05},
        alerts=[],
    )
    repo.save(metrics)

    out = tmp_path / "metrics" / "week_01.json"
    assert out.exists()
    with open(out) as fh:
        loaded = json.load(fh)
    assert loaded["week"] == 1
    assert abs(loaded["classification"]["f1"] - 0.90) < 1e-6


def test_save_metrics_creates_parent_dirs(tmp_path: Path) -> None:
    repo = JSONMetricsRepository(
        metrics_dir=str(tmp_path / "deep" / "nested"),
        reports_dir=str(tmp_path / "reports"),
    )
    metrics = WeeklyMetrics(
        week=5,
        classification=ClassificationResult(f1=0.0, precision=0.0, recall=0.0),
        dataset_drift=DatasetDriftResult(0.0, 0, False),
        feature_drift_scores={},
    )
    repo.save(metrics)
    assert (tmp_path / "deep" / "nested" / "week_05.json").exists()
