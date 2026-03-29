"""
Unit tests for SimulateDriftUseCase (drift logic).

Tests focus on correctness of drift transformations without requiring
the full dataset or model artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.use_cases.simulate_drift import SimulateDriftUseCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_batch() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "V4":    rng.normal(0, 1, n),
        "V11":   rng.normal(0, 1, n),
        "Amount": rng.exponential(50, n),
        "V1":    rng.normal(0, 1, n),
        "target": rng.integers(0, 2, n),
    })


@pytest.fixture
def schedule() -> dict:
    return {
        "subtle_weeks":        [1, 2, 3],
        "subtle_intensity":    0.3,
        "detectable_weeks":    [4, 5, 6, 7],
        "detectable_intensity": 0.7,
        "critical_weeks":      [8, 9, 10, 11, 12],
        "critical_intensity":  1.5,
    }


# ---------------------------------------------------------------------------
# _drift_intensity
# ---------------------------------------------------------------------------


def test_drift_intensity_subtle(schedule: dict) -> None:
    assert SimulateDriftUseCase._drift_intensity(1, schedule) == 0.3
    assert SimulateDriftUseCase._drift_intensity(3, schedule) == 0.3


def test_drift_intensity_detectable(schedule: dict) -> None:
    assert SimulateDriftUseCase._drift_intensity(4, schedule) == 0.7
    assert SimulateDriftUseCase._drift_intensity(7, schedule) == 0.7


def test_drift_intensity_critical(schedule: dict) -> None:
    assert SimulateDriftUseCase._drift_intensity(8, schedule) == 1.5
    assert SimulateDriftUseCase._drift_intensity(12, schedule) == 1.5


def test_drift_intensity_zero_for_unknown_week(schedule: dict) -> None:
    assert SimulateDriftUseCase._drift_intensity(0, schedule) == 0.0
    assert SimulateDriftUseCase._drift_intensity(13, schedule) == 0.0


# ---------------------------------------------------------------------------
# _apply_data_drift
# ---------------------------------------------------------------------------


def test_apply_data_drift_changes_mean(sample_batch: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    original_mean = sample_batch["V4"].mean()
    drifted = SimulateDriftUseCase._apply_data_drift(
        sample_batch, ["V4", "V11", "Amount"], intensity=2.0, rng=rng
    )
    assert abs(drifted["V4"].mean() - original_mean) > 0.01


def test_apply_data_drift_non_target_features_unchanged(sample_batch: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    original_v1 = sample_batch["V1"].copy()
    drifted = SimulateDriftUseCase._apply_data_drift(
        sample_batch, ["V4", "V11"], intensity=2.0, rng=rng
    )
    pd.testing.assert_series_equal(drifted["V1"], original_v1)


def test_apply_data_drift_returns_copy(sample_batch: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    original_v4 = sample_batch["V4"].copy()
    _ = SimulateDriftUseCase._apply_data_drift(sample_batch, ["V4"], intensity=5.0, rng=rng)
    pd.testing.assert_series_equal(sample_batch["V4"], original_v4)


def test_apply_data_drift_zero_intensity_no_change(sample_batch: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    original_mean = sample_batch["V4"].mean()
    drifted = SimulateDriftUseCase._apply_data_drift(
        sample_batch, ["V4"], intensity=0.0, rng=rng
    )
    assert abs(drifted["V4"].mean() - original_mean) < 0.5


# ---------------------------------------------------------------------------
# _apply_concept_drift
# ---------------------------------------------------------------------------


def test_concept_drift_before_start_unchanged(sample_batch: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    original = sample_batch["target"].copy()
    result = SimulateDriftUseCase._apply_concept_drift(
        sample_batch, week=3, start=8, n_weeks=12, rng=rng
    )
    pd.testing.assert_series_equal(result["target"], original)


def test_concept_drift_after_start_reduces_fraud(sample_batch: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    original_fraud = sample_batch["target"].sum()
    result = SimulateDriftUseCase._apply_concept_drift(
        sample_batch, week=12, start=8, n_weeks=12, rng=rng
    )
    assert result["target"].sum() < original_fraud


def test_concept_drift_returns_copy(sample_batch: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    original = sample_batch["target"].copy()
    _ = SimulateDriftUseCase._apply_concept_drift(
        sample_batch, week=12, start=8, n_weeks=12, rng=rng
    )
    pd.testing.assert_series_equal(sample_batch["target"], original)


def test_concept_drift_increases_with_week(sample_batch: pd.DataFrame) -> None:
    big = sample_batch.loc[sample_batch.index.repeat(10)].reset_index(drop=True)
    rng_factory = lambda seed: np.random.default_rng(seed)  # noqa: E731

    result_w8  = SimulateDriftUseCase._apply_concept_drift(big, week=8,  start=8, n_weeks=12, rng=rng_factory(1))
    result_w12 = SimulateDriftUseCase._apply_concept_drift(big, week=12, start=8, n_weeks=12, rng=rng_factory(2))

    assert result_w12["target"].sum() <= result_w8["target"].sum()
