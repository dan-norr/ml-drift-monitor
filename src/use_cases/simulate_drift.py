"""
use_cases/simulate_drift.py — Generate 12 weekly batches with progressive drift.

Drift schedule (from config.yaml):
  Weeks 1-3  : subtle     (PSI ~0.05-0.10)
  Weeks 4-7  : detectable (PSI ~0.10-0.20)
  Weeks 8-12 : critical   (PSI > 0.20, concept drift introduced)

All file I/O is delegated to the injected IDataRepository.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..domain.ports import IDataRepository

logger = logging.getLogger(__name__)


class SimulateDriftUseCase:
    def __init__(self, data_repo: IDataRepository, cfg: dict[str, Any]) -> None:
        self._data = data_repo
        self._cfg = cfg

    def execute(self) -> None:
        sim_cfg = self._cfg["drift_simulation"]
        n_weeks: int = sim_cfg["n_weeks"]
        rng = np.random.default_rng(sim_cfg["random_seed"])

        reference = self._data.load_reference()

        for week in range(1, n_weeks + 1):
            batch = self._generate_batch(reference, week, sim_cfg, rng)
            self._data.save_batch(batch, week)
            logger.info(
                "Week %2d saved (rows=%d, fraud_rate=%.2f%%)",
                week, len(batch), batch["target"].mean() * 100,
            )

        logger.info("All %d batches generated.", n_weeks)

    # ── private helpers ──────────────────────────────────────────────────────

    def _generate_batch(
        self,
        reference: pd.DataFrame,
        week: int,
        sim_cfg: dict[str, Any],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        batch_size: int = sim_cfg["batch_size"]
        min_fraud: int = sim_cfg.get("min_fraud_per_batch", 10)
        seed = int(rng.integers(1, 9999))

        fraud_ref = reference[reference["target"] == 1]
        legit_ref = reference[reference["target"] == 0]
        n_fraud = max(min_fraud, int(batch_size * reference["target"].mean()))
        n_legit = batch_size - n_fraud

        batch = pd.concat(
            [
                fraud_ref.sample(n=n_fraud, replace=True, random_state=seed),
                legit_ref.sample(n=n_legit, replace=True, random_state=seed + 1),
            ],
            ignore_index=True,
        ).sample(frac=1, random_state=seed).reset_index(drop=True)

        intensity = self._drift_intensity(week, sim_cfg["drift_schedule"])
        if intensity > 0:
            feature_cols = [c for c in sim_cfg["drift_features"] if c in batch.columns]
            batch = self._apply_data_drift(batch, feature_cols, intensity, rng)

        batch = self._apply_concept_drift(
            batch, week,
            start=sim_cfg["concept_drift_start_week"],
            n_weeks=sim_cfg["n_weeks"],
            rng=rng,
        )
        batch["week"] = week
        return batch

    @staticmethod
    def _drift_intensity(week: int, schedule: dict[str, Any]) -> float:
        if week in schedule["subtle_weeks"]:
            return float(schedule["subtle_intensity"])
        if week in schedule["detectable_weeks"]:
            return float(schedule["detectable_intensity"])
        if week in schedule["critical_weeks"]:
            return float(schedule["critical_intensity"])
        return 0.0

    @staticmethod
    def _apply_data_drift(
        batch: pd.DataFrame,
        features: list[str],
        intensity: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        result = batch.copy()
        for i, feat in enumerate(features):
            if feat not in result.columns:
                continue
            std = result[feat].std()
            direction = 1.0 if i % 2 == 0 else -1.0
            noise = rng.normal(
                loc=direction * intensity * std,
                scale=0.1 * std,
                size=len(result),
            )
            result[feat] = result[feat] + noise
        return result

    @staticmethod
    def _apply_concept_drift(
        batch: pd.DataFrame,
        week: int,
        start: int,
        n_weeks: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        if week < start:
            return batch
        result = batch.copy()
        progress = (week - start) / (n_weeks - start + 1)
        flip_rate = 0.40 * progress
        fraud_idx = result.index[result["target"] == 1].tolist()
        n_flip = int(len(fraud_idx) * flip_rate)
        if n_flip > 0:
            flip_idx = rng.choice(fraud_idx, size=n_flip, replace=False)
            result.loc[flip_idx, "target"] = 0
        return result
