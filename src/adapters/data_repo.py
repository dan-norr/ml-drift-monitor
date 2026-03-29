"""adapters/data_repo.py — Parquet-backed data repository."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetDataRepository:
    def __init__(self, reference_path: str, simulated_dir: str) -> None:
        self._ref_path = Path(reference_path)
        self._sim_dir = Path(simulated_dir)

    def load_raw(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def save_reference(self, df: pd.DataFrame) -> None:
        self._ref_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._ref_path, index=False)

    def load_reference(self) -> pd.DataFrame:
        return pd.read_parquet(self._ref_path)

    def save_batch(self, df: pd.DataFrame, week: int) -> None:
        self._sim_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._sim_dir / f"week_{week:02d}.parquet", index=False)

    def load_batch(self, week: int) -> pd.DataFrame:
        return pd.read_parquet(self._sim_dir / f"week_{week:02d}.parquet")

    def batch_paths(self) -> list[Path]:
        return sorted(self._sim_dir.glob("week_*.parquet"))
