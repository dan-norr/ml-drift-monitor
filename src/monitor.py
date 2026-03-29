"""Entrypoint: wires adapters and executes MonitorDriftUseCase."""

from __future__ import annotations

import logging
from pathlib import Path

from .adapters.config_repo import YAMLConfigRepository
from .adapters.data_repo import ParquetDataRepository
from .adapters.drift_analyser import EvidentlyDriftAnalyser
from .adapters.metrics_repo import JSONMetricsRepository
from .adapters.model_repo import PickleModelRepository
from .use_cases.monitor_drift import MonitorDriftUseCase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    config_path = "config.yaml" if Path("config.yaml").exists() else "../config.yaml"
    cfg = YAMLConfigRepository(config_path).load()

    data_repo = ParquetDataRepository(
        reference_path=cfg["data"]["reference_output"],
        simulated_dir=cfg["data"]["simulated_dir"],
    )
    model_repo = PickleModelRepository(
        model_path=cfg["data"]["model_output"],
        feature_importance_path=cfg["data"]["feature_importance_output"],
    )
    metrics_repo = JSONMetricsRepository(
        metrics_dir=cfg["monitoring"]["metrics_dir"],
        reports_dir=cfg["monitoring"]["reports_dir"],
    )

    MonitorDriftUseCase(
        data_repo=data_repo,
        model_repo=model_repo,
        metrics_repo=metrics_repo,
        drift_analyser=EvidentlyDriftAnalyser(),
        cfg=cfg,
    ).execute()


if __name__ == "__main__":
    main()
