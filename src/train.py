"""Entrypoint: wires adapters and executes TrainModelUseCase."""

from __future__ import annotations

import logging
from pathlib import Path

from .adapters.config_repo import YAMLConfigRepository
from .adapters.data_repo import ParquetDataRepository
from .adapters.model_repo import PickleModelRepository
from .use_cases.train_model import TrainModelUseCase

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

    result = TrainModelUseCase(data_repo, model_repo, cfg).execute()
    print(f"Done — F1={result.f1:.4f}  threshold={result.threshold:.4f}")


if __name__ == "__main__":
    main()
