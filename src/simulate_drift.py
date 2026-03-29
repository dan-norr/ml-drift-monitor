"""Entrypoint: wires adapters and executes SimulateDriftUseCase."""

from __future__ import annotations

import logging
from pathlib import Path

from .adapters.config_repo import YAMLConfigRepository
from .adapters.data_repo import ParquetDataRepository
from .use_cases.simulate_drift import SimulateDriftUseCase

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

    SimulateDriftUseCase(data_repo, cfg).execute()


if __name__ == "__main__":
    main()
