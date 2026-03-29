"""adapters/config_repo.py — YAML-backed config repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class YAMLConfigRepository:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self._path = Path(config_path)

    def load(self) -> dict[str, Any]:
        with open(self._path) as fh:
            return yaml.safe_load(fh)
