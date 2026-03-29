"""adapters/metrics_repo.py — JSON metrics + HTML report repository."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domain.entities import WeeklyMetrics


class JSONMetricsRepository:
    def __init__(self, metrics_dir: str, reports_dir: str) -> None:
        self._metrics_dir = Path(metrics_dir)
        self._reports_dir = Path(reports_dir)

    def save(self, metrics: WeeklyMetrics) -> None:
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        path = self._metrics_dir / f"week_{metrics.week:02d}.json"
        with open(path, "w") as fh:
            json.dump(metrics.to_dict(), fh, indent=2)

    def load_all(self) -> list[dict[str, Any]]:
        files = sorted(self._metrics_dir.glob("week_*.json"))
        result: list[dict[str, Any]] = []
        for f in files:
            with open(f) as fh:
                result.append(json.load(fh))
        return result

    def save_report_html(self, html: str, week: int) -> None:
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        path = self._reports_dir / f"week_{week:02d}.html"
        path.write_text(html, encoding="utf-8")

    def load_report_html(self, week: int) -> str:
        return (self._reports_dir / f"week_{week:02d}.html").read_text(encoding="utf-8")

    def report_exists(self, week: int) -> bool:
        return (self._reports_dir / f"week_{week:02d}.html").exists()
