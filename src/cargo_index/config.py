"""Load and expose the YAML run configuration."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path).resolve()
        with open(path) as f:
            raw = yaml.safe_load(f)
        # Repo root = parent of configs/
        return cls(raw=raw, root=path.parent.parent)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, name: str) -> Path:
        p = self.root / self.raw["paths"][name]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def start(self) -> dt.date:
        return _as_date(self.raw["window"]["start"])

    @property
    def end(self) -> dt.date:
        end = self.raw["window"].get("end")
        return _as_date(end) if end else dt.date.today() - dt.timedelta(days=1)

    def dates(self) -> list[dt.date]:
        stride = self.raw["window"].get("day_stride", 1)
        n = (self.end - self.start).days + 1
        return [self.start + dt.timedelta(days=i) for i in range(0, n, stride)]

    @property
    def target_airports(self) -> list[str]:
        a = self.raw["airports"]
        return [*a["treatment"], *a["waypoint"], *a["us_dest"], *a["control"], *a["other_asia"]]

    def region_of(self, icao: str | None) -> str | None:
        if not icao:
            return None
        best, best_len = None, -1
        for region, prefixes in self.raw["regions"].items():
            for p in prefixes:
                if icao.startswith(p) and len(p) > best_len:
                    best, best_len = region, len(p)
        return best


def _as_date(v: Any) -> dt.date:
    if isinstance(v, dt.date):
        return v
    return dt.date.fromisoformat(str(v))
