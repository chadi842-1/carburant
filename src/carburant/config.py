"""Central configuration: fuel catalogue, plausible ranges and the working-directory layout.

Layout (mirrors a GCP data lake so the code can be pointed at Cloud Storage / BigQuery later):

    lake/raw/date=YYYY-MM-DD/prix.xml       raw files as published (Cloud Storage bucket)
    lake/staging/{stations,prix}.parquet    parsed, typed, NOT yet validated
    warehouse/carburant.duckdb              warehouse (BigQuery dataset stand-in), built by dbt
    reports/                                 generated metrics and figures (committed)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42

# Official fuel names in the open-data files (id -> name).
FUELS: dict[int, str] = {1: "Gazole", 2: "SP95", 3: "E85", 4: "GPLc", 5: "E10", 6: "SP98"}
FUEL_IDS: dict[str, int] = {v: k for k, v in FUELS.items()}

# Plausible retail price range in EUR/litre. Values outside are quarantined by dbt.
PRICE_MIN, PRICE_MAX = 0.30, 4.00


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def lake(self) -> Path:
        return self.root / "lake"

    @property
    def raw(self) -> Path:
        return self.lake / "raw"

    @property
    def staging(self) -> Path:
        return self.lake / "staging"

    @property
    def warehouse_dir(self) -> Path:
        return self.root / "warehouse"

    @property
    def warehouse(self) -> Path:
        return self.warehouse_dir / "carburant.duckdb"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> Paths:
        for d in (self.raw, self.staging, self.warehouse_dir, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: str | os.PathLike | None = None) -> Paths:
    """Explicit root > $CARBURANT_HOME > current directory."""
    chosen = root or os.environ.get("CARBURANT_HOME") or Path.cwd()
    return Paths(Path(chosen).resolve())
