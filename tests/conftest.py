"""Shared fixture: one small warehouse built once per test session (generate -> dbt build)."""
from datetime import date

import pytest

from carburant.config import get_paths
from carburant.ingest import stage
from carburant.synthetic import generate
from carburant.warehouse import load_raw, run_dbt

RATES = {"duplicate": 0.02, "price_typo": 0.01, "zero_price": 0.005, "missing_price": 0.005,
         "bad_timestamp": 0.005, "unknown_fuel": 0.005}


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    paths = get_paths(tmp_path_factory.mktemp("carburant_home")).ensure()
    generate(paths, n_stations=120, days=30, start=date(2026, 6, 1), seed=7, defect_rates=RATES)
    stage(paths)
    load_raw(paths)
    summary = run_dbt(paths)
    assert summary["success"], summary["failures"]
    return paths, summary


@pytest.fixture
def home_env(built, monkeypatch):
    paths, _ = built
    monkeypatch.setenv("CARBURANT_HOME", str(paths.root))
    return paths
