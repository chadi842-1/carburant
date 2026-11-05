import csv

import pytest

from carburant.config import FUELS, PRICE_MAX, PRICE_MIN
from carburant.geo import DEPARTEMENTS
from carburant.warehouse import DBT_DIR, connect, quality_report


def test_dbt_build_runs_models_seed_and_all_tests(built):
    _, summary = built
    assert summary["by_type"]["model"] == {"success": 10}
    assert summary["by_type"]["seed"] == {"success": 1}
    assert summary["n_tests"] >= 30
    assert set(summary["by_type"]["test"]) <= {"pass", "warn"}


def test_star_schema_tables_exist(built):
    paths, _ = built
    with connect(paths, read_only=True) as con:
        tables = {r[0] for r in con.execute(
            "select table_name from information_schema.tables where table_schema='main'"
        ).fetchall()}
    assert {"fct_prix", "dim_station", "dim_carburant", "dim_date", "dim_localisation",
            "quarantine_prix", "mart_prix_journalier", "mart_dashboard"} <= tables


def test_fact_contains_only_valid_prices(built):
    paths, _ = built
    with connect(paths, read_only=True) as con:
        lo, hi, nulls = con.execute(
            "select min(prix_eur), max(prix_eur), count(*) filter (where prix_eur is null) "
            "from fct_prix").fetchone()
        fuels = {r[0] for r in con.execute("select carburant from dim_carburant").fetchall()}
    assert PRICE_MIN <= lo and hi <= PRICE_MAX and nulls == 0
    assert fuels == set(FUELS.values())


def test_quality_layer_catches_every_injected_defect_without_false_positives(built):
    paths, _ = built
    rep = quality_report(paths)
    assert rep["rows_conserved"]
    inj = rep["injected_defects"]
    assert inj["injected"] > 50
    assert inj["recall"] == 1.0 and inj["false_positives"] == 0
    assert set(rep["by_reason"]) == {"duplicate", "out_of_range", "missing_price",
                                     "bad_timestamp", "unknown_fuel"}
    assert 0 < rep["quarantined_share"] < 0.2


def test_dim_date_is_a_complete_spine(built):
    paths, _ = built
    with connect(paths, read_only=True) as con:
        n, d0, d1 = con.execute(
            "select count(*), min(date_prix), max(date_prix) from dim_date").fetchone()
    assert n == (d1 - d0).days + 1


def test_dashboard_mart_aggregates_the_daily_mart(built):
    paths, _ = built
    with connect(paths, read_only=True) as con:
        a = con.execute("select sum(n_stations) from mart_dashboard").fetchone()[0]
        b = con.execute("select count(*) from mart_prix_journalier").fetchone()[0]
    assert a == b


def test_seed_matches_the_python_reference_table():
    with open(DBT_DIR / "seeds" / "departements.csv", encoding="utf-8") as fh:
        seed = {r["dep"]: (r["nom"], r["region"]) for r in csv.DictReader(fh)}
    assert seed == {d["dep"]: (d["nom"], d["region"]) for d in DEPARTEMENTS}


def test_bigquery_target_and_partitioning_are_declared():
    profiles = (DBT_DIR / "profiles.yml").read_text()
    fact = (DBT_DIR / "models" / "marts" / "fct_prix.sql").read_text()
    assert "type: bigquery" in profiles
    assert "partition_by" in fact and "cluster_by" in fact


def test_dbt_failure_is_reported(tmp_path):
    from carburant.config import get_paths
    from carburant.warehouse import run_dbt

    with pytest.raises(RuntimeError):
        run_dbt(get_paths(tmp_path).ensure(), dbt_dir=tmp_path / "missing_project")
