"""Load (the L of ELT) and run dbt (the T), then score the quality layer.

- `load_raw` copies the staged parquet into schema `raw` of the warehouse, untouched.
- `run_dbt` builds and tests the dbt project (staging -> star schema + quarantine).
- `quality_report` measures how much was quarantined, why, and, on synthetic data, how many of
  the injected defects were caught (recall) and how many clean rows were wrongly quarantined.

The warehouse is DuckDB here; the same dbt project targets BigQuery through the `bigquery`
output in dbt/profiles.yml (date-partitioned, clustered fact table).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

from .config import Paths

DBT_DIR = Path(__file__).resolve().parents[2] / "dbt"


def connect(paths: Paths, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(paths.warehouse), read_only=read_only)


def load_raw(paths: Paths) -> dict[str, int]:
    """Replace raw.stations / raw.prix / raw.ruptures with the staged parquet files."""
    paths.ensure()
    counts = {}
    with connect(paths) as con:
        con.execute("create schema if not exists raw")
        for name in ("stations", "prix", "ruptures"):
            f = paths.staging / f"{name}.parquet"
            con.execute(f"create or replace table raw.{name} as "
                        f"select * from read_parquet('{f.as_posix()}')")
            counts[name] = con.execute(f"select count(*) from raw.{name}").fetchone()[0]
    return counts


@contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_dbt(paths: Paths, command: str = "build", dbt_dir: Path = DBT_DIR) -> dict:
    """Run `dbt <command>` (seed + run + test for build) against this warehouse.

    dbt runs in a subprocess, exactly as an orchestrator would call it, so it cannot leave a
    warehouse connection open in this process. Returns a summary parsed from run_results.json.
    """
    target_dir = paths.warehouse_dir / "dbt_target"
    args = [_dbt_executable(), command, "--project-dir", str(dbt_dir),
            "--profiles-dir", str(dbt_dir), "--target-path", str(target_dir),
            "--log-path", str(paths.warehouse_dir / "dbt_logs"), "--quiet"]
    with _env(CARBURANT_DUCKDB=paths.warehouse, DO_NOT_TRACK=1):   # no dbt telemetry
        proc = subprocess.run(args, capture_output=True, text=True)
    results_file = target_dir / "run_results.json"
    if not results_file.exists():
        raise RuntimeError(f"dbt {command} did not run:\n{proc.stdout}\n{proc.stderr}")
    results = json.loads(results_file.read_text())["results"]
    by_type: dict[str, dict[str, int]] = {}
    failures = []
    for r in results:
        kind = r["unique_id"].split(".")[0]
        by_type.setdefault(kind, {}).setdefault(r["status"], 0)
        by_type[kind][r["status"]] += 1
        if r["status"] in {"fail", "error"}:
            failures.append({"node": r["unique_id"], "message": r.get("message")})
    return {"success": proc.returncode == 0, "by_type": by_type, "failures": failures,
            "n_tests": sum(by_type.get("test", {}).values())}


def _dbt_executable() -> str:
    local = Path(sys.executable).parent / "dbt"
    found = str(local) if local.exists() else shutil.which("dbt")
    if not found:
        raise FileNotFoundError("dbt is not installed (pip install dbt-duckdb)")
    return found


def quality_report(paths: Paths) -> dict:
    """Quarantine statistics, plus recall / false positives against the synthetic manifest."""
    with connect(paths, read_only=True) as con:
        n_staged = con.execute("select count(*) from raw.prix").fetchone()[0]
        n_fact = con.execute("select count(*) from main.fct_prix").fetchone()[0]
        q = con.execute("select * from main.quarantine_prix").df()
    report = {
        "staged_rows": int(n_staged),
        "fact_rows": int(n_fact),
        "quarantined_rows": int(len(q)),
        "quarantined_share": round(len(q) / n_staged, 5) if n_staged else 0.0,
        "by_reason": {k: int(v) for k, v in q["reason"].value_counts().sort_index().items()},
        "rows_conserved": int(n_fact + len(q)) == int(n_staged),
    }
    manifest = paths.lake / "_truth" / "defects.parquet"
    if manifest.exists():
        report["injected_defects"] = _score_against_manifest(pd.read_parquet(manifest), q)
    return report


_EXPECTED_REASON = {"price_typo": "out_of_range", "zero_price": "out_of_range",
                    "missing_price": "missing_price", "bad_timestamp": "bad_timestamp",
                    "unknown_fuel": "unknown_fuel", "duplicate": "duplicate"}


def _score_against_manifest(truth: pd.DataFrame, q: pd.DataFrame) -> dict:
    key = ["pdv_id", "carburant", "maj_raw", "valeur_raw"]
    q = q.copy()
    q["valeur_raw"] = q["valeur_raw"].fillna("")
    t = truth.copy()
    t["valeur_raw"] = t["valeur_raw"].fillna("")
    per_type = {}
    caught_total = 0
    for kind, grp in t.groupby("defect"):
        reason = _EXPECTED_REASON[kind]
        qq = q[q["reason"] == reason]
        if kind == "duplicate":
            # every extra copy must be quarantined as a duplicate; count by key multiplicity
            want = grp.groupby(key).size()
            got = qq.groupby(key).size().reindex(want.index, fill_value=0)
            caught = int(got.clip(upper=want).sum())
        else:
            merged = grp.merge(qq[key].drop_duplicates(), on=key, how="left", indicator=True)
            caught = int((merged["_merge"] == "both").sum())
        per_type[kind] = {"injected": int(len(grp)), "caught": caught,
                          "recall": round(caught / len(grp), 4)}
        caught_total += caught
    injected = int(len(t))
    false_pos = max(int(len(q)) - caught_total, 0)
    return {"injected": injected, "caught": caught_total,
            "recall": round(caught_total / injected, 4) if injected else None,
            "false_positives": false_pos, "per_type": per_type}
