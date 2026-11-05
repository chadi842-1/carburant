"""Parse the DAG folder with Airflow's own DagBag and fail on any import error.

    python airflow/check_dag.py

Uses the parsed DAG objects directly (no metadata database needed).
"""
import sys
from pathlib import Path

from airflow.models.dagbag import DagBag

EXPECTED = ["ingest", "stage", "load", "dbt_build", "anomalies"]

bag = DagBag(dag_folder=str(Path(__file__).parent / "dags"), include_examples=False)
if bag.import_errors:
    for f, err in bag.import_errors.items():
        print(f"{f}: {err}")
    sys.exit(1)
dag = bag.dags["carburant_daily"]
order = [t.task_id for t in dag.topological_sort()]
print(f"parsed {len(bag.dags)} DAG(s); carburant_daily order: {order}")
if order != EXPECTED:
    print(f"unexpected task order, expected {EXPECTED}")
    sys.exit(1)
