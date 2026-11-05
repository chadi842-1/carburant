"""Daily fuel-price pipeline: ingest -> stage -> load -> dbt build -> anomalies.

Each task shells out to the `carburant` CLI (installed in the Airflow image), so the DAG holds no
business logic and every step can be run by hand exactly as Airflow runs it. If `dbt build`
fails (a blocking data test), the downstream anomaly task does not run and yesterday's tables
stay exposed.

Configuration (Airflow Variables):
    carburant_home   working directory holding lake/, warehouse/, reports/
    carburant_url    URL of the daily file (official feed); if unset the task falls back to
                     the synthetic generator, which is what the demo uses.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

HOME = "{{ var.value.get('carburant_home', '/opt/carburant') }}"
URL = "{{ var.value.get('carburant_url', '') }}"
CLI = f"carburant --root {HOME}"

default_args = {
    "owner": "data",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="carburant_daily",
    description="Fuel prices: ingest, ELT with dbt quality tests, anomaly detection",
    schedule="30 6 * * *",           # the official daily file is published overnight
    start_date=datetime(2026, 6, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["carburant", "elt", "dbt"],
) as dag:
    ingest = BashOperator(
        task_id="ingest",
        bash_command=(
            f'if [ -n "{URL}" ]; then {CLI} ingest --url "{URL}" --date {{{{ ds }}}}; '
            f"else {CLI} generate --stations 1000 --days 120; fi"
        ),
    )
    stage = BashOperator(task_id="stage", bash_command=f"{CLI} stage")
    load = BashOperator(task_id="load", bash_command=f"{CLI} load")
    dbt_build = BashOperator(task_id="dbt_build", bash_command=f"{CLI} dbt")
    anomalies = BashOperator(task_id="anomalies", bash_command=f"{CLI} anomalies",
                             trigger_rule="all_success")

    ingest >> stage >> load >> dbt_build >> anomalies
