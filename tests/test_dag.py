"""The DAG is checked here without importing Airflow (not a runtime dependency): its tasks must
call existing CLI subcommands in the right order. Airflow itself parses the DAG in the separate
`airflow-dag` CI job (airflow/check_dag.py)."""
import ast
import re
from pathlib import Path

from carburant.cli import main

DAG_FILE = Path(__file__).resolve().parents[1] / "airflow" / "dags" / "carburant_daily.py"


def _tasks() -> dict[str, str]:
    tree = ast.parse(DAG_FILE.read_text())
    tasks = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "BashOperator":
            kw = {k.arg: k.value for k in node.keywords}
            tasks[ast.literal_eval(kw["task_id"])] = ast.unparse(kw["bash_command"])
    return tasks


def test_dag_has_the_pipeline_tasks_in_order():
    src = DAG_FILE.read_text()
    assert set(_tasks()) == {"ingest", "stage", "load", "dbt_build", "anomalies"}
    assert "ingest >> stage >> load >> dbt_build >> anomalies" in src


def test_every_task_calls_a_real_cli_subcommand(capsys):
    import pytest

    subcommands = set()
    for cmd in _tasks().values():
        subcommands |= set(re.findall(r"\}\s+(generate|ingest|stage|load|dbt|anomalies|eval|"
                                      r"report|run)\b", cmd))
    assert subcommands == {"ingest", "generate", "stage", "load", "dbt", "anomalies"}
    for sub in subcommands:
        with pytest.raises(SystemExit) as exc:
            main([sub, "--help"])
        assert exc.value.code == 0
