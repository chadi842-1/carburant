import json

from carburant.cli import main
from carburant.config import get_paths
from carburant.report import render_results


def test_run_end_to_end_writes_every_report(tmp_path):
    main(["--root", str(tmp_path), "run", "--stations", "80", "--days", "30", "--seed", "3"])
    rep = tmp_path / "reports"
    for f in ("run_config.json", "quality.json", "anomaly_metrics.json", "agent_eval.json",
              "anomaly_eval.png", "dashboard.png", "results.md"):
        assert (rep / f).exists(), f
    md = (rep / "results.md").read_text(encoding="utf-8")
    for heading in ("## Warehouse and data quality", "## Anomaly detection",
                    "## Agent (RAG + SQL tool)", "## Dashboard"):
        assert heading in md
    assert md == render_results(get_paths(tmp_path))     # pure function of the JSON files
    q = json.loads((rep / "quality.json").read_text())
    assert q["quality"]["rows_conserved"]


def test_ask_command(tmp_path, capsys, built):
    paths, _ = built
    main(["--root", str(paths.root), "ask", "Que signifie l'attribut pop ?"])
    out = json.loads(capsys.readouterr().out)
    assert out["sources"][0]["ref"] == "donnees#station-sur-autoroute"
