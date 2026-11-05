import json
from types import SimpleNamespace

import pytest

from carburant.anomaly import run_anomaly
from carburant.evaluate import EVAL_FILE, Oracle, run_eval, score_data


def _ans(intent, **data):
    return SimpleNamespace(intent=intent, data=data)


def test_eval_file_is_well_formed():
    qs = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["questions"]
    assert len(qs) >= 40 and len({q["id"] for q in qs}) == len(qs)
    kinds = {q["kind"] for q in qs}
    assert kinds == {"data", "doc", "out_of_scope"}
    for q in qs:
        if q["kind"] == "data":
            assert q["spec"]["op"] in {"cheapest", "most_expensive", "average", "trend",
                                       "count_stations", "motorway_gap", "anomalies"}
        if q["kind"] == "doc":
            assert q["accept"]


def test_doc_accept_ids_exist_in_the_corpus():
    from carburant.rag import load_chunks

    ids = {c.id for c in load_chunks()}
    qs = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["questions"]
    for q in qs:
        if q["kind"] == "doc":
            assert set(q["accept"]) <= ids, q["id"]


def test_score_data_rules():
    spec = {"op": "average"}
    assert score_data(spec, {"prix_moyen": 1.7}, _ans("average", prix_moyen=1.7002))[0]
    assert not score_data(spec, {"prix_moyen": 1.7}, _ans("average", prix_moyen=1.71))[0]
    assert not score_data(spec, {"prix_moyen": 1.7}, _ans("trend", prix_moyen=1.7))[0]
    rank = {"op": "cheapest"}
    want = {"rows": [{"pdv_id": "A", "prix_eur": 1.5}, {"pdv_id": "B", "prix_eur": 1.6}]}
    good = _ans("cheapest", rows=[dict(r) for r in want["rows"]])
    swapped = _ans("cheapest", rows=list(reversed(good.data["rows"])))
    assert score_data(rank, want, good)[0] and not score_data(rank, want, swapped)[0]
    assert score_data(rank, {"rows": []}, _ans("cheapest", rows=[]))[0]


@pytest.fixture(scope="module")
def evaluated(built):
    paths, _ = built
    run_anomaly(paths, n_boot=20)
    return paths, run_eval(paths)


def test_oracle_agrees_with_itself_on_simple_cases(evaluated):
    paths, _ = evaluated
    o = Oracle(paths)
    cheap = o.answer({"op": "cheapest", "fuel": "Gazole", "zone": None, "days": 7, "k": 3})
    prices = [r["prix_eur"] for r in cheap["rows"]]
    assert prices == sorted(prices)
    avg = o.answer({"op": "average", "fuel": "Gazole", "zone": None, "days": 7})["prix_moyen"]
    assert prices[0] <= avg


def test_run_eval_summary_and_report(evaluated):
    paths, s = evaluated
    assert s["n_questions"] >= 40
    assert 0 <= s["accuracy"] <= 1 and s["latency_ms"]["median"] > 0
    assert set(s["accuracy_by_kind"]) == {"data", "doc", "out_of_scope"}
    assert s["accuracy"] > 0.5                       # far from random on a real warehouse
    rep = json.loads((paths.reports / "agent_eval.json").read_text(encoding="utf-8"))
    assert len(rep["questions"]) == s["n_questions"]
