import json

import pytest
from fastapi.testclient import TestClient

from carburant.anomaly import run_anomaly
from carburant.api import app, get_agent


@pytest.fixture
def client(home_env):
    run_anomaly(home_env, n_boot=20)
    (home_env.reports / "quality.json").write_text(json.dumps({"quality": {"fact_rows": 1}}))
    get_agent.cache_clear()
    yield TestClient(app)
    get_agent.cache_clear()


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] == "ready"


def test_ask_returns_a_typed_answer_with_sources(client):
    r = client.post("/ask", json={"question": "Prix moyen du gazole à Lyon cette semaine ?"})
    body = r.json()
    assert r.status_code == 200 and body["intent"] == "average" and body["grounded"]
    assert body["sources"][0]["kind"] == "sql" and body["latency_ms"] > 0


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_cheapest_endpoint_is_sorted_and_validates_fuel(client):
    r = client.get("/cheapest", params={"fuel": "Gazole", "ville": "Lyon", "k": 3})
    prices = [s["prix_eur"] for s in r.json()["stations"]]
    assert r.status_code == 200 and prices == sorted(prices) and len(prices) <= 3
    assert client.get("/cheapest", params={"fuel": "Kerosene"}).status_code == 422


def test_anomalies_and_quality(client):
    r = client.get("/anomalies", params={"days": 30})
    assert r.status_code == 200 and r.json()["total"] >= len(r.json()["top"])
    assert client.get("/quality").json() == {"fact_rows": 1}


def test_503_without_a_warehouse(tmp_path, monkeypatch):
    monkeypatch.setenv("CARBURANT_HOME", str(tmp_path))
    get_agent.cache_clear()
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/ready").status_code == 503
    assert "carburant run" in c.post("/ask", json={"question": "prix du gazole"}).json()["detail"]
    get_agent.cache_clear()
