import pytest

from carburant.agent import Agent
from carburant.intent import (
    INTENTS,
    parse_autoroute,
    parse_days_v2,
    parse_fuel_v2,
    parse_k_v2,
    train,
    training_data,
)
from carburant.rag import normalize


@pytest.fixture(scope="module")
def model():
    return train()


def test_training_data_covers_every_intent_and_excludes_eval_questions():
    import json

    from carburant.evaluate import EVAL_FILE

    X, y = training_data()
    assert set(y) == set(INTENTS)
    evalq = set()
    for f in (EVAL_FILE, EVAL_FILE.with_name("questions_test_v2.json")):
        evalq |= {normalize(q["question"]).strip(" ?") for q in
                  json.loads(f.read_text(encoding="utf-8"))["questions"]}
    assert not evalq & {normalize(x).strip(" ?") for x in X}


def test_model_is_deterministic_and_confident_on_clear_questions(model):
    again = train()
    q = "Quel est le prix moyen du gazole à Nantes ?"
    assert model.predict(q) == again.predict(q)
    label, conf = model.predict(q)
    assert label == "average" and conf > model.threshold


@pytest.mark.parametrize("q,fuel", [
    ("le prix du sp95-e10", "E10"), ("sp95 e10", "E10"), ("sans plomb 95", "SP95"),
    ("sp98", "SP98"), ("le gasoil", "Gazole"), ("du gpl", "GPLc"), ("rien", None),
])
def test_fuel_v2(q, fuel):
    assert parse_fuel_v2(normalize(q)) == fuel


@pytest.mark.parametrize("q,k", [
    ("top 3 des stations", 3), ("top5", 5), ("les trois stations", 3), ("4 stations", 4),
    ("les 2 premieres stations", 2), ("la station", 1), ("top 50 des stations", 10),
])
def test_k_v2(q, k):
    assert parse_k_v2(q) == k


@pytest.mark.parametrize("q,days", [
    ("aujourd'hui", 1), ("depuis deux semaines", 14), ("sur 15 jours", 14), ("ce mois-ci", 30),
    ("ces 30 derniers jours", 30), ("en ce moment", 7),
])
def test_days_v2(q, days):
    assert parse_days_v2(normalize(q)) == days


def test_autoroute_filter_except_for_the_comparison():
    assert parse_autoroute("prix moyen sur autoroute", "average")
    assert not parse_autoroute("ecart autoroute route", "motorway_gap")
    assert not parse_autoroute("prix moyen a lyon", "average")


def test_ml_router_end_to_end(built, model):
    from carburant.anomaly import run_anomaly

    paths, _ = built
    run_anomaly(paths, n_boot=20)
    a = Agent(paths, router="ml", intent_model=model)
    ans = a.ask("Prix moyen du gazole sur autoroute cette semaine ?")
    assert ans.intent == "average" and "autoroute" in ans.answer
    assert ans.data["intent_confidence"] >= model.threshold
    assert a.ask("Raconte-moi une histoire de pirates").intent == "abstain"
    assert a.ask("Que signifie l'attribut pop ?").intent == "doc"


def test_unknown_router_is_rejected(built):
    with pytest.raises(ValueError):
        Agent(built[0], router="magic")
