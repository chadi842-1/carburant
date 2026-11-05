import duckdb
import pytest

from carburant.agent import Agent, ZoneResolver, classify, parse_days, parse_fuel, parse_k
from carburant.anomaly import run_anomaly
from carburant.rag import Retriever, load_chunks, normalize, tokenize
from carburant.tools import SqlTool, Zone


# ---- retrieval
def test_chunks_cover_every_doc_and_have_unique_ids():
    chunks = load_chunks()
    assert {c.doc for c in chunks} == {"architecture", "qualite", "donnees", "anomalies",
                                       "assistant"}
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.text for c in chunks)


def test_tokenize_stems_french_inflections_together():
    assert tokenize("évaluée")[0] == tokenize("évaluation")[0]
    assert tokenize("réelles") == tokenize("réel")
    assert tokenize("le la les de") == []


def test_retriever_finds_the_right_section_and_scores_off_topic_low():
    r = Retriever()
    assert r.search("Que signifie l'attribut pop ?")[0].chunk.id == "donnees#station-sur-autoroute"
    assert r.search("Qui a gagné la coupe du monde ?") == []


# ---- parsing
@pytest.mark.parametrize("q,fuel", [
    ("prix du gazole", "Gazole"), ("diesel pas cher", "Gazole"), ("sans plomb 98", "SP98"),
    ("le sp95", "SP95"), ("e10 a lyon", "E10"), ("superethanol", "E85"), ("gpl", "GPLc"),
    ("prix moyen", None),
])
def test_parse_fuel(q, fuel):
    assert parse_fuel(normalize(q)) == fuel


def test_parse_days_and_k():
    assert parse_days("aujourd'hui") == 1
    assert parse_days("ces 30 derniers jours") == 30 and parse_days("ce mois") == 30
    assert parse_days("cette semaine") == 7 and parse_days("") == 7
    assert parse_k("les 3 stations les moins cheres") == 3
    assert parse_k("les cinq stations") == 5 and parse_k("la station") == 1


def test_classify_doc_versus_data():
    z = Zone("ville", "Lyon", "Lyon")
    assert classify("comment est evaluee la detection d anomalies", None, Zone()) == "doc"
    assert classify("comment le prix du gazole a-t-il evolue a lyon", "Gazole", z) == "trend"
    assert classify("quelle station est la moins chere", "Gazole", z) == "cheapest"
    assert classify("combien de stations a lyon", None, z) == "count_stations"
    assert classify("prix suspects a lyon", None, z) == "anomalies"


# ---- tools and agent on the built warehouse
@pytest.fixture(scope="module")
def agent(built):
    paths, _ = built
    run_anomaly(paths, n_boot=20)
    return Agent(paths)


def test_zone_resolver_prefers_longest_match(agent):
    g = agent.tool.gazetteer()
    res = ZoneResolver(g)
    assert res.resolve(normalize("à Lyon")).column == "ville"
    assert res.resolve(normalize("en Provence-Alpes-Côte d'Azur")).column == "region"
    assert res.resolve(normalize("dans le département 13")).value == "13"
    assert res.resolve(normalize("au Groenland")).column is None


def test_sql_tool_is_read_only(built):
    paths, _ = built
    tool = SqlTool(paths)
    with pytest.raises(duckdb.Error):
        tool._query("create table hack as select 1", [])


def test_sql_tool_rejects_unknown_zone_columns(built):
    paths, _ = built
    with pytest.raises(ValueError):
        SqlTool(paths).average("Gazole", Zone("pdv_id; drop table x", "1", "x"), 7)


def test_cheapest_answer_matches_an_independent_pandas_computation(agent, built):
    paths, _ = built
    ans = agent.ask("Quelle est la station la moins chère pour le gazole à Lyon cette semaine ?")
    assert ans.intent == "cheapest" and ans.grounded and ans.sources[0].kind == "sql"
    with duckdb.connect(str(paths.warehouse), read_only=True) as con:
        df = con.execute("select * from mart_prix_journalier").df()
    end = df["date_prix"].max()
    w = df[(df.carburant == "Gazole") & (df.ville == "Lyon")
           & (df.date_prix > end - __import__("pandas").Timedelta(days=7))]
    latest = w.sort_values("date_prix").groupby("pdv_id").tail(1)
    best = latest.sort_values(["prix_eur", "pdv_id"]).iloc[0]
    assert ans.data["rows"][0]["pdv_id"] == best.pdv_id
    assert ans.data["rows"][0]["prix_eur"] == pytest.approx(best.prix_eur)
    assert f"{best.prix_eur:.3f}".replace(".", ",") in ans.answer


def test_top_k_and_default_fuel_note(agent):
    ans = agent.ask("Les 3 stations les plus chères à Marseille")
    assert len(ans.data["rows"]) <= 3 and "Gazole par défaut" in ans.answer
    prices = [r["prix_eur"] for r in ans.data["rows"]]
    assert prices == sorted(prices, reverse=True)


def test_doc_answer_cites_its_section(agent):
    ans = agent.ask("Pourquoi mettre les lignes en quarantaine plutôt que les supprimer ?")
    assert ans.intent == "doc" and ans.grounded
    assert ans.sources[0].ref == "qualite#pourquoi-mettre-en-quarantaine-plutot-que-supprimer"


def test_off_topic_question_abstains(agent):
    ans = agent.ask("Quelle est la météo à Lyon demain ?")
    assert ans.intent == "abstain" and not ans.grounded and not ans.sources


def test_anomaly_question_counts_flags(agent, built):
    paths, _ = built
    ans = agent.ask("Combien de prix suspects en France ce mois ?")
    assert ans.intent == "anomalies" and ans.grounded
    with duckdb.connect(str(paths.warehouse), read_only=True) as con:
        total = con.execute("select count(*) from ml.anomalies_prix").fetchone()[0]
    assert 0 <= ans.data["n_anomalies"] <= total


def test_llm_hook_rewords_without_touching_the_data(built):
    paths, _ = built
    a = Agent(paths, llm_fn=lambda q, facts: "REWORDED")
    ans = a.ask("Prix moyen du SP98 en Bretagne ?")
    assert ans.answer == "REWORDED" and "prix_moyen" in ans.data


def test_latency_is_recorded(agent):
    assert agent.ask("Prix moyen du gazole à Paris ?").latency_ms > 0
