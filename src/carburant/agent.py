"""LangGraph agent: route -> (sql | rag | abstain) -> compose an answer with its sources.

No LLM is called. The router is a transparent rule set (intent cues + slot extraction for fuel,
zone, period and top-k), so every answer is reproducible and its failure modes are visible in
the evaluation. `llm_fn` lets a language model reword the final answer from the computed facts;
it cannot change the numbers, which stay in `data`.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import Paths, get_paths
from .rag import Retriever, normalize
from .tools import SqlTool, ToolResult, Zone

Intent = Literal["cheapest", "most_expensive", "average", "trend", "count_stations",
                 "anomalies", "motorway_gap", "doc", "abstain"]

FUEL_PATTERNS: list[tuple[str, str]] = [
    (r"\bsp\s?98\b|sans plomb 98|super 98", "SP98"),
    (r"\bsp\s?95\b|sans plomb 95", "SP95"),
    (r"\be\s?10\b", "E10"),
    (r"\be\s?85\b|superethanol|bioethanol", "E85"),
    (r"\bgplc?\b|\bgaz de petrole", "GPLc"),
    (r"\bgazole\b|\bdiesel\b|\bgasoil\b", "Gazole"),
]
DATA_CUES = {
    "count_stations": [r"combien de stations", r"nombre de stations", r"combien .*stations"],
    "anomalies": [r"anomal", r"suspect", r"aberrant"],
    "motorway_gap": [r"autoroute.*(ecart|difference|plus cher|surcout|compar)",
                     r"(ecart|difference|surcout|compar).*autoroute"],
    "trend": [r"evolu", r"tendance", r"(a|ont) (augmente|baisse|bouge)", r"variation",
              r"hausse ou (en )?baisse"],
    "cheapest": [r"moins ch[e]r", r"meilleur prix", r"plus bas", r"le moins couteux",
                 r"(prix|tarif)s? les? plus bas", r"moins onereu"],
    "most_expensive": [r"plus ch[e]r", r"plus eleve", r"plus couteu"],
    "average": [r"prix moyen", r"\bmoyenne\b", r"en moyenne", r"combien (coute|vaut|coutait)",
                r"quel est le prix", r"\bprix du\b", r"\bprix de\b"],
}
DOC_CUES = [r"^comment\b", r"^pourquoi\b", r"qu.est.ce", r"que signifie", r"a quoi sert",
            r"\bexplique", r"\bdefinition\b", r"\bregle", r"\bmethode\b", r"\bprotocole\b",
            r"\bpipeline\b", r"\bdbt\b", r"\bairflow\b", r"\bquarantaine\b", r"\bbm25\b",
            r"\bformat\b", r"\bunite", r"\bentrepot\b", r"\bpartition", r"\bisolation forest\b",
            r"\bvrai(es)? donnees\b", r"\bsynthetique", r"\btest", r"\bevalu", r"\blimite",
            r"\bgithub\b", r"\bci\b", r"\bpop\b", r"\bmodele de langage\b", r"\bllm\b"]
NUMBER_WORDS = {"deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "dix": 10}


@dataclass
class Parsed:
    intent: Intent
    fuel: str | None
    fuel_defaulted: bool
    zone: Zone
    days: int
    k: int


def _search(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def parse_fuel(q: str) -> str | None:
    for pat, fuel in FUEL_PATTERNS:
        if re.search(pat, q):
            return fuel
    return None


def parse_days(q: str) -> int:
    if "aujourd" in q:
        return 1
    m = re.search(r"(\d+) derniers jours", q)
    if m:
        return max(1, int(m.group(1)))
    if re.search(r"\bce mois\b|dernier mois|30 jours|un mois", q):
        return 30
    if re.search(r"deux semaines|15 jours|quinzaine", q):
        return 14
    return 7                                   # default: "cette semaine"


def parse_k(q: str) -> int:
    m = re.search(r"\b(\d+|deux|trois|quatre|cinq|dix) (premieres |premiers )?stations", q)
    if not m:
        return 1
    v = m.group(1)
    return min(int(v) if v.isdigit() else NUMBER_WORDS[v], 10)


def _zone_key(s: str) -> str:
    """Same normalisation for place names and questions: no accents, case, hyphen or apostrophe."""
    s = re.sub(r"[-'’?!.,;:]", " ", normalize(s))
    return re.sub(r"\s+", " ", s).strip()


class ZoneResolver:
    """Longest-match lookup of a city, département, département code or region in the question."""

    def __init__(self, gazetteer: dict[str, list[str]]):
        self.entries: list[tuple[str, str, str]] = []      # (normalized name, column, value)
        for col in ("ville", "departement", "region"):
            for v in gazetteer.get(col, []):
                self.entries.append((_zone_key(v), col, v))
        self.entries.sort(key=lambda e: -len(e[0]))
        self.deps = set(gazetteer.get("dep", []))

    def resolve(self, q: str) -> Zone:
        qq = f" {_zone_key(q)} "
        for name, col, value in self.entries:
            if f" {name} " in qq:
                return Zone(col, value, value)
        m = re.search(r"\b(?:departement|dept|dep)\.?\s*(?:du |de |n° ?)?(\d{2}|2a|2b)\b", q)
        if m and m.group(1).upper() in {d.upper() for d in self.deps}:
            code = m.group(1).upper()
            return Zone("dep", code, f"département {code}")
        return Zone()


STRONG_DOC = [r"^comment (est|sont|fonctionne|marche|se fait|on)\b", r"^pourquoi\b",
              r"qu.est.ce", r"que signifie", r"a quoi sert", r"\bexplique"]


def classify(q: str, fuel: str | None, zone: Zone) -> Intent:
    # a "how/why/what is" question with no fuel and no place is about the method, not the data
    if _search(STRONG_DOC, q) and not fuel and zone.column is None:
        return "doc"
    for intent in ("count_stations", "anomalies", "motorway_gap", "trend", "cheapest",
                   "most_expensive"):
        if _search(DATA_CUES[intent], q):
            return intent
    doc = _search(DOC_CUES, q)
    if _search(DATA_CUES["average"], q) and (fuel or zone.column) and not doc:
        return "average"
    if doc:
        return "doc"
    if fuel and zone.column:
        return "average"
    return "doc"                          # retrieval decides; low relevance -> abstain


# --------------------------------------------------------------------------- graph
class Source(BaseModel):
    kind: Literal["sql", "doc"]
    ref: str
    detail: str | None = None


class Answer(BaseModel):
    question: str
    intent: str
    answer: str
    data: dict = Field(default_factory=dict)
    sources: list[Source] = Field(default_factory=list)
    grounded: bool = False
    latency_ms: float = 0.0


class State(TypedDict, total=False):
    question: str
    q: str
    parsed: Parsed
    tool: ToolResult | None
    hits: list
    answer: Answer


def _fmt_price(p: float) -> str:
    return f"{p:.3f}".replace(".", ",") + " €/l"


def _period(meta: dict) -> str:
    return f"du {meta['start']:%d/%m} (exclu) au {meta['end']:%d/%m/%Y}"


def route_node(state: State, resolver: ZoneResolver) -> dict:
    q = normalize(state["question"])
    fuel = parse_fuel(q)
    zone = resolver.resolve(q)
    intent = classify(q, fuel, zone)
    needs_fuel = intent in {"cheapest", "most_expensive", "average", "trend", "motorway_gap"}
    parsed = Parsed(intent, fuel or ("Gazole" if needs_fuel else None),
                    needs_fuel and fuel is None, zone, parse_days(q), parse_k(q))
    return {"q": q, "parsed": parsed}


def sql_node(state: State, tool: SqlTool) -> dict:
    p = state["parsed"]
    if p.intent in ("cheapest", "most_expensive"):
        res = tool.extreme(p.fuel, p.zone, p.days, p.k, cheapest=p.intent == "cheapest")
    elif p.intent == "average":
        res = tool.average(p.fuel, p.zone, p.days)
    elif p.intent == "trend":
        res = tool.trend(p.fuel, p.zone, max(p.days, 14))
    elif p.intent == "count_stations":
        res = tool.count_stations(p.zone, p.days, p.fuel)
    elif p.intent == "motorway_gap":
        res = tool.motorway_gap(p.fuel, p.zone, p.days)
    else:
        res = tool.anomalies(p.zone, p.days, p.fuel)
    return {"tool": res}


def rag_node(state: State, retriever: Retriever) -> dict:
    return {"hits": retriever.search(state["question"], k=3)}


def compose_node(state: State, min_score: float, min_terms: int,
                 llm_fn: Callable[[str, dict], str] | None) -> dict:
    p, question = state["parsed"], state["question"]
    tool, hits = state.get("tool"), state.get("hits") or []
    ans = Answer(question=question, intent=p.intent, answer="")
    if tool is not None:
        _compose_sql(ans, p, tool)
    elif hits and hits[0].score >= min_score and hits[0].matched_terms >= min_terms:
        top = hits[0]
        ans.answer = top.chunk.text
        ans.data = {"chunk": top.chunk.id, "score": round(top.score, 3)}
        ans.sources = [Source(kind="doc", ref=h.chunk.id, detail=h.chunk.title)
                       for h in hits if h.score >= min_score]
        ans.grounded = True
    else:
        ans.intent = "abstain"
        ans.answer = ("Je ne sais pas répondre à cette question avec les données et la "
                      "documentation disponibles.")
    if llm_fn is not None and ans.grounded:
        ans.answer = llm_fn(question, {"answer": ans.answer, "data": ans.data})
    return {"answer": ans}


def _compose_sql(ans: Answer, p: Parsed, tool: ToolResult) -> None:
    rows, meta = tool.rows, tool.meta
    where = _where(p.zone)
    note = " (carburant non précisé : Gazole par défaut)" if p.fuel_defaulted else ""
    ans.sources = [Source(kind="sql", ref=tool.name, detail=" ".join(tool.sql.split()))]
    if meta.get("missing"):
        ans.sources = []
        ans.answer = "La détection d'anomalies n'a pas encore été exécutée sur cet entrepôt."
        return
    if tool.name == "anomalies":
        _compose_anomalies(ans, p, rows, meta, where)
        return
    empty = not rows or all(v is None for v in rows[0].values())
    if empty or (tool.name in ("cheapest", "most_expensive") and not rows):
        ans.answer = f"Aucune donnée pour {p.fuel or 'ces critères'} {where} {_period(meta)}."
        ans.data = {"rows": []}
        return
    ans.grounded = True
    if tool.name in ("cheapest", "most_expensive"):
        adj = "la moins chère" if tool.name == "cheapest" else "la plus chère"
        lines = [f"{r['pdv_id']} ({r['ville']}{', autoroute' if r['autoroute'] else ''}) : "
                 f"{_fmt_price(r['prix_eur'])} le {r['date_prix']:%d/%m}" for r in rows]
        head = (f"Station {adj} pour le {p.fuel} {where} {_period(meta)}{note} : "
                if len(rows) == 1 else
                f"Les {len(rows)} stations {adj.replace('la ', 'les ').replace('chère', 'chères')}"
                f" pour le {p.fuel} {where} {_period(meta)}{note} : ")
        ans.answer = head + " ; ".join(lines) + "."
        ans.data = {"rows": [{k: (str(v) if k == "date_prix" else v) for k, v in r.items()}
                             for r in rows]}
    elif tool.name == "average":
        r = rows[0]
        ans.answer = (f"Prix moyen du {p.fuel} {where} {_period(meta)}{note} : "
                      f"{_fmt_price(r['prix_moyen'])} (moyenne de {r['n_releves']} relevés "
                      f"journaliers, {r['n_stations']} stations).")
        ans.data = {"prix_moyen": r["prix_moyen"], "n_stations": r["n_stations"]}
    elif tool.name == "trend":
        r = rows[0]
        if r["debut"] is None or r["fin"] is None:
            ans.grounded = False
            ans.answer = f"Pas assez de données pour mesurer une évolution {where}."
            return
        delta = r["fin"] - r["debut"]
        sens = "hausse" if delta > 0 else "baisse" if delta < 0 else "stabilité"
        cents = _fmt_num(abs(delta) * 100, ".1f")
        ans.answer = (f"Évolution du {p.fuel} {where} {_period(meta)}{note} : {sens} de "
                      f"{cents} centimes ({_fmt_price(r['debut'])} en moyenne "
                      f"la première semaine, {_fmt_price(r['fin'])} la dernière).")
        ans.data = {"debut": r["debut"], "fin": r["fin"], "delta": delta}
    elif tool.name == "count_stations":
        n = rows[0]["n_stations"]
        fuel = f" vendant du {p.fuel}" if p.fuel else ""
        ans.answer = f"{n} stations{fuel} ont publié un prix {where} {_period(meta)}."
        ans.data = {"n_stations": n}
    elif tool.name == "motorway_gap":
        r = rows[0]
        if r["autoroute"] is None or r["route"] is None:
            ans.grounded = False
            ans.answer = f"Pas de station d'autoroute et de route à comparer {where}."
            return
        gap = r["autoroute"] - r["route"]
        ans.answer = (f"Sur autoroute, le {p.fuel} coûte en moyenne {_fmt_price(r['autoroute'])}"
                      f" contre {_fmt_price(r['route'])} sur route {where} {_period(meta)}{note}"
                      f", soit {_fmt_num(gap * 100, "+.1f")} centimes.")
        ans.data = {"autoroute": r["autoroute"], "route": r["route"], "ecart": gap}



def _compose_anomalies(ans: Answer, p: Parsed, rows: list[dict], meta: dict, where: str) -> None:
    n = meta["n_total"]
    fuel = f" ({p.fuel})" if p.fuel else ""
    top = "; ".join(f"{r['pdv_id']} {r['carburant']} {_fmt_price(r['prix_eur'])} le "
                    f"{r['date_prix']:%d/%m} ({_fmt_num(r['dev_peers'] * 100, '+.1f')} % vs pairs)"
                    for r in rows)
    label = "prix suspect" if n <= 1 else "prix suspects"
    ans.answer = (f"{n} {label}{fuel} {where} {_period(meta)}."
                  + (f" Les plus marqués : {top}." if rows else ""))
    ans.data = {"n_anomalies": n,
                "rows": [{k: (str(v) if k == "date_prix" else v) for k, v in r.items()}
                         for r in rows]}
    ans.grounded = True


def _fmt_num(x: float, spec: str) -> str:
    return format(x, spec).replace(".", ",")


def _where(zone: Zone) -> str:
    if zone.column is None:
        return "en France"
    if zone.column == "ville":
        return f"à {zone.label}"
    if zone.column == "region":
        return f"en {zone.label}"
    if zone.column == "dep":
        return f"dans le {zone.label}"
    return f"dans le département {zone.label}"


class Agent:
    def __init__(self, paths: Paths | None = None, min_score: float = 3.0, min_terms: int = 1,
                 llm_fn: Callable[[str, dict], str] | None = None):
        self.paths = paths or get_paths()
        self.tool = SqlTool(self.paths)
        self.retriever = Retriever()
        self.resolver = ZoneResolver(self.tool.gazetteer())
        g = StateGraph(State)
        g.add_node("route", partial(route_node, resolver=self.resolver))
        g.add_node("sql", partial(sql_node, tool=self.tool))
        g.add_node("rag", partial(rag_node, retriever=self.retriever))
        g.add_node("compose", partial(compose_node, min_score=min_score, min_terms=min_terms,
                                      llm_fn=llm_fn))
        g.add_edge(START, "route")
        g.add_conditional_edges(
            "route", lambda s: "rag" if s["parsed"].intent == "doc" else "sql",
            {"rag": "rag", "sql": "sql"})
        g.add_edge("sql", "compose")
        g.add_edge("rag", "compose")
        g.add_edge("compose", END)
        self.graph = g.compile()

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        out = self.graph.invoke({"question": question})["answer"]
        out.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return out
