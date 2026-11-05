"""Evaluate the agent on eval/questions.json.

Expected answers for data questions are computed by an independent pandas oracle over the
warehouse tables (it shares no code with the SQL tool). Scoring:

- data: right intent AND the numbers match the oracle (same stations in the same order for
  rankings; |difference| < 0.0005 EUR for prices; exact for counts). If the oracle finds no data,
  the agent is right only if it also returns no rows.
- doc: the top cited section is one of the accepted sections.
- out_of_scope: the agent abstains.

Also reported: grounding (share of answered questions that cite a source), hallucinations
(out-of-scope questions answered), and latency (median / p95, warm, in-process).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .agent import Agent
from .config import Paths, get_paths

EVAL_FILE = Path(__file__).resolve().parents[2] / "eval" / "questions.json"          # dev set
TEST_FILE = EVAL_FILE.with_name("questions_test_v2.json")                          # held out
TOL = 5e-4


class Oracle:
    def __init__(self, paths: Paths):
        with duckdb.connect(str(paths.warehouse), read_only=True) as con:
            self.daily = con.execute("select * from mart_prix_journalier").df()
            self.flags = con.execute("select * from ml.anomalies_prix").df()
        self.daily["date_prix"] = pd.to_datetime(self.daily["date_prix"])
        self.flags["date_prix"] = pd.to_datetime(self.flags["date_prix"])
        self.end = self.daily["date_prix"].max()

    def _window(self, df: pd.DataFrame, spec: dict) -> pd.DataFrame:
        start = self.end - pd.Timedelta(days=spec["days"])
        w = df[(df["date_prix"] > start) & (df["date_prix"] <= self.end)]
        if spec.get("fuel"):
            w = w[w["carburant"] == spec["fuel"]]
        if spec.get("autoroute"):
            w = w[w["autoroute"]]
        return w

    def _zone(self, df: pd.DataFrame, zone) -> pd.DataFrame:
        if not zone:
            return df
        col, val = zone
        if col in df.columns:
            return df[df[col] == val]
        deps = set(self.daily.loc[self.daily[col] == val, "dep"])     # flags lack region/dept
        return df[df["dep"].isin(deps)]

    def answer(self, spec: dict) -> dict:
        op = spec["op"]
        if op == "anomalies":
            w = self._zone(self._window(self.flags, spec), spec.get("zone"))
            return {"n_anomalies": int(len(w))}
        w = self._zone(self._window(self.daily, spec), spec.get("zone"))
        if op in ("cheapest", "most_expensive"):
            latest = w.sort_values("date_prix").groupby("pdv_id").tail(1)
            latest = latest.sort_values(["prix_eur", "pdv_id"],
                                        ascending=[op == "cheapest", True])
            return {"rows": [{"pdv_id": r.pdv_id, "prix_eur": r.prix_eur}
                             for r in latest.head(spec.get("k", 1)).itertuples()]}
        if op == "average":
            return {"prix_moyen": float(w["prix_eur"].mean()) if len(w) else None}
        if op == "trend":
            start = self.end - pd.Timedelta(days=max(spec["days"], 14))
            ww = self._zone(self._window(self.daily, {**spec, "days": max(spec["days"], 14)}),
                            spec.get("zone"))
            a = ww[ww["date_prix"] <= start + pd.Timedelta(days=7)]["prix_eur"].mean()
            b = ww[ww["date_prix"] > self.end - pd.Timedelta(days=7)]["prix_eur"].mean()
            return {"delta": float(b - a) if len(ww) else None}
        if op == "count_stations":
            return {"n_stations": int(w["pdv_id"].nunique())}
        if op == "motorway_gap":
            a = w.loc[w["autoroute"], "prix_eur"].mean()
            r = w.loc[~w["autoroute"], "prix_eur"].mean()
            return {"autoroute": float(a), "route": float(r)}
        raise ValueError(op)


def _close(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) < TOL


def score_data(spec: dict, expected: dict, ans) -> tuple[bool, str]:
    op = spec["op"]
    if ans.intent != op:
        return False, f"routed to {ans.intent}"
    d = ans.data
    if op in ("cheapest", "most_expensive"):
        got = [(r["pdv_id"], r["prix_eur"]) for r in d.get("rows", [])]
        want = [(r["pdv_id"], r["prix_eur"]) for r in expected["rows"]]
        if not want:
            return (not got, "oracle empty")
        ok = len(got) == len(want) and all(g[0] == w[0] and _close(g[1], w[1])
                                           for g, w in zip(got, want, strict=False))
        return ok, "" if ok else f"got {got} want {want}"
    if op == "average":
        ok = _close(d.get("prix_moyen"), expected["prix_moyen"])
        return ok, "" if ok else f"got {d.get('prix_moyen')} want {expected['prix_moyen']}"
    if op == "trend":
        ok = _close(d.get("delta"), expected["delta"])
        return ok, "" if ok else f"got {d.get('delta')} want {expected['delta']}"
    if op == "count_stations":
        ok = d.get("n_stations") == expected["n_stations"]
        return ok, "" if ok else f"got {d.get('n_stations')} want {expected['n_stations']}"
    if op == "motorway_gap":
        ok = _close(d.get("autoroute"), expected["autoroute"]) and _close(
            d.get("route"), expected["route"])
        return ok, "" if ok else f"got {d} want {expected}"
    if op == "anomalies":
        ok = d.get("n_anomalies") == expected["n_anomalies"]
        return ok, "" if ok else f"got {d.get('n_anomalies')} want {expected['n_anomalies']}"
    raise ValueError(op)


def run_eval(paths: Paths | None = None, eval_file: Path = EVAL_FILE,
             agent: Agent | None = None, out_name: str = "agent_eval.json") -> dict:
    paths = paths or get_paths()
    agent = agent or Agent(paths)
    oracle = Oracle(paths)
    questions = json.loads(eval_file.read_text(encoding="utf-8"))["questions"]
    agent.ask("warm-up question about gazole à Paris")
    rows = []
    for q in questions:
        ans = agent.ask(q["question"])
        if q["kind"] == "data":
            ok, why = score_data(q["spec"], oracle.answer(q["spec"]), ans)
        elif q["kind"] == "doc":
            top = ans.sources[0].ref if ans.sources and ans.sources[0].kind == "doc" else None
            ok = top in q["accept"]
            why = "" if ok else f"top source {top} (intent {ans.intent})"
        else:
            ok = ans.intent == "abstain"
            why = "" if ok else f"answered as {ans.intent}"
        rows.append({"id": q["id"], "kind": q["kind"], "question": q["question"],
                     "correct": bool(ok), "why": why, "intent": ans.intent,
                     "answered": ans.intent != "abstain", "cited": bool(ans.sources),
                     "latency_ms": ans.latency_ms, "answer": ans.answer})
    df = pd.DataFrame(rows)
    answered = df[df["answered"]]
    lat = df["latency_ms"].to_numpy()
    summary = {
        "n_questions": int(len(df)),
        "accuracy": round(float(df["correct"].mean()), 4),
        "accuracy_by_kind": {k: {"n": int(len(g)), "correct": int(g["correct"].sum()),
                                 "accuracy": round(float(g["correct"].mean()), 4)}
                             for k, g in df.groupby("kind")},
        "grounded_rate": round(float(answered["cited"].mean()), 4) if len(answered) else None,
        "hallucinated_out_of_scope": int(((df["kind"] == "out_of_scope") & df["answered"]).sum()),
        "latency_ms": {"median": round(float(statistics.median(lat)), 1),
                       "p95": round(float(np.quantile(lat, 0.95)), 1),
                       "max": round(float(lat.max()), 1)},
        "failures": df.loc[~df["correct"], ["id", "question", "why"]].to_dict("records"),
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / out_name).write_text(
        json.dumps({"summary": summary, "questions": rows}, indent=2, ensure_ascii=False,
                   default=str), encoding="utf-8")
    return summary


def compare_routers(paths: Paths | None = None) -> dict:
    """Both routers on the development set and on the held-out test set.

    Writes reports/agent_eval_{router}_{set}.json and reports/agent_eval_compare.json.
    """
    from .intent import train

    paths = paths or get_paths()
    model = train()
    out = {}
    for router in ("rules", "ml"):
        agent = Agent(paths, router=router, intent_model=model if router == "ml" else None)
        for name, f in (("dev", EVAL_FILE), ("test", TEST_FILE)):
            s = run_eval(paths, f, agent, out_name=f"agent_eval_{router}_{name}.json")
            out[f"{router}/{name}"] = s
    (paths.reports / "agent_eval_compare.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out
