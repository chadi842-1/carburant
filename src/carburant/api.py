"""FastAPI service.  Run:  uvicorn carburant.api:app --port 8000

POST /ask         question in French -> answer, structured data, sources, latency
GET  /cheapest    direct access to the SQL tool (fuel, optional city, days, k)
GET  /anomalies   suspect prices flagged by the detector
GET  /quality     last data-quality report
GET  /health      liveness;  GET /ready: the warehouse can be queried
"""
from __future__ import annotations

import json
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import __version__
from .agent import Agent, Answer
from .config import FUELS, get_paths
from .tools import Zone

app = FastAPI(title="Carburant: fuel prices data platform and agent", version=__version__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    paths = get_paths()
    if not paths.warehouse.exists():
        raise FileNotFoundError(f"No warehouse at {paths.warehouse}. Run `carburant run` first.")
    return Agent(paths)


def _agent() -> Agent:
    try:
        return get_agent()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> dict:
    agent = _agent()
    return {"status": "ready", "last_date": str(agent.tool.last_date())}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    return _agent().ask(req.question)


@app.get("/cheapest")
def cheapest(fuel: str = Query(..., description="Gazole, SP95, SP98, E10, E85 or GPLc"),
             ville: str | None = None, days: int = Query(7, ge=1, le=365),
             k: int = Query(5, ge=1, le=50)) -> dict:
    if fuel not in FUELS.values():
        raise HTTPException(422, f"unknown fuel {fuel!r}; expected one of {list(FUELS.values())}")
    zone = Zone("ville", ville, ville) if ville else Zone()
    res = _agent().tool.extreme(fuel, zone, days, k, cheapest=True)
    return {"window": {k_: str(v) for k_, v in res.meta.items()},
            "stations": [{k_: (str(v) if k_ == "date_prix" else v) for k_, v in r.items()}
                         for r in res.rows]}


@app.get("/anomalies")
def anomalies(days: int = Query(7, ge=1, le=365), fuel: str | None = None,
              k: int = Query(10, ge=1, le=100)) -> dict:
    res = _agent().tool.anomalies(Zone(), days, fuel, k)
    if res.meta.get("missing"):
        raise HTTPException(503, "anomaly detection has not been run on this warehouse")
    return {"total": res.meta["n_total"],
            "top": [{k_: (str(v) if k_ == "date_prix" else v) for k_, v in r.items()}
                    for r in res.rows]}


@app.get("/quality")
def quality() -> dict:
    f = get_paths().reports / "quality.json"
    if not f.exists():
        raise HTTPException(404, "no quality report yet")
    return json.loads(f.read_text())["quality"]
