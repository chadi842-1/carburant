"""SYNTHETIC source that writes daily files in the official XML format.

The official feed is not reachable from the environment this project was built in, so the
pipeline is exercised end to end on synthetic files that follow the documented format exactly
(see ingest.py). Point `carburant ingest --url ...` at the real feed to use real data.

Price model, per station s and fuel f on day t:

    price = national_f(t) + regional_effect(dep) + station_effect(s, f) + motorway(s) + noise

national_f(t) is a random walk around a plausible level; a station re-prices a given fuel on
about 35% of days. Stations are spread over 20 départements (geo.py).

Data-quality defects are injected on purpose, and every injected row is recorded in
lake/_truth/defects.parquet, so the quality layer can be scored on what it catches:

    duplicate      the same <prix> element published again (same file or the next day's file)
    price_typo     decimal slip, e.g. 17.99 instead of 1.799
    zero_price     valeur="0"
    missing_price  valeur=""
    bad_timestamp  maj that is not a valid date
    unknown_fuel   a fuel name outside the official catalogue, with no id
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import pandas as pd

from .config import FUELS, SEED, Paths
from .geo import DEPARTEMENTS
from .ingest import raw_path

BASE_PRICE = {"Gazole": 1.70, "SP95": 1.80, "E10": 1.77, "SP98": 1.88, "E85": 0.86, "GPLc": 0.99}
OFFER_PROBA = {"Gazole": 0.99, "SP95": 0.55, "E10": 0.90, "SP98": 0.80, "E85": 0.30, "GPLc": 0.15}
MOTORWAY_SHARE, MOTORWAY_PREMIUM = 0.08, 0.15
REPRICE_PROBA = 0.35

DEFECT_RATES = {
    "duplicate": 0.006,
    "price_typo": 0.002,
    "zero_price": 0.001,
    "missing_price": 0.001,
    "bad_timestamp": 0.001,
    "unknown_fuel": 0.0005,
}
STREETS = ["avenue de la République", "route nationale", "boulevard Jean Jaurès", "rue de Paris",
           "avenue du Général de Gaulle", "zone commerciale", "rue Victor Hugo", "route de Lyon",
           "aire de service", "avenue Pasteur"]


@dataclass
class SyntheticRun:
    days: int
    n_stations: int
    n_prix_elements: int
    defects: pd.DataFrame


def make_stations(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    weights = np.array([3 if d["dep"] in {"75", "13", "69", "59"} else 1 for d in DEPARTEMENTS])
    deps = rng.choice(len(DEPARTEMENTS), size=n, p=weights / weights.sum())
    for i, k in enumerate(deps):
        d = DEPARTEMENTS[k]
        motorway = rng.random() < MOTORWAY_SHARE
        fuels = [f for f in FUELS.values() if rng.random() < OFFER_PROBA[f]] or ["Gazole"]
        rows.append({
            "pdv_id": f"{int(d['dep'])}{i:06d}",
            "dep": d["dep"],
            "cp": f"{d['dep']}{rng.integers(0, 1000):03d}" if d["dep"] != "75"
                  else f"750{rng.integers(1, 21):02d}",
            "ville": d["ville"],
            "lat": d["lat"] + rng.normal(0, 0.08),
            "lon": d["lon"] + rng.normal(0, 0.10),
            "pop": "A" if motorway else "R",
            "adresse": f"{rng.integers(1, 300)} {STREETS[rng.integers(len(STREETS))]}",
            "fuels": fuels,
            "effect": d["price_effect"] + (MOTORWAY_PREMIUM if motorway else 0.0),
            "station_effect": {f: rng.normal(0, 0.03) for f in fuels},
        })
    return pd.DataFrame(rows)


def national_paths(days: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    common = np.cumsum(rng.normal(0, 0.004, days))          # shared crude-oil shock
    return {f: BASE_PRICE[f] + common + np.cumsum(rng.normal(0, 0.0015, days))
            for f in FUELS.values()}


def _prix_el(nom: str | None, fid: str | None, maj: str, valeur: str) -> str:
    attrs = []
    if nom is not None:
        attrs.append(f"nom={quoteattr(nom)}")
    if fid is not None:
        attrs.append(f"id={quoteattr(fid)}")
    attrs += [f"maj={quoteattr(maj)}", f"valeur={quoteattr(valeur)}"]
    return f"<prix {' '.join(attrs)}/>"


def _valeur(price: float, decimal_format: bool) -> str:
    return f"{price:.3f}" if decimal_format else str(int(round(price * 1000)))


def generate(
    paths: Paths,
    n_stations: int = 400,
    days: int = 90,
    start: date = date(2026, 6, 1),
    seed: int = SEED,
    defect_rates: dict[str, float] | None = None,
) -> SyntheticRun:
    """Write `days` daily XML files into the raw zone and the defect manifest into lake/_truth."""
    paths.ensure()
    rates = DEFECT_RATES if defect_rates is None else defect_rates
    rng = np.random.default_rng(seed)
    st = make_stations(n_stations, rng)
    nat = national_paths(days, rng)
    current: dict[tuple[str, str], float] = {}
    carry_dups: list[tuple[str, str]] = []     # (pdv_id, element) re-published next day
    defects, n_elements = [], 0
    fid_of = {v: str(k) for k, v in FUELS.items()}

    for t in range(days):
        day = start + timedelta(days=t)
        # the feed switched from x1000 integers to decimals at some point; mix both formats
        decimal_format = t >= days // 2
        body: dict[str, list[str]] = {pid: [] for pid in st["pdv_id"]}
        for pid, el in carry_dups:
            body[pid].append(el)
            n_elements += 1
        carry_dups = []
        for s in st.itertuples():
            for f in s.fuels:
                key = (s.pdv_id, f)
                if key in current and rng.random() > REPRICE_PROBA:
                    continue
                price = nat[f][t] + s.effect + s.station_effect[f] + rng.normal(0, 0.01)
                price = round(float(price), 3)
                current[key] = price
                ts = datetime.combine(day, datetime.min.time()) + timedelta(
                    hours=int(rng.integers(5, 22)), minutes=int(rng.integers(0, 60)))
                maj = ts.strftime("%Y-%m-%d %H:%M:%S")
                nom, fid, valeur = f, fid_of[f], _valeur(price, decimal_format)
                kind = None
                u = rng.random()
                cum = 0.0
                for k, r in rates.items():
                    cum += r
                    if u < cum:
                        kind = k
                        break
                if kind == "price_typo":
                    valeur = f"{price * 10:.2f}"
                elif kind == "zero_price":
                    valeur = "0"
                elif kind == "missing_price":
                    valeur = ""
                elif kind == "bad_timestamp":
                    maj = f"{day.year}-13-45 99:{rng.integers(10, 60)}:00"
                elif kind == "unknown_fuel":
                    nom, fid = "Superéthanol premium", None
                el = _prix_el(nom, fid, maj, valeur)
                body[s.pdv_id].append(el)
                n_elements += 1
                if kind is not None and kind != "duplicate":
                    defects.append({"pdv_id": s.pdv_id, "carburant": nom, "maj_raw": maj,
                                    "valeur_raw": valeur, "defect": kind, "day": day})
                if kind == "duplicate":
                    defects.append({"pdv_id": s.pdv_id, "carburant": nom, "maj_raw": maj,
                                    "valeur_raw": valeur, "defect": kind, "day": day})
                    if rng.random() < 0.5 and t + 1 < days:
                        carry_dups.append((s.pdv_id, el))
                    else:
                        body[s.pdv_id].append(el)
                        n_elements += 1
        _write_day(paths, day, st, body)

    truth = paths.lake / "_truth"
    truth.mkdir(parents=True, exist_ok=True)
    defects_df = pd.DataFrame(defects, columns=["pdv_id", "carburant", "maj_raw", "valeur_raw",
                                                "defect", "day"])
    defects_df.to_parquet(truth / "defects.parquet", index=False)
    st.drop(columns=["fuels", "station_effect"]).to_parquet(truth / "stations.parquet",
                                                            index=False)
    return SyntheticRun(days, n_stations, n_elements, defects_df)


def _write_day(paths: Paths, day: date, st: pd.DataFrame, body: dict[str, list[str]]) -> None:
    out = ['<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>', "<pdv_liste>"]
    for s in st.itertuples():
        out.append(
            f'<pdv id="{s.pdv_id}" latitude="{round(s.lat * 100_000)}" '
            f'longitude="{round(s.lon * 100_000)}" cp="{s.cp}" pop="{s.pop}">'
            f"<adresse>{escape(s.adresse)}</adresse><ville>{escape(s.ville)}</ville>"
            + "".join(body[s.pdv_id]) + "</pdv>")
    out.append("</pdv_liste>")
    f = raw_path(paths, day.isoformat())
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes("\n".join(out).encode("iso-8859-1"))
