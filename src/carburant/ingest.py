"""Ingestion: fetch raw XML into the raw zone, and parse it into typed staging tables.

The parser targets the official format published on data.gouv.fr ("Prix des carburants en
France"): <pdv_liste><pdv id latitude longitude cp pop><adresse/><ville/>
<prix nom id maj valeur/>...</pdv></pdv_liste>, ISO-8859-1 encoded.

Two format details handled explicitly:
- coordinates are published multiplied by 100000; older/newer files sometimes carry decimals;
- `valeur` is euros x 1000 in the documented format, but recent files use decimal euros.

Parsing is deliberately PERMISSIVE: anything that is structurally readable is kept in staging
(with nulls where a field cannot be parsed) so that the data-quality layer (dbt) can quarantine it
with an explicit reason, and the share of invalid rows can be measured.
"""
from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from lxml import etree

from .config import FUEL_IDS, Paths

MAJ_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M")


# ------------------------------------------------------------------------- field parsing
def parse_coordinate(raw: str | None) -> float | None:
    """'4620114' -> 46.20114 ; '46.20114' -> 46.20114 ; junk -> None."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v / 100_000 if abs(v) > 1000 else v


def parse_price(raw: str | None) -> float | None:
    """Price in EUR/l. '1799' (x1000) -> 1.799 ; '1.799' -> 1.799 ; junk/empty -> None.

    A decimal value is taken as euros as written, so a typo such as '17.99' survives parsing and
    is caught later by the range test (that is the point: the quality layer must see it).
    """
    if raw is None or str(raw).strip() == "":
        return None
    s = str(raw).strip().replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    if "." not in s and v >= 100:
        return v / 1000
    return v


def parse_maj(raw: str | None) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip()
    for fmt in MAJ_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ------------------------------------------------------------------------------- parsing
@dataclass
class ParseResult:
    stations: pd.DataFrame
    prix: pd.DataFrame
    ruptures: pd.DataFrame
    n_pdv: int = 0
    n_prix: int = 0
    warnings: list[str] = field(default_factory=list)


def parse_xml(path: str | Path, source: str | None = None) -> ParseResult:
    """Stream-parse one official XML file into stations / prix / ruptures dataframes."""
    source = source or Path(path).name
    stations, prix, ruptures, warnings = [], [], [], []
    for _, pdv in etree.iterparse(str(path), events=("end",), tag="pdv", recover=True):
        pid = pdv.get("id")
        if not pid:
            warnings.append("pdv without id skipped")
            pdv.clear()
            continue
        stations.append({
            "pdv_id": pid,
            "latitude": parse_coordinate(pdv.get("latitude")),
            "longitude": parse_coordinate(pdv.get("longitude")),
            "cp": (pdv.get("cp") or "").strip() or None,
            "pop": pdv.get("pop"),
            "adresse": (pdv.findtext("adresse") or "").strip() or None,
            "ville": (pdv.findtext("ville") or "").strip() or None,
            "source_file": source,
        })
        for p in pdv.iterfind("prix"):
            nom = p.get("nom")
            fid = p.get("id")
            prix.append({
                "pdv_id": pid,
                "carburant_id": int(fid) if fid and fid.isdigit() else FUEL_IDS.get(nom),
                "carburant": nom,
                "maj_raw": p.get("maj"),
                "maj": parse_maj(p.get("maj")),
                "valeur_raw": p.get("valeur"),
                "prix_eur": parse_price(p.get("valeur")),
                "source_file": source,
            })
        for r in pdv.iterfind("rupture"):
            ruptures.append({"pdv_id": pid, "carburant": r.get("nom"),
                             "debut": parse_maj(r.get("debut")), "source_file": source})
        pdv.clear()
    res = ParseResult(
        stations=pd.DataFrame(stations, columns=[
            "pdv_id", "latitude", "longitude", "cp", "pop", "adresse", "ville", "source_file"]),
        prix=pd.DataFrame(prix, columns=[
            "pdv_id", "carburant_id", "carburant", "maj_raw", "maj", "valeur_raw", "prix_eur",
            "source_file"]),
        ruptures=pd.DataFrame(ruptures, columns=["pdv_id", "carburant", "debut", "source_file"]),
        warnings=warnings,
    )
    res.n_pdv, res.n_prix = len(res.stations), len(res.prix)
    return res


# ----------------------------------------------------------------------------- raw zone
def raw_path(paths: Paths, day: str) -> Path:
    """Partitioned like a Cloud Storage bucket: raw/date=YYYY-MM-DD/prix.xml"""
    return paths.raw / f"date={day}" / "prix.xml"


def fetch(url: str, paths: Paths, day: str, timeout: int = 60) -> Path:
    """Download one published file into the raw zone (http(s):// or file://).

    Zipped files (the official daily/annual downloads) are unpacked so the raw zone holds XML.
    """
    dest = raw_path(paths, day)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".download")
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    if tmp.read_bytes()[:2] == b"PK":
        import zipfile

        with zipfile.ZipFile(tmp) as zf:
            xml_name = next(n for n in zf.namelist() if n.lower().endswith(".xml"))
            dest.write_bytes(zf.read(xml_name))
        tmp.unlink()
    else:
        tmp.replace(dest)
    return dest


def list_raw(paths: Paths) -> list[tuple[str, Path]]:
    out = []
    for d in sorted(paths.raw.glob("date=*")):
        f = d / "prix.xml"
        if f.exists():
            out.append((d.name.split("=", 1)[1], f))
    return out


def stage(paths: Paths) -> dict:
    """Parse every raw file and write lake/staging/{stations,prix,ruptures}.parquet."""
    paths.ensure()
    parts = [parse_xml(f, source=f"{day}/prix.xml") for day, f in list_raw(paths)]
    if not parts:
        raise FileNotFoundError(f"no raw files under {paths.raw}")
    stations = pd.concat([p.stations for p in parts], ignore_index=True)
    prix = pd.concat([p.prix for p in parts], ignore_index=True)
    ruptures = pd.concat([p.ruptures for p in parts], ignore_index=True)
    stations.to_parquet(paths.staging / "stations.parquet", index=False)
    prix.to_parquet(paths.staging / "prix.parquet", index=False)
    ruptures.to_parquet(paths.staging / "ruptures.parquet", index=False)
    return {"files": len(parts), "station_rows": len(stations), "prix_rows": len(prix),
            "rupture_rows": len(ruptures),
            "warnings": sum(len(p.warnings) for p in parts)}
