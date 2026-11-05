"""Read-only SQL tool over the warehouse: fixed, parameterised queries only (no free SQL).

Time is relative to the data: "today" is the last date present in mart_prix_journalier, and a
window of N days is (today - N, today].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import duckdb

from .config import Paths

ZONE_COLUMNS = ("ville", "departement", "dep", "region")


@dataclass(frozen=True)
class Zone:
    column: str | None = None       # None = France entière
    value: str | None = None
    label: str = "France"


@dataclass
class ToolResult:
    name: str
    sql: str
    params: list
    rows: list[dict]
    meta: dict = field(default_factory=dict)


def _zone_sql(zone: Zone) -> tuple[str, list]:
    if zone.column is None:
        return "", []
    if zone.column not in ZONE_COLUMNS:
        raise ValueError(f"unsupported zone column {zone.column}")
    return f" and {zone.column} = ?", [zone.value]


class SqlTool:
    def __init__(self, paths: Paths):
        self.paths = paths

    def _query(self, sql: str, params: list) -> list[dict]:
        with duckdb.connect(str(self.paths.warehouse), read_only=True) as con:
            cur = con.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    # ----------------------------------------------------------------- reference data
    def last_date(self) -> date:
        return self._query("select max(date_prix) as d from mart_prix_journalier", [])[0]["d"]

    def gazetteer(self) -> dict[str, list[str]]:
        rows = self._query("select distinct ville, departement, dep, region "
                           "from mart_prix_journalier", [])
        return {c: sorted({r[c] for r in rows if r[c]}) for c in ZONE_COLUMNS}

    def window(self, days: int, end: date | None = None) -> tuple[date, date]:
        end = end or self.last_date()
        return end - timedelta(days=days), end

    # -------------------------------------------------------------------------- tools
    def extreme(self, fuel: str, zone: Zone, days: int, k: int = 1,
                cheapest: bool = True, end: date | None = None) -> ToolResult:
        """Stations ranked by their latest price in the window."""
        start, end = self.window(days, end)
        zsql, zp = _zone_sql(zone)
        order = "asc" if cheapest else "desc"
        sql = f"""
            with w as (
                select * from mart_prix_journalier
                where carburant = ? and date_prix > ? and date_prix <= ?{zsql}
            ), latest as (
                select *, row_number() over (partition by pdv_id order by date_prix desc) as rn
                from w
            )
            select pdv_id, ville, departement, autoroute, prix_eur, date_prix
            from latest where rn = 1
            order by prix_eur {order}, pdv_id
            limit ?"""
        params = [fuel, start, end, *zp, k]
        return ToolResult("cheapest" if cheapest else "most_expensive", sql, params,
                          self._query(sql, params), {"start": start, "end": end})

    def average(self, fuel: str, zone: Zone, days: int, end: date | None = None) -> ToolResult:
        """Mean of the daily prices observed in the window, and how many stations they cover."""
        start, end = self.window(days, end)
        zsql, zp = _zone_sql(zone)
        sql = f"""
            select avg(prix_eur) as prix_moyen, count(*) as n_releves,
                   count(distinct pdv_id) as n_stations
            from mart_prix_journalier
            where carburant = ? and date_prix > ? and date_prix <= ?{zsql}"""
        params = [fuel, start, end, *zp]
        return ToolResult("average", sql, params, self._query(sql, params),
                          {"start": start, "end": end})

    def trend(self, fuel: str, zone: Zone, days: int, end: date | None = None) -> ToolResult:
        """Average of the first 7 days of the window vs the last 7 days."""
        start, end = self.window(days, end)
        zsql, zp = _zone_sql(zone)
        sql = f"""
            select
                avg(prix_eur) filter (where date_prix <= ?) as debut,
                avg(prix_eur) filter (where date_prix > ?)  as fin
            from mart_prix_journalier
            where carburant = ? and date_prix > ? and date_prix <= ?{zsql}"""
        params = [start + timedelta(days=7), end - timedelta(days=7), fuel, start, end, *zp]
        return ToolResult("trend", sql, params, self._query(sql, params),
                          {"start": start, "end": end})

    def count_stations(self, zone: Zone, days: int, fuel: str | None = None,
                       end: date | None = None) -> ToolResult:
        start, end = self.window(days, end)
        zsql, zp = _zone_sql(zone)
        fsql, fp = (" and carburant = ?", [fuel]) if fuel else ("", [])
        sql = f"""
            select count(distinct pdv_id) as n_stations
            from mart_prix_journalier
            where date_prix > ? and date_prix <= ?{zsql}{fsql}"""
        params = [start, end, *zp, *fp]
        return ToolResult("count_stations", sql, params, self._query(sql, params),
                          {"start": start, "end": end})

    def motorway_gap(self, fuel: str, zone: Zone, days: int,
                     end: date | None = None) -> ToolResult:
        start, end = self.window(days, end)
        zsql, zp = _zone_sql(zone)
        sql = f"""
            select avg(prix_eur) filter (where autoroute)     as autoroute,
                   avg(prix_eur) filter (where not autoroute) as route
            from mart_prix_journalier
            where carburant = ? and date_prix > ? and date_prix <= ?{zsql}"""
        params = [fuel, start, end, *zp]
        return ToolResult("motorway_gap", sql, params, self._query(sql, params),
                          {"start": start, "end": end})

    def anomalies(self, zone: Zone, days: int, fuel: str | None = None, k: int = 3,
                  end: date | None = None) -> ToolResult:
        start, end = self.window(days, end)
        zsql, zp = _zone_sql(zone)
        if zone.column in ("departement", "region"):   # not in anomalies_prix: map via the mart
            zsql = (" and dep in (select distinct dep from mart_prix_journalier "
                    f"where {zone.column} = ?)")
        fsql, fp = (" and carburant = ?", [fuel]) if fuel else ("", [])
        exists = self._query("select count(*) as n from information_schema.tables "
                             "where table_schema = 'ml' and table_name = 'anomalies_prix'", [])
        if not exists[0]["n"]:
            return ToolResult("anomalies", "", [], [],
                              {"start": start, "end": end, "n_total": 0, "missing": True})
        base = f"from ml.anomalies_prix where date_prix > ? and date_prix <= ?{zsql}{fsql}"
        params = [start, end, *zp, *fp]
        n = self._query(f"select count(*) as n {base}", params)[0]["n"]
        sql = (f"select date_prix, pdv_id, ville, carburant, prix_eur, dev_peers, score {base} "
               f"order by score desc, pdv_id limit ?")
        rows = self._query(sql, [*params, k])
        return ToolResult("anomalies", sql, [*params, k], rows,
                          {"start": start, "end": end, "n_total": n})
