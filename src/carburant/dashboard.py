"""Static dashboard figure from mart_dashboard (the table a BI report would read)."""
from __future__ import annotations

import pandas as pd

from .config import Paths
from .warehouse import connect

FUELS_SHOWN = ["Gazole", "E10", "SP98"]


def plot_dashboard(paths: Paths, out_path=None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from .viz import COLORS, INK, INK_2, apply_style, title
    from .viz import GRID as GRID_LINE

    with connect(paths, read_only=True) as con:
        daily = con.execute("""
            select date_prix, carburant,
                   sum(prix_moyen * n_stations) / sum(n_stations) as prix
            from mart_dashboard group by all order by date_prix""").df()
        last = con.execute("select max(date_prix) from mart_dashboard").fetchone()[0]
        regions = con.execute("""
            select region, sum(prix_moyen * n_stations) / sum(n_stations) as prix
            from mart_dashboard
            where carburant = 'Gazole' and date_prix > ? - interval 7 day
            group by region order by prix""", [last]).df()
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for f in FUELS_SHOWN:
        s = daily[daily["carburant"] == f]
        ax.plot(pd.to_datetime(s["date_prix"]), s["prix"], color=COLORS[f], lw=2)
        ax.annotate(f, (pd.to_datetime(s["date_prix"].iloc[-1]), s["prix"].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_ylabel("EUR / litre")
    ax.margins(x=0.08)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    title(ax, "National average price per day", "station-weighted, valid prices only")

    ax = axes[1]                       # dot plot: no zero baseline needed, unlike bars
    y = range(len(regions))
    ax.hlines(y, regions["prix"].min() - 0.01, regions["prix"], color=GRID_LINE, lw=1)
    ax.plot(regions["prix"], y, "o", color=COLORS["Gazole"], ms=7)
    for yv, v in zip(y, regions["prix"], strict=True):
        ax.text(v + 0.003, yv, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(regions["region"], fontsize=8)
    ax.set_xlim(regions["prix"].min() - 0.01, regions["prix"].max() + 0.02)
    ax.grid(axis="y", visible=False)
    title(ax, "Gazole by region, last 7 days", "EUR / litre, station-weighted mean")
    fig.tight_layout()
    out_path = out_path or paths.reports / "dashboard.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {"days": int(daily["date_prix"].nunique()), "regions": int(len(regions))}
