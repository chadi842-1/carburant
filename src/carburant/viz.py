"""Shared matplotlib style: recessive grid, ink-coloured text, first slots of a validated
categorical palette (blue, orange, aqua)."""
from __future__ import annotations

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = {"robust_rule": BLUE, "isolation_forest": ORANGE, "Gazole": BLUE, "E10": ORANGE,
          "SP98": AQUA}


def apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
    })


def title(ax, main: str, sub: str | None = None) -> None:
    ax.set_title(main, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
