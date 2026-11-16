"""Shared matplotlib styling for the plotting tools."""
from __future__ import annotations

import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c9c8c2", "axes.grid": True, "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6, "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    })
