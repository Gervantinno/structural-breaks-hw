"""Common plotting style for the notebook"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
BLUE = "#2563eb"
RED = "#dc2626"
ORANGE = "#ea7c1f"
GREEN = "#16a34a"
PURPLE = "#7c3aed"
REGIME_FILLS = ["#dbeafe", "#fde8d7", "#dcfce7", "#ede9fe", "#fef3c7", "#fce7f3", "#e0f2fe"]


def set_style():
    plt.rcParams.update({
        "figure.figsize": (10, 4.2), "figure.dpi": 110, "savefig.dpi": 200,
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "lines.linewidth": 1.4,
    })


def savefig(fig, name: str, folder="../figures"):
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    fig.savefig(p / f"{name}.png", bbox_inches="tight", facecolor="white")


def shade_regimes(ax, x, ends, alpha=1.0):
    """Shades the regimes between breaks; ends are the right (exclusive) boundaries."""
    bounds = [0] + list(ends) + [len(x)]
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1] - 1
        ax.axvspan(x[a], x[b], color=REGIME_FILLS[j % len(REGIME_FILLS)], alpha=alpha, lw=0, zorder=0)


def plot_regime_means(ax, x, y, ends, means, color=RED, label="regime mean"):
    bounds = [0] + list(ends) + [len(x)]
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1] - 1
        ax.hlines(means[j], x[a], x[b], color=color, lw=2.4, label=label if j == 0 else None, zorder=3)


def plot_cusum(ax, res, x, title):
    xs = np.asarray(x)[res.t_index]
    ax.plot(xs, res.path, color=BLUE, label="statistic")
    ax.plot(xs, res.upper, color=RED, ls="--", lw=1, label=f"{int((1 - res.alpha) * 100)}% boundary")
    ax.plot(xs, res.lower, color=RED, ls="--", lw=1)
    verdict = "stability rejected" if res.reject else "stability not rejected"
    ax.set_title(f"{title}: {verdict}")
    lo, hi = min(res.lower.min(), res.path.min()), max(res.upper.max(), res.path.max())
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.30 * (hi - lo))   # room for the legend
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    if np.issubdtype(np.asarray(xs).dtype, np.datetime64):
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
