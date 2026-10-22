"""Shared matplotlib style for all plots (colors from a colorblind-checked palette)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # blue, orange, aqua, yellow (validated order)
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
    })
    return plt
