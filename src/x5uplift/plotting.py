"""Shared figure style for every notebook: one palette, one look, one save function."""

import matplotlib as mpl
import matplotlib.pyplot as plt

from x5uplift.config import FIGURES

# Categorical colours, always used in this order. The set was checked for colour
# vision deficiency (worst adjacent pair still clearly distinguishable).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW]

# Fixed roles, the same in every figure of the project
TREATMENT = BLUE
CONTROL = ORANGE

# Ink and chart chrome
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL_FILL = "#e9e8e3"


def set_style() -> None:
    """Apply the project style to all matplotlib figures created afterwards.

    Starts from matplotlib's defaults, so a style set by another library (causalml switches to
    "fivethirtyeight" on import) does not leak into the project figures.
    """
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.labelsize": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 1,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "figure.dpi": 110,
        }
    )


def titles(ax, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title that states the finding, with an optional subtitle that says what is plotted."""
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK,
                 pad=30 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10, color=INK_2)


def save(fig, name: str) -> None:
    """Save a figure as PNG into figures/ (used for the README and the notebooks)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
