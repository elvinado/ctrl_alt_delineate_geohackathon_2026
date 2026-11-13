"""Shared figure style so every stage speaks the same visual language.

Colour rules (validated palette, see docs/LEARNING_GUIDE.md):
* Categorical colours carry *one* identity each and are never cycled.  For
  T-Z sources we use at most three slots (blue, orange, aqua) - those three are
  colour-blind-safe as a set.  Wells are told apart with **direct text labels**,
  not with 13 different colours.
* Status colours (good / warning / serious / critical) are reserved for QC
  states and always come with a text label or glyph (never colour alone).
* Seismic uses a diverging blue <-> red map with a neutral grey midpoint.
* Every figure names its units, datum and (for maps) the coordinate caveat.
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")  # file output only; safe in terminals and on servers
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#d9d8d4"
NEUTRAL = "#9a9893"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
          "neutral": "#c9c7c1"}

TZ_STYLE = {
    "measured_checkshot": {"color": SERIES[2], "ls": "-", "label": "approved measured checkshot"},
    "tdr_table": {"color": SERIES[0], "ls": "-", "label": "T-Z table (team export, provenance unverified)"},
    "regional_polynomial": {"color": SERIES[1], "ls": "--", "label": "provisional regional polynomial"},
    "regional_linear_velocity": {"color": SERIES[1], "ls": "--",
                                 "label": "provisional regional V(z)=V0+kz (selected by LOWO)"},
    "historic_polynomial": {"color": TEXT_2, "ls": "-.", "label": "historic polynomial (read-only)"},
    "none": {"color": STATUS["critical"], "ls": ":", "label": "no T-Z available"},
}

# One fixed colour per class, used by every stage (strip logs, bars, probabilities).
# Classes are always labelled in text as well; colour never carries the meaning alone.
CLASS_ORDER = ["shale", "brine_sand", "oil_sand", "gas_sand", "undetermined_sand", "other_rock", "unlabelled"]
CLASS_COLORS = {"shale": "#8a8580", "brine_sand": SERIES[0], "oil_sand": SERIES[5], "gas_sand": SERIES[7],
                "undetermined_sand": SERIES[3], "other_rock": SERIES[6], "unlabelled": "#ecebe8"}
LITH_ORDER = ["sand", "heterolithic", "shale", "tight_nonshale", "not_interpretable"]
LITH_COLORS = {"sand": "#f0cf7a", "heterolithic": "#c9b79c", "shale": "#8a8580",
               "tight_nonshale": SERIES[6], "not_interpretable": "#ecebe8"}
FLUID_COLORS = {"water_bearing": SERIES[0], "hydrocarbon_potential": SERIES[1],
                "unresolved": "#c9c7c1", "conflict": STATUS["critical"]}
GRADE_ORDER = ["B", "C", "U"]
GRADE_COLORS = {"A": "#0b0b0b", "B": "#52514e", "C": "#a9a7a2", "U": "#ecebe8"}

COORD_NOTE = ("X/Y: organiser coordinates, CRS undisclosed (randomised for confidentiality); "
              "wells and seismic stated to share one system. Azimuths assumed grid-referenced.")
DATUM_NOTE = "Depth: TVDSS = TVD below RT minus RT elevation (m below MSL, positive down). TWT datum: MSL."

SEISMIC_CMAP = LinearSegmentedColormap.from_list(
    "seismic_bwr_grey", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"])


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.8,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "legend.fontsize": 9, "legend.frameon": False,
        "lines.linewidth": 1.6, "axes.spines.top": False, "axes.spines.right": False,
    })


def footer(fig, text: str, y: float = 0.005) -> None:
    fig.text(0.01, y, text, fontsize=7.5, color=TEXT_2, ha="left", va="bottom", wrap=True)


def not_available_figure(title: str, reasons: list[str], subtitle: str = "", size=(12.0, 5.0)):
    """An honest placeholder: big NOT AVAILABLE + the exact reasons.  No fake data."""
    setup_style()
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.5, 0.86, title, ha="center", va="center", fontsize=13, fontweight="bold")
    if subtitle:
        ax.text(0.5, 0.78, subtitle, ha="center", va="center", fontsize=10, color=TEXT_2)
    ax.text(0.5, 0.62, "NOT AVAILABLE", ha="center", va="center", fontsize=28, fontweight="bold",
            color=STATUS["critical"])
    body = "\n".join("- " + textwrap.fill(r, 110, subsequent_indent="  ") for r in reasons)
    ax.text(0.08, 0.48, "Reasons:\n" + body, ha="left", va="top", fontsize=10, family="monospace")
    ax.text(0.5, 0.04, "This panel intentionally contains no synthetic or invented data.",
            ha="center", fontsize=8.5, color=TEXT_2)
    return fig


def status_text(ax, x, y, text: str, level: str, **kw) -> None:
    """Text inside a status-coloured box (label + colour, never colour alone)."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=kw.pop("fontsize", 8.5),
            bbox={"boxstyle": "round,pad=0.3", "facecolor": STATUS.get(level, STATUS["neutral"]),
                  "edgecolor": "none", "alpha": 0.9}, **kw)
