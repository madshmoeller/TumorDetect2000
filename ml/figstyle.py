"""Shared figure style, so all 16 figures read as one system.

Design decisions, and why:

**The figures are not in Comic Sans.** The website's early-2000s costume stops
at the frame. Inside it, the science is presented plainly — a rainbow-gradient
axis label would make a real result look like a joke. What the figures *do*
borrow is the site's paper colour (`--paper #fdf8ec`), so they sit inside the
beveled frames without a white rectangle punching through.

**Colours are computed, not chosen.** Every palette below was run through
`ml/palette_check.py` (a port of the data-viz validator: OKLab ΔE under
Machado-Oliveira-Fernandes CVD simulation, lightness band, chroma floor, WCAG
contrast). Results, on the `#fdf8ec` surface, all-pairs:

- `SERIES` (model vs baselines) — worst CVD ΔE 9.2, worst normal-vision ΔE 24.0,
  0 failures. `SERIES[2]` sits at 2.66:1 contrast, below the 3:1 relief
  threshold, so anything drawn in it carries a direct label.
- `DIFF` (TP / FN / FP overlays, measured against the near-black MRI they are
  drawn on) — worst CVD ΔE 23.6, worst normal-vision ΔE 31.9.

  The GUI's *existing* `--diff-missed #1eb85e` / `--diff-imagined #e62346` pair
  scores deutan ΔE **7.6** — below the 8.0 target. That is red-versus-green
  encoding false-negatives against false-positives, which is the single worst
  pair in this whole project to make indistinguishable. `DIFF` replaces the
  green with blue and leaves the yellow alone.

Sequential magnitude uses a perceptually-uniform matplotlib ramp (`viridis` /
`magma`), never a rainbow.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from . import config as C

# ── palette ────────────────────────────────────────────────────────────────

PAPER = "#fdf8ec"  # site --paper
INK = "#1a1330"  # site --ink
INK_SOFT = "#4a4260"
MUTED = "#8a8398"
GRID = "#e6ddc8"

#: Categorical, fixed order, never cycled. Validated all-pairs.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES_NAMES = ["2.5D U-Net", "FLAIR threshold", "Random forest"]

#: TP / FN / FP. Drawn on the MRI, validated against a near-black surface.
DIFF = {"correct": "#ffd600", "missed": "#3987e5", "imagined": "#e34948"}
DIFF_LABEL = {
    "correct": "Correct (TP)",
    "missed": "Missed (FN)",
    "imagined": "Imagined (FP)",
}

#: Tumour sub-labels. Three-way stack, adjacent-pair validated.
SUBLABEL_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}

SEQUENTIAL = "viridis"
SEQUENTIAL_ALT = "magma"

#: Ground truth contour, on greyscale MRI. Matches the lab's own example figure.
CONTOUR = "#ff2e88"
PREDICTION = "#00e0f0"

FIGURE_TITLES = {
    "F01": "Cohort overview: tumour burden across 60 cases",
    "F02": "Why per-case normalisation is not optional",
    "F03": "One case, four co-registered channels",
    "F04": "Where the tumours are",
    "F05": "Class imbalance: whole volume vs within brain",
    "F06": "The 2.5D U-Net",
    "F07": "Training curves, 3-fold cross-validation",
    "F08": "Choosing the decision threshold",
    "F09": "Per-case Dice: model vs baselines",
    "F10": "Does tumour size predict difficulty?",
    "F11": "Best, median and worst cases",
    "F12": "Volume agreement",
    "F13": "Ablations",
    "F14": "Boundary quality (HD95)",
    "F15": "Error taxonomy",
    "F16": "Stress tests",
}


def apply() -> None:
    """Install the project's rcParams. Call once at the top of a figure script."""
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.dpi": 150,
            "figure.dpi": 110,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SOFT,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SOFT,
            "ytick.labelcolor": INK_SOFT,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK_SOFT,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "text.color": INK,
            "image.cmap": SEQUENTIAL,
            "image.interpolation": "nearest",
        }
    )


#: Characters per line of subtitle, per inch of figure width. Calibrated for
#: 9 pt DejaVu Sans — the wrap has to be deterministic, because the header
#: reserves vertical space based on the resulting line count.
_WRAP_CHARS_PER_INCH = 15.5


def titled(fig, code: str, subtitle: str | None = None) -> None:
    """Standard figure header: bold title, optional explanatory subtitle.

    The subtitle carries the *finding*, not a restatement of the axes — a reader
    who only reads titles should still learn something true.

    Space for the header is reserved in *inches*, then converted to a figure
    fraction, so a tall figure and a short one get the same visual gap and the
    title can never land on top of the first row of axes.
    """
    import textwrap

    title = FIGURE_TITLES.get(code, code)
    w_in, h_in = fig.get_size_inches()

    lines = []
    if subtitle:
        lines = textwrap.wrap(subtitle, width=int(w_in * _WRAP_CHARS_PER_INCH))

    # 0.34 + 0.19/line covers the title + subtitle text itself; +0.34 on top of
    # that is clearance for a per-axes `ax.set_title()` ("a ·" panel labels),
    # whose pad + font ascent sit *above* the axes-top boundary we set below —
    # without it, panel titles overlap the header subtitle.
    header_in = 0.34 + 0.19 * len(lines) + 0.34
    fig.subplots_adjust(top=1 - (header_in + 0.16) / h_in)

    y = 1 - 0.16 / h_in
    fig.text(0.012, y, f"{code} · {title}", ha="left", va="top",
             fontsize=12.5, fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        fig.text(0.012, y - (0.32 + 0.19 * i) / h_in, line,
                 ha="left", va="top", fontsize=9, color=INK_SOFT)


def footnote(fig, text: str) -> None:
    """Provenance line. Every figure states its n and what it was computed on."""
    import textwrap

    w_in, h_in = fig.get_size_inches()
    lines = textwrap.wrap(text, width=int(w_in * 17.5))
    bottom_in = 0.16 + 0.15 * len(lines)
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, bottom_in / h_in))
    for i, line in enumerate(lines):
        fig.text(0.012, (bottom_in - 0.15 * (i + 1)) / h_in, line,
                 ha="left", va="bottom", fontsize=7.5, color=MUTED)


def bare(ax) -> None:
    """Strip a set of axes down for image panels."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)


def save(fig, code: str) -> "object":
    """Write to assets/figures/<code>.png and return the path."""
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / f"{code}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def mask_cmap(color: str, alpha: float = 0.55):
    """A transparent-to-`color` colormap for drawing a binary mask over an image."""
    rgb = mcolors.to_rgb(color)
    return mcolors.LinearSegmentedColormap.from_list("mask", [(*rgb, 0.0), (*rgb, alpha)])
