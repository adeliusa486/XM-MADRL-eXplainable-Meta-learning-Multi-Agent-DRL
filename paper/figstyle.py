"""Publication-grade matplotlib style for the XM-MADRL paper.

Design decisions (validated, not eyeballed):
* Colorblind-safe categorical palette (Okabe-Ito subset), verified with the
  dataviz validator: worst adjacent CVD delta-E 8.5-11 (>= floor), normal-vision
  floor 21.
* Serif type to match the IEEE (Times) body; sizes tuned for single-column
  (~3.4 in) reproduction at 300 dpi.
* Recessive grid behind the data; top/right spines removed; thin marks; direct
  value labels; error bars = 95% CI where seeds allow, else SD.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- validated colorblind-safe palette (Okabe-Ito subset) ------------------ #
PALETTE = {
    "XM-MADRL": "#0072B2",   # strong blue  -> proposed (most prominent)
    "PPO":      "#D55E00",   # vermillion
    "A2C":      "#009E73",   # bluish green
    "DDPG":     "#E69F00",   # orange
    "MADDPG":   "#CC79A7",   # reddish purple
}
PROPOSED = "#0072B2"
GRID = "#D9D9D9"
INK = "#222222"
MUTED = "#666666"


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "errorbar.capsize": 2.5,
    })


def color(method):
    return PALETTE.get(method, "#888888")


def bar_labels(ax, bars, values, errs=None, fmt="{:.1f}", dy=0.04, fs=7.5):
    """Direct value labels above bars, clearing the error-bar caps."""
    ymax = max(values) if values else 1
    errs = errs or [0] * len(bars)
    for b, v, e in zip(bars, values, errs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + e + dy * ymax,
                fmt.format(v), ha="center", va="bottom", fontsize=fs, color=INK)


def finalize(ax):
    ax.grid(axis="x", visible=False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
