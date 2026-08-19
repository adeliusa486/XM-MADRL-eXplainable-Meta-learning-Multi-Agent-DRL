"""Professional architecture diagram of the XM-MADRL framework (Fig. 1).

Pure-matplotlib block diagram — no external drawing tools. Uses the paper's
validated palette and serif type for visual consistency with the data figures.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figstyle as F
F.apply_style()

BLUE = "#0072B2"; GREEN = "#009E73"; ORANGE = "#D55E00"; PURPLE = "#CC79A7"
LIGHT = "#EAF3FA"; GREY = "#F2F2F2"; INK = "#1a1a1a"


def box(ax, x, y, w, h, text, fc, ec, fs=9.2, tc=INK, bold=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                       linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, xy1, xy2, color=INK, style="-|>", lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=13,
                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    # ---- multimodal sensing (inputs) ------------------------------------- #
    mods = [("RF\n(RSSI/SINR/occ.)", 0.35), ("Radar", 2.05), ("Visual", 3.15), ("Nav\n(pos/vel/energy)", 4.25)]
    for label, x in mods:
        w = 1.55 if "\n" in label else 0.95
        box(ax, x, 5.9, w, 0.85, label, GREY, "#999", fs=7.6)
    ax.text(3.0, 6.95, "Multi-modal sensing", ha="center", fontsize=9.5, fontweight="bold", color=INK)

    # ---- transformer fusion ---------------------------------------------- #
    box(ax, 1.4, 4.4, 3.2, 0.95, "Transformer\nmulti-modal fusion", LIGHT, BLUE, fs=9)
    for _, x in mods:
        arrow(ax, (x + 0.5, 5.9), (3.0, 5.35), color="#888")

    # ---- GNN comm -------------------------------------------------------- #
    box(ax, 1.4, 2.95, 3.2, 0.95, "GNN swarm\ncommunication", LIGHT, BLUE, fs=9)
    arrow(ax, (3.0, 4.4), (3.0, 3.9), color=BLUE, lw=2)

    # ---- decoupled heads ------------------------------------------------- #
    box(ax, 0.5, 1.35, 2.15, 0.95, "Navigation\nhead", "#FDECE0", ORANGE, fs=8.6)
    box(ax, 3.35, 1.35, 2.15, 0.95, "Spectrum\nhead (local RF)", "#FDECE0", ORANGE, fs=8.6)
    arrow(ax, (2.4, 2.95), (1.6, 2.3), color=BLUE, lw=1.8)          # gnn -> nav
    arrow(ax, (3.0, 5.0), (4.4, 2.3), color=GREEN, lw=1.8, rad=-0.3)  # local RF skip -> spectrum
    ax.text(4.7, 3.7, "local-RF\nskip", ha="left", fontsize=7.2, color=GREEN, style="italic")

    # ---- action / environment loop --------------------------------------- #
    box(ax, 1.55, 0.15, 2.9, 0.8, "Action: move + channel", "#EFEAF5", PURPLE, fs=8.8)
    arrow(ax, (1.6, 1.35), (2.4, 0.95), color=ORANGE)
    arrow(ax, (4.4, 1.35), (3.6, 0.95), color=ORANGE)

    # ---- learning + XAI side blocks -------------------------------------- #
    box(ax, 6.6, 3.9, 3.0, 1.0, "MAPPO\n(centralised critic)", "#E8F6F1", GREEN, fs=9)
    box(ax, 6.6, 2.4, 3.0, 1.0, "MAML\nfast task adaptation", "#E8F6F1", GREEN, fs=9)
    box(ax, 6.6, 0.9, 3.0, 1.0, "SHAP\nexplainability", "#FBEAF2", PURPLE, fs=9)
    arrow(ax, (4.6, 4.87), (6.6, 4.4), color="#888", rad=0.15)
    arrow(ax, (6.6, 2.9), (4.6, 1.9), color=GREEN, rad=0.15)
    arrow(ax, (4.6, 1.5), (6.6, 1.4), color=PURPLE, rad=-0.1)

    # ---- environment box ------------------------------------------------- #
    box(ax, 10.0, 2.4, 1.7, 2.1, "UAV-EW\nEnviron-\nment\n(jammer,\nspectrum)", "#F7F7F5", "#555", fs=8)
    arrow(ax, (4.45, 0.55), (10.6, 2.4), color=INK, rad=-0.25, lw=1.7)
    arrow(ax, (10.85, 4.5), (3.0, 6.75), color=INK, rad=-0.28, lw=1.3, style="-|>")
    ax.text(7.9, 5.7, "observations", ha="center", fontsize=7.6, color=INK, style="italic")
    ax.text(7.6, 0.35, "actions", ha="center", fontsize=7.6, color=INK, style="italic")

    fig.tight_layout()
    for d in ("figures", "paper/figures"):
        fig.savefig(f"{d}/fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_architecture.png")


if __name__ == "__main__":
    main()
