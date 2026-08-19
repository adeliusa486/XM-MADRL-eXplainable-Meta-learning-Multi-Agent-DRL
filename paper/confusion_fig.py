"""Confusion matrix for modulation classification (Fig. 7), from a real classifier.

Trains the same CNN encoder used in the few-shot study on the synthetic
modulation dataset (or RadioML if present) and renders a normalized confusion
matrix. This is genuine model output, not a mock-up.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE)); sys.path.insert(0, _HERE)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import figstyle as F
F.apply_style()

from signal_maml import synth_modulations, load_radioml
from paper_classifier import ConstellationCNN, to_hist

LABELS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4"]


def main():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    data = "data/GOLD_XYZ_OSC.0001_1024.hdf5"
    X, Y, synth = (load_radioml(data) if os.path.exists(data) else synth_modulations(seed=0))
    ncls = int(Y.max() + 1)
    H = torch.tensor(to_hist(X))                          # constellation-density images
    idx = rng.permutation(len(X)); ntr = int(0.8 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    Htr, Ytr = H[tr], torch.tensor(Y[tr], dtype=torch.long)
    Hte, Yte = H[te], Y[te]

    net = ConstellationCNN(ncls); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(15):
        perm = torch.randperm(len(Htr))
        for i in range(0, len(Htr), 256):
            b = perm[i:i + 256]
            opt.zero_grad(); Fn.cross_entropy(net(Htr[b]), Ytr[b]).backward(); opt.step()
    with torch.no_grad():
        pred = net(Hte).argmax(1).numpy()
    acc = (pred == Yte).mean()

    cm = np.zeros((ncls, ncls))
    for t, p in zip(Yte, pred):
        cm[t, p] += 1
    cm = cm / cm.sum(1, keepdims=True) * 100

    labels = LABELS[:ncls]
    blues = LinearSegmentedColormap.from_list("b", ["#FFFFFF", "#0072B2"])
    fig, ax = plt.subplots(figsize=(3.7, 3.3))
    im = ax.imshow(cm, cmap=blues, vmin=0, vmax=100)
    ax.set_xticks(range(ncls)); ax.set_yticks(range(ncls))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(ncls):
        for j in range(ncls):
            v = cm[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 55 else F.INK)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Modulation confusion matrix (acc {acc*100:.1f}%)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=7)
    cb.set_label("%", fontsize=7)
    ax.grid(False)
    fig.tight_layout()
    for d in ("figures", "paper/figures"):
        fig.savefig(f"{d}/fig7_confusion.png", dpi=300)
    plt.close(fig)
    print(f"wrote fig7_confusion.png (acc {acc*100:.1f}%, {'synthetic' if synth else 'RadioML'})")


if __name__ == "__main__":
    main()
