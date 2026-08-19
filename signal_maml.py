"""Few-shot signal (modulation) classification with MAML.

Reproduces the paper's claim that meta-learning enables high-accuracy
modulation recognition from few labelled samples.

Data:
  * If ``data/GOLD_XYZ_OSC.0001_1024.hdf5`` (RadioML 2018.01A) is present it is
    used directly.  Download it from https://www.deepsig.ai/datasets .
  * Otherwise a synthetic modulation dataset (BPSK/QPSK/8PSK/QAM16/...) is
    generated so the pipeline is runnable end-to-end without the download.  The
    synthetic mode is clearly flagged in the output and in the saved JSON.

Protocol: N-way K-shot episodic MAML.  We report meta-test accuracy vs. a
"from-scratch" (no meta-learning) baseline trained on the same K shots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_radioml(path: str):
    import h5py
    with h5py.File(path, "r") as f:
        X = np.array(f["X"])          # (n, 1024, 2)
        Y = np.array(f["Y"]).argmax(1)
    return X.astype(np.float32), Y.astype(np.int64), False


def _constellation(kind):
    """Return a unit-power complex constellation for a modulation type."""
    if kind == "BPSK":
        pts = np.array([1, -1], dtype=complex)
    elif kind == "QPSK":
        pts = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
    elif kind == "8PSK":
        pts = np.exp(1j * np.arange(8) * np.pi / 4)
    elif kind in ("16QAM", "64QAM"):
        m = 4 if kind == "16QAM" else 8
        lv = np.arange(-(m - 1), m, 2)
        I, Q = np.meshgrid(lv, lv)
        pts = (I + 1j * Q).ravel().astype(complex)
    elif kind == "PAM4":
        pts = np.array([-3, -1, 1, 3], dtype=complex)
    else:  # noise-like / analog FM surrogate
        pts = None
    if pts is not None:
        pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))   # unit average power
    return pts


def synth_modulations(n_per_class=800, n_classes=6, length=128, seed=0):
    """Six genuinely-separable digital modulations at readable SNR (8-18 dB).

    Distinct constellation families (PSK orders vs QAM grids vs PAM) give a
    classification task a CNN can actually solve, unlike a phase-only surrogate.
    """
    rng = np.random.default_rng(seed)
    kinds = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4"][:n_classes]
    X, Y = [], []
    for c, kind in enumerate(kinds):
        cst = _constellation(kind)
        for _ in range(n_per_class):
            sym = cst[rng.integers(0, len(cst), size=length)]
            snr_lin = 10 ** (rng.uniform(8, 18) / 10)
            noise = (rng.normal(0, 1, length) + 1j * rng.normal(0, 1, length)) / np.sqrt(2 * snr_lin)
            s = sym + noise
            X.append(np.stack([s.real, s.imag])); Y.append(c)
    return np.array(X, np.float32), np.array(Y, np.int64), True


MOD_NAMES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4"]


class Encoder(nn.Module):
    def __init__(self, in_ch=2, n_classes=5, length=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.conv(x).squeeze(-1))


# --------------------------------------------------------------------------- #
# MAML
# --------------------------------------------------------------------------- #
def sample_episode(X, Y, classes, k_shot, q_query, rng, n_way=5):
    picked = rng.choice(classes, size=min(n_way, len(classes)), replace=False)
    xs, ys, xq, yq = [], [], [], []
    for new_label, c in enumerate(picked):
        idx = np.where(Y == c)[0]
        sel = rng.choice(idx, size=k_shot + q_query, replace=False)
        xs.append(X[sel[:k_shot]]); ys += [new_label] * k_shot
        xq.append(X[sel[k_shot:]]); yq += [new_label] * q_query
    return (np.concatenate(xs), np.array(ys, dtype=np.int64),
            np.concatenate(xq), np.array(yq, dtype=np.int64))


def run(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper"))
    from paper_classifier import ConstellationCNN, to_hist

    data = Path("data/GOLD_XYZ_OSC.0001_1024.hdf5")
    if data.exists():
        X, Y, synth = load_radioml(str(data))
    else:
        X, Y, synth = synth_modulations(seed=args.seed)
    H = to_hist(X)                                          # constellation-density images
    all_classes = np.unique(Y)
    rng.shuffle(all_classes)                                # mix families across the split
    half = len(all_classes) // 2
    train_classes = all_classes[:half]
    test_classes = all_classes[half:]
    n_way = min(len(train_classes), len(test_classes))     # held-out few-shot task width

    def new_model():
        return ConstellationCNN(n_way).to(dev)

    def adapt_and_eval(state):
        accs = []
        for _ in range(args.test_episodes):
            xs, ys, xq, yq = sample_episode(H, Y, test_classes, args.k_shot, args.q_query, rng, n_way)
            m = new_model()
            if state is not None:
                m.load_state_dict(state)
            opt = torch.optim.SGD(m.parameters(), lr=args.inner_lr)
            xs_t = torch.tensor(xs, device=dev); ys_t = torch.tensor(ys, device=dev)
            for _ in range(args.inner_steps):
                opt.zero_grad(); F.cross_entropy(m(xs_t), ys_t).backward(); opt.step()
            with torch.no_grad():
                pred = m(torch.tensor(xq, device=dev)).argmax(1).cpu().numpy()
            accs.append((pred == yq).mean())
        return float(np.mean(accs)), float(np.std(accs))

    # ---- MAML meta-training (first-order) --------------------------------- #
    meta = new_model()
    meta_opt = torch.optim.Adam(meta.parameters(), lr=args.meta_lr)
    for it in range(args.meta_iters):
        meta_grads = [torch.zeros_like(p) for p in meta.parameters()]
        for _ in range(args.tasks):
            xs, ys, xq, yq = sample_episode(H, Y, train_classes, args.k_shot, args.q_query, rng, n_way)
            fast = new_model(); fast.load_state_dict(meta.state_dict())
            opt = torch.optim.SGD(fast.parameters(), lr=args.inner_lr)
            xs_t = torch.tensor(xs, device=dev); ys_t = torch.tensor(ys, device=dev)
            for _ in range(args.inner_steps):
                opt.zero_grad(); F.cross_entropy(fast(xs_t), ys_t).backward(); opt.step()
            fast.zero_grad()
            F.cross_entropy(fast(torch.tensor(xq, device=dev)),
                            torch.tensor(yq, device=dev)).backward()
            for mg, fp in zip(meta_grads, fast.parameters()):
                if fp.grad is not None:
                    mg += fp.grad.detach() / args.tasks
        meta_opt.zero_grad()
        for p, g in zip(meta.parameters(), meta_grads):
            p.grad = g.clone()
        meta_opt.step()
        if (it + 1) % max(1, args.meta_iters // 5) == 0:
            acc, sd = adapt_and_eval(meta.state_dict())
            print(f"[maml it {it+1}/{args.meta_iters}] meta-test acc={acc*100:.1f}%", flush=True)

    maml_acc, maml_sd = adapt_and_eval(meta.state_dict())
    scratch_acc, scratch_sd = adapt_and_eval(None)         # from-scratch baseline

    result = {
        "dataset": "synthetic" if synth else "RadioML2018.01A",
        "k_shot": args.k_shot, "n_way": n_way,
        "maml_acc": maml_acc * 100, "maml_sd": maml_sd * 100,
        "scratch_acc": scratch_acc * 100,
        "scratch_sd": scratch_sd * 100,
        "seed": args.seed,
    }
    out = Path(args.results); out.mkdir(parents=True, exist_ok=True)
    (out / f"signal_maml_seed{args.seed}.json").write_text(json.dumps(result, indent=2))
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--q_query", type=int, default=15)
    ap.add_argument("--meta_iters", type=int, default=200)
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--inner_lr", type=float, default=0.01)
    ap.add_argument("--inner_steps", type=int, default=5)
    ap.add_argument("--meta_lr", type=float, default=1e-3)
    ap.add_argument("--test_episodes", type=int, default=100)
    ap.add_argument("--results", default="results")
    run(ap.parse_args())
