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


def synth_modulations(n_per_class=800, n_classes=8, length=128, seed=0):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for c in range(n_classes):
        order = 2 ** (1 + c % 4)                     # 2,4,8,16-ary
        for _ in range(n_per_class):
            sym = rng.integers(0, order, size=length)
            phase = 2 * np.pi * sym / order
            amp = 1.0 + 0.3 * (c // 4)
            i = amp * np.cos(phase); q = amp * np.sin(phase)
            snr_lin = 10 ** (rng.uniform(0, 20) / 10)
            noise = rng.normal(0, 1 / np.sqrt(2 * snr_lin), size=(2, length))
            X.append(np.stack([i, q]) + noise); Y.append(c)
    return np.array(X, np.float32), np.array(Y, np.int64), True


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
def sample_episode(X, Y, classes, k_shot, q_query, rng):
    picked = rng.choice(classes, size=min(5, len(classes)), replace=False)
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

    data = Path("data/GOLD_XYZ_OSC.0001_1024.hdf5")
    if data.exists():
        X, Y, synth = load_radioml(str(data))
    else:
        X, Y, synth = synth_modulations(seed=args.seed)
    length = X.shape[-1]
    all_classes = np.unique(Y)
    n_meta_train = max(5, len(all_classes) // 2)
    train_classes = all_classes[:n_meta_train]
    test_classes = all_classes[n_meta_train:] if len(all_classes) > 5 else all_classes

    def meta_test(meta_model):
        accs = []
        for _ in range(args.test_episodes):
            xs, ys, xq, yq = sample_episode(X, Y, test_classes, args.k_shot, args.q_query, rng)
            m = type(meta_model)(2, 5, length).to(dev)
            m.load_state_dict(meta_model.state_dict())
            opt = torch.optim.SGD(m.parameters(), lr=args.inner_lr)
            xs_t = torch.tensor(xs, device=dev); ys_t = torch.tensor(ys, device=dev)
            for _ in range(args.inner_steps):
                opt.zero_grad()
                F.cross_entropy(m(xs_t), ys_t).backward(); opt.step()
            with torch.no_grad():
                pred = m(torch.tensor(xq, device=dev)).argmax(1).cpu().numpy()
            accs.append((pred == yq).mean())
        return float(np.mean(accs)), float(np.std(accs))

    # ---- MAML meta-training (first-order) --------------------------------- #
    meta = Encoder(2, 5, length).to(dev)
    meta_opt = torch.optim.Adam(meta.parameters(), lr=args.meta_lr)
    for it in range(args.meta_iters):
        meta_grads = [torch.zeros_like(p) for p in meta.parameters()]
        for _ in range(args.tasks):
            xs, ys, xq, yq = sample_episode(X, Y, train_classes, args.k_shot, args.q_query, rng)
            fast = Encoder(2, 5, length).to(dev); fast.load_state_dict(meta.state_dict())
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
            acc, sd = meta_test(meta)
            print(f"[maml it {it+1}/{args.meta_iters}] meta-test acc={acc*100:.1f}%", flush=True)

    maml_acc, maml_sd = meta_test(meta)

    # ---- from-scratch baseline (no meta-init) ----------------------------- #
    scratch_accs = []
    for _ in range(args.test_episodes):
        xs, ys, xq, yq = sample_episode(X, Y, test_classes, args.k_shot, args.q_query, rng)
        m = Encoder(2, 5, length).to(dev)
        opt = torch.optim.SGD(m.parameters(), lr=args.inner_lr)
        xs_t = torch.tensor(xs, device=dev); ys_t = torch.tensor(ys, device=dev)
        for _ in range(args.inner_steps):
            opt.zero_grad(); F.cross_entropy(m(xs_t), ys_t).backward(); opt.step()
        with torch.no_grad():
            pred = m(torch.tensor(xq, device=dev)).argmax(1).cpu().numpy()
        scratch_accs.append((pred == yq).mean())

    result = {
        "dataset": "synthetic" if synth else "RadioML2018.01A",
        "k_shot": args.k_shot, "n_way": 5,
        "maml_acc": maml_acc * 100, "maml_sd": maml_sd * 100,
        "scratch_acc": float(np.mean(scratch_accs)) * 100,
        "scratch_sd": float(np.std(scratch_accs)) * 100,
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
