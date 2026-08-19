"""Generate all publication-grade figures for the paper from real result logs.

Usage:  python paper/pro_figures.py
Writes PNGs (300 dpi) to figures/ and paper/figures/.
"""
import os, json, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

import figstyle as F
F.apply_style()

R = os.environ.get("FIGR", "results")
OUT = ["figures", "paper/figures"]
MAIN = ["XM-MADRL", "PPO", "A2C", "DDPG", "MADDPG"]
ABL = ["XM-MADRL", "XM-noMAML", "XM-noGNN", "XM-noTrans"]
os.makedirs("figures", exist_ok=True); os.makedirs("paper/figures", exist_ok=True)


def load(path, method, key):
    v = [json.load(open(f))[key] for f in glob.glob(f"{path}/{method}_seed*_eval.json")]
    return np.array(v, float) if v else None


def ci95(v):
    """Return (mean, half-width of 95% CI)."""
    n = len(v)
    if n < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(n) * st.t.ppf(0.975, n - 1))


def save(fig, name):
    for d in OUT:
        fig.savefig(f"{d}/{name}.png")
    plt.close(fig)


# --- Fig: overall comparison (4 metrics, grouped) -------------------------- #
def fig_overall():
    metrics = [("mission_success_pct", "Mission success (%)"),
               ("pdr_pct", "Packet delivery ratio (%)"),
               ("antijam_pct", "Anti-jamming (%)"),
               ("energy_used", "Energy consumed (norm.)")]
    present = [m for m in MAIN if load(R, m, "pdr_pct") is not None]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    for ax, (key, label) in zip(axes.ravel(), metrics):
        means, errs, cols = [], [], []
        for m in present:
            v = load(R, m, key); mu, h = ci95(v)
            means.append(mu); errs.append(h); cols.append(F.color(m))
        x = np.arange(len(present))
        bars = ax.bar(x, means, yerr=errs, color=cols, width=0.68,
                      edgecolor="white", linewidth=0.6,
                      error_kw=dict(ecolor=F.MUTED, lw=1.0))
        F.bar_labels(ax, bars, means, errs)
        ax.set_xticks(x); ax.set_xticklabels(present, rotation=20, ha="right", fontsize=7.5)
        ax.set_title(label); ax.set_ylabel(""); F.finalize(ax)
        ax.margins(y=0.20)
    fig.suptitle("Overall performance (mean, 95% CI over 5 seeds)",
                 fontsize=10, fontweight="bold", y=1.005)
    fig.tight_layout()
    save(fig, "fig3_overall_comparison")


# --- Fig: convergence with CI band ----------------------------------------- #
def _succ(entry):
    return entry["mission_success_pct"] if "mission_success_pct" in entry else entry.get("mission_success", 0) * 100


def _smooth(y, w=5):
    """Edge-aware moving average (no convolution boundary dip)."""
    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    w = min(w, len(y)); half = w // 2
    out = np.convolve(y, np.ones(w) / w, mode="same")
    for i in range(half):                                  # shrink window at edges
        out[i] = y[:i + half + 1].mean()
        out[-(i + 1)] = y[-(i + half + 1):].mean()
    return out


def fig_convergence(cap=300000):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    any_ = False
    for m in MAIN:
        hs = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/{m}_seed*_history.json"))]
        hs = [h for h in hs if h]
        if not hs:
            continue
        L = min(len(h) for h in hs)
        if L < 2:
            continue
        steps = np.array([hs[0][j].get("step", j) for j in range(L)])
        cur = np.array([[_succ(h[j]) for j in range(L)] for h in hs])
        mask = steps <= cap                                    # common range for fairness
        steps = steps[mask]; cur = cur[:, mask]
        mu = _smooth(cur.mean(0)); sd = _smooth(cur.std(0))
        c = F.color(m); lw = 2.4 if m == "XM-MADRL" else 1.3
        z = 4 if m == "XM-MADRL" else 2
        ax.plot(steps, mu, color=c, lw=lw, label=m, zorder=z)
        ax.fill_between(steps, mu - sd, mu + sd, color=c, alpha=0.10, lw=0, zorder=1)
        any_ = True
    if not any_:
        plt.close(fig); return
    ax.set_xlabel("Environment steps"); ax.set_ylabel("Mission success (%)")
    ax.set_title("Convergence"); ax.legend(ncol=1, loc="lower right")
    F.finalize(fig.axes[0]); fig.tight_layout()
    save(fig, "fig8_convergence")


# --- Fig: single-metric bar (energy) --------------------------------------- #
def fig_bar(key, title, name, fmt="{:.2f}"):
    present = [m for m in MAIN if load(R, m, key) is not None]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    means, errs, cols = [], [], []
    for m in present:
        mu, h = ci95(load(R, m, key)); means.append(mu); errs.append(h); cols.append(F.color(m))
    x = np.arange(len(present))
    bars = ax.bar(x, means, yerr=errs, color=cols, width=0.66, edgecolor="white",
                  linewidth=0.6, error_kw=dict(ecolor=F.MUTED, lw=1.0))
    F.bar_labels(ax, bars, means, errs, fmt=fmt)
    ax.set_xticks(x); ax.set_xticklabels(present, rotation=20, ha="right", fontsize=7.5)
    ax.set_title(title); F.finalize(ax); ax.margins(y=0.20)
    fig.tight_layout(); save(fig, name)


# --- Fig: ablation --------------------------------------------------------- #
def fig_ablation():
    labels = {"XM-MADRL": "Full", "XM-noMAML": "-MAML", "XM-noGNN": "-GNN", "XM-noTrans": "-Transf."}
    present = [m for m in ABL if load(R, m, "pdr_pct") is not None]
    metrics = [("mission_success_pct", "Success"), ("pdr_pct", "PDR"), ("antijam_pct", "Anti-jam")]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(len(present)); w = 0.26
    mcolors = ["#0072B2", "#D55E00", "#009E73"]
    for i, (key, lab) in enumerate(metrics):
        vals = [ci95(load(R, m, key))[0] for m in present]
        ax.bar(x + (i - 1) * w, vals, w, label=lab, color=mcolors[i], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in present], fontsize=7.5)
    ax.set_ylabel("Score (%)"); ax.set_title("Ablation (component removal)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28), columnspacing=1.0)
    F.finalize(ax); fig.tight_layout(); save(fig, "fig10_ablation")


# --- Fig: SHAP importance (sequential single hue) -------------------------- #
def fig_shap():
    f = f"{R}/shap_importance.json"
    if not os.path.exists(f):
        return
    imp = json.load(open(f))
    names = list(imp.keys()); vals = np.array([imp[k] * 100 for k in names])
    order = np.argsort(vals)
    names = [names[i].replace("_", " ") for i in order]; vals = vals[order]
    base = np.array([0, 0.45, 0.70])  # blue in RGB-ish for a light->dark ramp
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    shades = [plt.cm.Blues(0.35 + 0.6 * v / vals.max()) for v in vals]
    bars = ax.barh(names, vals, color=shades, edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2, f"{v:.0f}%",
                va="center", fontsize=7.5, color=F.INK)
    ax.set_xlabel("Relative importance (%)"); ax.set_title("SHAP feature-group importance")
    ax.grid(axis="y", visible=False); F.finalize(ax); ax.margins(x=0.12)
    fig.tight_layout(); save(fig, "fig_shap_importance")


# --- Fig: scalability (2 panels) ------------------------------------------- #
def fig_scalability():
    counts = [6, 12, 18]; methods = ["XM-MADRL", "PPO"]
    def agg(m, n, key):
        fs = glob.glob(f"results_scale/n{n}/{m}_seed*_eval.json")
        if not fs:
            return None
        v = np.array([json.load(open(f))[key] for f in fs])
        return float(v.mean()), float(v.std())   # mean +/- SD (robust for few seeds)
    have = any(agg("XM-MADRL", n, "pdr_pct") for n in counts)
    if not have:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for m in methods:
        xs, sm, se, pm, pe = [], [], [], [], []
        for n in counts:
            s = agg(m, n, "mission_success_pct"); p = agg(m, n, "pdr_pct")
            if s and p:
                xs.append(n); sm.append(s[0]); se.append(s[1]); pm.append(p[0]); pe.append(p[1])
        c = F.color(m); lw = 2.2 if m == "XM-MADRL" else 1.6
        a1.errorbar(xs, sm, yerr=se, marker="o", color=c, lw=lw, label=m, capsize=2.5)
        a2.errorbar(xs, pm, yerr=pe, marker="o", color=c, lw=lw, label=m, capsize=2.5)
    a1.set_xlabel("Number of UAVs"); a1.set_ylabel("Mission success (%)"); a1.set_title("Scalability: mission")
    a2.set_xlabel("Number of UAVs"); a2.set_ylabel("Packet delivery ratio (%)"); a2.set_title("Scalability: communication")
    for a in (a1, a2):
        a.set_xticks(counts); a.legend(loc="best"); F.finalize(a)
    fig.tight_layout(); save(fig, "fig_scalability")


if __name__ == "__main__":
    fig_overall()
    fig_convergence()
    fig_bar("energy_used", "Energy consumption", "fig9_energy", fmt="{:.2f}")
    fig_ablation()
    fig_shap()
    fig_scalability()
    print("professional figures written to figures/ and paper/figures/")
