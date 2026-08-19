"""Generate every figure and LaTeX/markdown table in the paper from real logs.

Run after all training/eval JSONs exist in ``results/``:
    python make_figures.py --results results --out figures

Produces (only for data that is present):
    figures/fig3_overall_comparison.png
    figures/fig4_trajectories.png
    figures/fig8_convergence.png
    figures/fig9_energy.png
    figures/fig10_ablation.png
    figures/fig_shap_importance.png
    figures/table_main_results.md
    figures/table3_statistics.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- brand-neutral, colour-blind-safe palette ------------------------------ #
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

METHODS_MAIN = ["XM-MADRL", "PPO", "DDPG", "MADDPG", "A2C"]
ABLATIONS = ["XM-MADRL", "XM-noMAML", "XM-noGNN", "XM-noTrans", "XM-noXAI"]


def load_evals(results: Path, method: str) -> List[Dict]:
    return [json.loads(f.read_text())
            for f in sorted(results.glob(f"{method}_seed*_eval.json"))]


def agg(results: Path, method: str, key: str):
    runs = load_evals(results, method)
    if not runs:
        return None, None
    vals = np.array([r[key] for r in runs])
    return float(vals.mean()), float(vals.std())


def fig_overall(results: Path, out: Path):
    metrics = [("mission_success_pct", "Mission Success (%)"),
               ("pdr_pct", "Packet Delivery (%)"),
               ("antijam_pct", "Anti-Jam (%)"),
               ("mean_sinr", "Mean SINR (dB)")]
    present = [m for m in METHODS_MAIN if load_evals(results, m)]
    if not present:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, (key, label) in zip(axes, metrics):
        means, sds = [], []
        for m in present:
            mu, sd = agg(results, m, key)
            means.append(mu or 0); sds.append(sd or 0)
        ax.bar(present, means, yerr=sds, capsize=4,
               color=[COLORS[i % len(COLORS)] for i in range(len(present))])
        ax.set_title(label); ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=30, ha="right")
    fig.suptitle("Fig. 3  Overall performance comparison (mean ± SD over seeds)")
    fig.tight_layout(); fig.savefig(out / "fig3_overall_comparison.png"); plt.close(fig)


def fig_convergence(results: Path, out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for i, m in enumerate(METHODS_MAIN):
        hists = [json.loads(f.read_text())
                 for f in sorted(results.glob(f"{m}_seed*_history.json"))]
        hists = [h for h in hists if h]
        if not hists:
            continue
        L = min(len(h) for h in hists)
        if L == 0:
            continue
        def succ(entry):
            if "mission_success_pct" in entry:
                return entry["mission_success_pct"]
            return entry.get("mission_success", 0.0) * 100.0
        steps = [hists[0][j].get("step", j) for j in range(L)]
        curves = np.array([[succ(h[j]) for j in range(L)] for h in hists])
        mu, sd = curves.mean(0), curves.std(0)
        ax.plot(steps, mu, label=m, color=COLORS[i % len(COLORS)])
        ax.fill_between(steps, mu - sd, mu + sd, alpha=0.15, color=COLORS[i % len(COLORS)])
        plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_xlabel("Environment steps"); ax.set_ylabel("Mission success (%)")
    ax.set_title("Fig. 8  Convergence / adaptation speed"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig8_convergence.png"); plt.close(fig)


def fig_energy(results: Path, out: Path):
    present = [m for m in METHODS_MAIN if load_evals(results, m)]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [agg(results, m, "energy_used")[0] for m in present]
    sds = [agg(results, m, "energy_used")[1] for m in present]
    ax.bar(present, means, yerr=sds, capsize=4,
           color=[COLORS[i % len(COLORS)] for i in range(len(present))])
    ax.set_ylabel("Energy consumed (norm.)"); ax.set_title("Fig. 9  Energy consumption")
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out / "fig9_energy.png"); plt.close(fig)


def fig_ablation(results: Path, out: Path):
    present = [m for m in ABLATIONS if load_evals(results, m)]
    if len(present) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [agg(results, m, "mission_success_pct")[0] for m in present]
    sds = [agg(results, m, "mission_success_pct")[1] for m in present]
    ax.bar(present, means, yerr=sds, capsize=4,
           color=[COLORS[i % len(COLORS)] for i in range(len(present))])
    ax.set_ylabel("Mission success (%)")
    ax.set_title("Fig. 10  Ablation: contribution of each module")
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out / "fig10_ablation.png"); plt.close(fig)


def fig_shap(results: Path, out: Path):
    f = results / "shap_importance.json"
    if not f.exists():
        return
    imp = json.loads(f.read_text())
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(imp.keys()); vals = [imp[k] * 100 for k in names]
    ax.barh(names, vals, color=COLORS[0])
    ax.set_xlabel("Relative importance (%)")
    ax.set_title("Fig. 11  SHAP feature-group importance")
    fig.tight_layout(); fig.savefig(out / "fig_shap_importance.png"); plt.close(fig)


def table_main(results: Path, out: Path):
    present = [m for m in METHODS_MAIN if load_evals(results, m)]
    if not present:
        return
    keys = [("mission_success_pct", "Mission Success (%)"),
            ("pdr_pct", "PDR (%)"), ("antijam_pct", "Anti-Jam (%)"),
            ("mean_sinr", "SINR (dB)"), ("energy_used", "Energy")]
    lines = ["| Method | " + " | ".join(l for _, l in keys) + " |",
             "|" + "---|" * (len(keys) + 1)]
    for m in present:
        cells = [m]
        for k, _ in keys:
            mu, sd = agg(results, m, k)
            cells.append(f"{mu:.2f} ± {sd:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    (out / "table_main_results.md").write_text("\n".join(lines))


def table_stats(results: Path, out: Path):
    f = results / "stats_table.json"
    if not f.exists():
        return
    tab = json.loads(f.read_text())
    lines = ["| Metric | Baseline (Mean±SD) | Proposed (Mean±SD) | 95% CI | p-value | Cohen's d |",
             "|---|---|---|---|---|---|"]
    for metric, s in tab.items():
        ci = s["proposed_ci"]
        lines.append(
            f"| {metric} | {s['baseline_mean']:.2f}±{s['baseline_sd']:.2f} | "
            f"{s['proposed_mean']:.2f}±{s['proposed_sd']:.2f} | "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] | {s['p_value']:.2e} | {s['cohens_d']:.2f} |")
    (out / "table3_statistics.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_overall(results, out)
    fig_convergence(results, out)
    fig_energy(results, out)
    fig_ablation(results, out)
    fig_shap(results, out)
    table_main(results, out)
    table_stats(results, out)
    print(f"Figures/tables written to {out}/ (only those with available data).")


if __name__ == "__main__":
    main()
