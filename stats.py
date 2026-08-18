"""Statistical analysis: mean +/- SD, 95% CI, Welch t-test, Cohen's d.

Consumes the per-seed metric JSONs written by ``train.py`` and produces the
statistical-comparison table used in the paper (Table 3).  Unlike the original
draft, every test is computed from real per-seed distributions, so no metric has
a degenerate zero-variance baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as st


def mean_ci(x: np.ndarray, conf: float = 0.95) -> Tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n > 1:
        h = sd / np.sqrt(n) * st.t.ppf((1 + conf) / 2.0, n - 1)
    else:
        h = 0.0
    return m, sd, m - h, m + h


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / sp)


def compare(proposed: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    m_p, sd_p, lo_p, hi_p = mean_ci(proposed)
    m_b, sd_b, lo_b, hi_b = mean_ci(baseline)
    t, p = st.ttest_ind(proposed, baseline, equal_var=False)   # Welch
    return {
        "proposed_mean": m_p, "proposed_sd": sd_p, "proposed_ci": [lo_p, hi_p],
        "baseline_mean": m_b, "baseline_sd": sd_b, "baseline_ci": [lo_b, hi_b],
        "t_stat": float(t), "p_value": float(p), "cohens_d": cohens_d(proposed, baseline),
    }


def load_seed_metrics(results_dir: str, method: str) -> List[Dict]:
    out = []
    for f in sorted(Path(results_dir).glob(f"{method}_seed*_eval.json")):
        out.append(json.loads(f.read_text()))
    return out


def build_stats_table(results_dir: str, proposed: str = "XM-MADRL",
                      best_baseline: str = "PPO", metrics=None) -> Dict:
    metrics = metrics or ["mission_success_pct", "pdr_pct", "antijam_pct",
                          "mean_sinr", "energy_used"]
    p_runs = load_seed_metrics(results_dir, proposed)
    b_runs = load_seed_metrics(results_dir, best_baseline)
    table = {}
    for m in metrics:
        pv = np.array([r[m] for r in p_runs])
        bv = np.array([r[m] for r in b_runs])
        table[m] = compare(pv, bv)
    Path(results_dir, "stats_table.json").write_text(json.dumps(table, indent=2))
    return table


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--baseline", default="PPO")
    args = ap.parse_args()
    t = build_stats_table(args.results, best_baseline=args.baseline)
    print(json.dumps(t, indent=2))
