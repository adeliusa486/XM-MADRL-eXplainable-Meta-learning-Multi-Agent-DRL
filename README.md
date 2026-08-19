<div align="center">

# XM-MADRL

### eXplainable Meta-learning Multi-Agent Deep Reinforcement Learning<br/>for Cognitive UAV Swarms in Electronic-Warfare Environments

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Reproducible](https://img.shields.io/badge/results-reproducible-brightgreen.svg)](#reproducibility)
[![Code style](https://img.shields.io/badge/status-research%20code-informational.svg)](#)

*A unified framework combining multi-modal sensing, transformer fusion, graph-based
swarm communication, meta-learning and SHAP explainability for cooperative UAV
decision-making in contested spectrum.*

</div>

---

## Overview

Cognitive UAV swarms operating in electronic-warfare environments must navigate
GPS-denied airspace, share a congested radio spectrum, resist adaptive jamming,
and coordinate — all while remaining **interpretable** for mission-critical use.
**XM-MADRL** addresses these jointly in a single learning pipeline:

| # | Capability | Component | Source |
|:-:|---|---|---|
| 1 | Multi-modal sensing + fusion | per-modality tokens → **Transformer** encoder | [`models/policy.py`](models/policy.py) |
| 2 | Swarm communication | dense **2-layer GCN** over the comm graph | [`models/gnn.py`](models/gnn.py) |
| 3 | Cooperative control | **Multi-Agent PPO**, centralised critic | [`algos/mappo.py`](algos/mappo.py) |
| 4 | Fast task adaptation | first-order **MAML** meta-training | [`algos/mappo.py`](algos/mappo.py) |
| 5 | Explainability | **SHAP** feature-group attribution | [`xai/shap_explain.py`](xai/shap_explain.py) |

**Key design insight (decoupled heads).** An ablation-driven finding of this work
is that graph aggregation *over-smooths* the local spectrum-sensing signal each
agent needs for channel selection. XM-MADRL therefore uses a **decoupled policy**:
a coordinated navigation head (Transformer+GNN) and a **local-RF spectrum head**
that bypasses the graph. This more than doubles packet-delivery ratio versus a
naive graph-only policy ([`models/policy.py`](models/policy.py)).

All methods are trained and evaluated in a fast, fully reproducible **NumPy
multi-agent environment** ([`env/uav_swarm_env.py`](env/uav_swarm_env.py)) that
models UAV kinematics, an SINR/PDR fading channel, an **adaptive jammer**,
dynamic spectrum access, target detection and a multi-objective reward. It is
intentionally lightweight — the entire study runs on a single laptop, no
photorealistic simulator required.

📄 **The compiled paper** (IEEE format, generated entirely from these logs) is at
[`paper/main.pdf`](paper/main.pdf), with an editable copy at
[`paper/XM-MADRL_paper.docx`](paper/XM-MADRL_paper.docx). All figures/tables are
regenerated from real runs by [`finalize.sh`](finalize.sh) — no hand-entered
numbers.

## Architecture

```
        ┌──────────────────────── per-UAV observation ────────────────────────┐
        │  RF (RSSI · SINR · occupancy) │ radar │ visual │ nav (pos·vel·energy) │
        └───────────────────────────────┬──────────────────────────────────────┘
                                         │  modality tokens
                                 ┌───────▼────────┐
                                 │  Transformer   │   multi-modal fusion
                                 │    encoder     │
                                 └───────┬────────┘
                                         │  per-agent feature
                            ┌────────────▼────────────┐
                            │  Graph Conv (GCN) comm  │   swarm coordination
                            │   over comm-range graph │
                            └────────────┬────────────┘
                                         │  contextual feature
                       ┌─────────────────┼─────────────────┐
                 ┌─────▼─────┐     ┌──────▼──────┐    ┌──────▼──────┐
                 │  Actor    │     │  Centralised│    │    SHAP     │
                 │ (MAPPO)   │     │   Critic    │    │ explainer   │
                 └─────┬─────┘     └─────────────┘    └─────────────┘
                       │  action = [move_x, move_y, channel-select]
                 ┌─────▼──────────────────────────────────┐
                 │  MAML outer loop: adapt across EW tasks │
                 └─────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/adeliusa486/XM-MADRL-eXplainable-Meta-learning-Multi-Agent-DRL.git
cd XM-MADRL-eXplainable-Meta-learning-Multi-Agent-DRL
pip install -r requirements.txt
```

Requires Python 3.10+. A GPU is **not** needed — the networks are small and the
workload is CPU-bound (see [Performance](#performance)).

## Quick start

```bash
# 5-minute sanity check that the whole pipeline runs end-to-end
QUICK=1 python run_parallel.py --workers 4
```

## Reproduce the full study

```bash
# Recommended: parallel across CPU cores (N× faster on an N-core machine)
python run_parallel.py --workers 12

# Sequential equivalent
bash run_all.sh
```

Either command runs **5 seeds × {proposed, 4 baselines, 4 ablations}** plus the
few-shot signal-classification experiment, then automatically produces the
statistics, SHAP analysis, and every figure/table. Runs **checkpoint** to
`results/` — stop and re-run any time; finished runs are skipped.

<details>
<summary><b>Run individual pieces</b></summary>

```bash
python train.py --method XM-MADRL --seed 11     # proposed
python train.py --method PPO      --seed 11     # baseline
python train.py --method XM-noGNN --seed 11     # ablation
python signal_maml.py --seed 11                 # few-shot MAML (RadioML/synthetic)
python stats.py       --results results --baseline PPO
python run_shap.py    --weights results/XM-MADRL_seed11.pt
python make_figures.py --results results --out figures
```
</details>

## Methods & ablations

| Tag | Transformer | GNN | MAML | Role |
|---|:--:|:--:|:--:|---|
| `XM-MADRL` | ✅ | ✅ | ✅ | **proposed** |
| `XM-noMAML` | ✅ | ✅ | ❌ | ablate meta-learning |
| `XM-noGNN` | ✅ | ❌ | ✅ | ablate swarm communication |
| `XM-noTrans` | ❌ | ✅ | ✅ | ablate transformer fusion |
| `XM-noXAI` | ✅ | ✅ | ✅ | XAI is analysis-only (perf identical) |
| `PPO`, `A2C` | — | — | — | on-policy baselines |
| `DDPG`, `MADDPG` | — | — | — | off-policy baselines (replay + target nets) |

## Repository structure

```
.
├── env/                 # multi-agent UAV-EW environment (NumPy)
│   └── uav_swarm_env.py
├── models/              # transformer fusion + GCN + actor-critic
│   ├── policy.py
│   └── gnn.py
├── algos/               # MAPPO + MAML, and baseline trainers
│   ├── mappo.py
│   └── baselines.py
├── xai/                 # SHAP explainability
│   └── shap_explain.py
├── configs/default.yaml # all hyperparameters
├── train.py             # single-run training entry point
├── signal_maml.py       # few-shot signal-classification (MAML)
├── evaluate.py          # shared evaluation / metrics
├── stats.py             # CI, Welch t-test, Cohen's d
├── make_figures.py      # all figures + tables from logs
├── run_parallel.py      # parallel launcher (recommended)
└── run_all.sh           # sequential launcher
```

## Datasets

- **UAV navigation / EW scenarios** — generated by the built-in environment
  (GPS-denied navigation, adaptive jamming, spectrum congestion). No download.
- **RadioML 2018.01A** — for few-shot signal classification. Place
  `GOLD_XYZ_OSC.0001_1024.hdf5` in `data/` (from
  [DeepSig](https://www.deepsig.ai/datasets)). If absent, a clearly-flagged
  **synthetic** modulation dataset is generated so the script still runs.

## Performance

The RL bottleneck is the single-threaded, CPU-bound environment rollout — **not**
GPU compute, since the networks are tiny. Running many single-threaded jobs at
once (one per core) is therefore much faster than one GPU job. On a 14-core CPU
the full 5-seed study completes in a few hours.

## Reproducibility

- Seeds `{11, 22, 33, 44, 55}`; metrics as **mean ± SD**.
- **95 % confidence intervals**, **Welch's t-test**, **Cohen's *d*** ([`stats.py`](stats.py)).
- All hyperparameters in [`configs/default.yaml`](configs/default.yaml).
- **Every number in the paper is generated from these logs — none are hand-entered.**

## Citation

```bibtex
@article{ahmad_xmmadrl_2026,
  title   = {Adaptive Explainable Meta-Reinforcement Learning with Graph
             Intelligence for Cognitive UAV Swarms in Electronic-Warfare Environments},
  author  = {Ahmad, Adeel},
  year    = {2026},
  note    = {Code: https://github.com/adeliusa486/XM-MADRL-eXplainable-Meta-learning-Multi-Agent-DRL}
}
```

## License

Released under the [MIT License](LICENSE).
