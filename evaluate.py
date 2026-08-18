"""Shared evaluation: roll out a policy and aggregate the paper's metrics.

A *policy* here is any callable ``policy(obs, adj) -> actions`` (numpy in/out),
so both the torch on-policy networks and the off-policy actors reuse this code.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


METRIC_KEYS = [
    "mission_success",   # fraction of targets found  -> Mission Success Rate
    "pdr",               # packet delivery ratio       -> Communication reliability
    "mean_sinr",         # dB                          -> Signal quality
    "jammed_frac",       # fraction of steps jammed    -> (1-this) ~ anti-jamming
    "energy_used",       # total energy                -> Energy efficiency
    "collisions",        # count                       -> Safety
]


def evaluate_policy(policy: Callable, env, episodes: int = 20, seed: int = 10_000) -> Dict[str, float]:
    """Return mean metrics over ``episodes`` deterministic evaluation episodes."""
    runs: List[Dict] = []
    for ep in range(episodes):
        obs = env.reset(seed + ep)
        while True:
            adj = env._adjacency() if hasattr(env, "_adjacency") else None
            act = policy(obs, adj)
            obs, _, done, info = env.step(act)
            if done:
                runs.append(info["episode"])
                break
    agg = {}
    for k in METRIC_KEYS:
        agg[k] = float(np.mean([r[k] for r in runs]))
    # derived, reviewer-facing metrics
    agg["mission_success_pct"] = 100.0 * agg["mission_success"]
    agg["pdr_pct"] = 100.0 * agg["pdr"]
    agg["antijam_pct"] = 100.0 * (1.0 - agg["jammed_frac"])
    return agg


def torch_greedy_policy(net, device: str = "cpu") -> Callable:
    """Wrap a torch actor-critic into the numpy policy interface (greedy)."""
    import torch

    def policy(obs, adj=None):
        o = torch.as_tensor(obs, dtype=torch.float32, device=device)
        a = torch.as_tensor(adj, device=device) if adj is not None else None
        with torch.no_grad():
            action, _, _ = net.act(o, a, deterministic=True)
        return action.cpu().numpy()

    return policy
