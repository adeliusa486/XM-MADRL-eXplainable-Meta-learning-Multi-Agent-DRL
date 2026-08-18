"""SHAP-based explainability for the XM-MADRL policy.

We treat the trained actor as a function f(obs) -> action and use SHAP's
KernelExplainer over a background sample of observations collected from the
environment.  Per-feature Shapley values are aggregated into the six sensing
groups (RF-RSSI, RF-SINR, RF-occupancy, radar, visual, nav) so the paper can
report *which modality* drives channel-selection / navigation decisions.

This module is analysis-only: it does not affect training, which is exactly why
removing it in the ablation study leaves task performance unchanged while
removing all decision transparency.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from env.uav_swarm_env import TaskConfig, feature_groups


def _collect_background(policy_net, env, n: int, device: str) -> np.ndarray:
    obs_all = []
    obs = env.reset()
    while len(obs_all) < n:
        adj = env._adjacency()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=device)
            a = torch.as_tensor(adj, device=device)
            action, _, _ = policy_net.act(o, a, deterministic=True)
        obs_all.extend(list(obs))
        obs, _, done, _ = env.step(action.cpu().numpy())
        if done:
            obs = env.reset()
    return np.array(obs_all[:n], dtype=np.float32)


def explain_policy(policy_net, env, device: str = "cpu",
                   n_background: int = 100, n_explain: int = 50) -> np.ndarray:
    """Return SHAP values of shape (n_explain, obs_dim) for the action output.

    The scalar model output explained is the movement magnitude + selected-channel
    confidence, a compact proxy for "how strongly the observation drove the act".
    """
    import shap

    bg = _collect_background(policy_net, env, n_background, device)
    expl_x = _collect_background(policy_net, env, n_explain, device)

    def f(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(x, dtype=torch.float32, device=device)
            # actor is graph-aware; for attribution we score each row independently
            fake_adj = torch.eye(o.size(0), device=device)
            action, _, _ = policy_net.act(o, fake_adj, deterministic=True)
            a = action.cpu().numpy()
        move_mag = np.linalg.norm(a[:, :2], axis=1)
        ch_conf = np.max(a[:, 2:], axis=1)
        return move_mag + ch_conf

    explainer = shap.KernelExplainer(f, shap.kmeans(bg, min(10, len(bg))))
    shap_values = explainer.shap_values(expl_x, nsamples=100, silent=True)
    return np.asarray(shap_values)


def feature_group_importance(shap_values: np.ndarray, cfg: TaskConfig) -> Dict[str, float]:
    """Aggregate per-feature |SHAP| into the six sensing modalities (normalised)."""
    groups = feature_groups(cfg)
    mean_abs = np.abs(shap_values).mean(axis=0)
    out = {name: float(mean_abs[sl].sum()) for name, sl in groups.items()}
    total = sum(out.values()) + 1e-9
    return {k: v / total for k, v in out.items()}
