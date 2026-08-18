"""Load a trained XM-MADRL policy and compute SHAP feature-group importance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from env import UAVSwarmEWEnv, TaskConfig
from env.uav_swarm_env import feature_groups
from models import XMActorCritic
from xai import explain_policy, feature_group_importance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="results/XM-MADRL_seed11.pt")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    cfg = TaskConfig(seed=args.seed)
    env = UAVSwarmEWEnv(cfg)
    fg = feature_groups(cfg)
    net = XMActorCritic(env.obs_dim, env.act_dim, env.n, fg)
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    net.eval()

    shap_values = explain_policy(net, env, device="cpu", n_background=60, n_explain=40)
    importance = feature_group_importance(shap_values, cfg)
    Path(args.results, "shap_importance.json").write_text(json.dumps(importance, indent=2))
    print("SHAP feature-group importance:", json.dumps(importance, indent=2))


if __name__ == "__main__":
    main()
