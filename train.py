"""Unified training entry point.

Examples
--------
    # proposed method, one seed
    python train.py --method XM-MADRL --seed 11

    # ablations
    python train.py --method XM-noMAML --seed 11
    python train.py --method XM-noGNN  --seed 11

    # baselines
    python train.py --method PPO   --seed 11
    python train.py --method MADDPG --seed 11

Every run writes:
    results/<method>_seed<seed>_history.json   (learning curve)
    results/<method>_seed<seed>_eval.json      (final metrics)
    results/<method>_seed<seed>.pt             (weights)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from env import UAVSwarmEWEnv, TaskConfig, sample_task
from env.uav_swarm_env import feature_groups
from models import XMActorCritic, MLPActorCritic
from algos.mappo import MAPPO, PPOConfig, RolloutBuffer
from algos.baselines import make_baseline
from evaluate import evaluate_policy, torch_greedy_policy


# method -> (backbone flags, ppo flags)
PROPOSED = {
    "XM-MADRL":    dict(use_transformer=True,  use_gnn=True,  meta=True),
    "XM-noMAML":   dict(use_transformer=True,  use_gnn=True,  meta=False),
    "XM-noGNN":    dict(use_transformer=True,  use_gnn=False, meta=True),
    "XM-noTrans":  dict(use_transformer=False, use_gnn=True,  meta=True),
    "XM-noXAI":    dict(use_transformer=True,  use_gnn=True,  meta=True),  # XAI is analysis-only
}
ONPOLICY_BASELINES = {
    "PPO": dict(epochs=4,  clip=0.2),
    "A2C": dict(epochs=1,  clip=100.0),          # no clipping -> vanilla advantage AC
}
OFFPOLICY_BASELINES = ("DDPG", "MADDPG")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(seed: int) -> UAVSwarmEWEnv:
    return UAVSwarmEWEnv(TaskConfig(seed=seed))


def train_onpolicy(method: str, seed: int, total_steps: int, device: str, out: Path,
                   log_every: int = 4096):
    cfg_base = TaskConfig(seed=seed)
    flags = PROPOSED.get(method, {})
    is_proposed = method in PROPOSED
    ppo_extra = ONPOLICY_BASELINES.get(method, {})

    env = UAVSwarmEWEnv(cfg_base)
    fg = feature_groups(cfg_base)

    if is_proposed:
        policy = XMActorCritic(env.obs_dim, env.act_dim, env.n, fg,
                               use_transformer=flags["use_transformer"],
                               use_gnn=flags["use_gnn"])
        meta = flags["meta"]
    else:
        policy = MLPActorCritic(env.obs_dim, env.act_dim,
                                centralised_critic=False, n_agents=env.n)
        meta = False

    pcfg = PPOConfig(rollout_steps=256, meta=meta, meta_tasks=4,
                     inner_lr=0.01, inner_steps=1, **ppo_extra)

    rng = np.random.default_rng(seed)
    make_task = (lambda r: UAVSwarmEWEnv(sample_task(r, cfg_base))) if meta else None
    trainer = MAPPO(policy, lambda: UAVSwarmEWEnv(cfg_base), pcfg, device, make_task)

    history = []
    obs = env.reset()
    steps_done = 0
    t0 = time.time()
    while steps_done < total_steps:
        if meta:
            info = trainer.meta_update(rng)
            steps_done += pcfg.rollout_steps * pcfg.meta_tasks * 2
        # standard on-policy improvement on the fixed training task
        buf = RolloutBuffer()
        obs = trainer.collect(env, buf, pcfg.rollout_steps, obs)
        last_v = trainer._bootstrap_value(policy, env, obs)
        stats = trainer.update(buf, last_v)
        steps_done += pcfg.rollout_steps

        if steps_done // log_every > len(history):
            m = evaluate_policy(torch_greedy_policy(policy, device), build_env(seed + 999), episodes=5)
            m["step"] = steps_done
            history.append(m)
            print(f"[{method} s{seed}] step {steps_done}/{total_steps} "
                  f"success={m['mission_success_pct']:.1f}% pdr={m['pdr_pct']:.1f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # final evaluation on held-out task seeds
    final = evaluate_policy(torch_greedy_policy(policy, device), build_env(seed + 999), episodes=30)
    torch.save(policy.state_dict(), out / f"{method}_seed{seed}.pt")
    return history, final


def train_offpolicy(method: str, seed: int, total_steps: int, device: str, out: Path):
    trainer, _ = make_baseline(method, lambda: build_env(seed), device, total_steps)
    history = trainer.learn(total_steps, log_every=4096)
    final = trainer.evaluate(build_env(seed + 999), episodes=30)
    torch.save(trainer.state_dict(), out / f"{method}_seed{seed}.pt")
    return history, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps", type=int, default=300_000,
                    help="total environment steps (reduce for a quick test)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.results); out.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.method} seed={args.seed} steps={args.steps} device={args.device} ===",
          flush=True)

    if args.method in OFFPOLICY_BASELINES:
        history, final = train_offpolicy(args.method, args.seed, args.steps, args.device, out)
    else:
        history, final = train_onpolicy(args.method, args.seed, args.steps, args.device, out)

    (out / f"{args.method}_seed{args.seed}_history.json").write_text(json.dumps(history, indent=2))
    (out / f"{args.method}_seed{args.seed}_eval.json").write_text(json.dumps(final, indent=2))
    print(f"FINAL {args.method} s{args.seed}: "
          f"success={final['mission_success_pct']:.1f}%  pdr={final['pdr_pct']:.1f}%  "
          f"antijam={final['antijam_pct']:.1f}%  sinr={final['mean_sinr']:.2f}dB  "
          f"energy={final['energy_used']:.3f}", flush=True)


if __name__ == "__main__":
    main()
