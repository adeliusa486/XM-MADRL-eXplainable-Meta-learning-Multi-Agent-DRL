"""Real data-backed environment figures: UAV trajectories and SINR-under-jamming.

Rolls out a trained policy in the UAV-EW environment, records per-step positions,
target detections, jammer positions, and SINR, and renders two publication-grade
figures. Usage:  python paper/env_figures.py --method XM-MADRL --weights <pt>
"""
import argparse, glob, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root (env, models)
sys.path.insert(0, _HERE)                     # paper/ (figstyle)
import numpy as np
import torch
import matplotlib.pyplot as plt
import figstyle as F
F.apply_style()

from env import UAVSwarmEWEnv, TaskConfig
from env.uav_swarm_env import feature_groups
from models import XMActorCritic, MLPActorCritic

BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"


def load_policy(method, weights, env):
    sd = torch.load(weights, map_location="cpu")
    if method in ("DDPG", "MADDPG"):                      # deterministic actor
        from algos.baselines import DetActor
        net = DetActor(env.obs_dim, env.act_dim); net.load_state_dict(sd["actor"]); net.eval()

        def policy(obs, adj=None):
            with torch.no_grad():
                return net(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy()
        return policy

    if method.startswith("XM"):
        net = XMActorCritic(env.obs_dim, env.act_dim, env.n, feature_groups(env.cfg))
    else:
        net = MLPActorCritic(env.obs_dim, env.act_dim, n_agents=env.n)
    net.load_state_dict(sd); net.eval()

    def policy(obs, adj=None):
        o = torch.as_tensor(obs, dtype=torch.float32)
        a = torch.as_tensor(adj, dtype=torch.float32) if adj is not None else None
        with torch.no_grad():
            act, _, _ = net.act(o, a, deterministic=True)
        return act.cpu().numpy()
    return policy


def rollout(policy, seed=1234):
    env = UAVSwarmEWEnv(TaskConfig(seed=seed))
    obs = env.reset(seed)
    pos_hist, sinr_hist, jam_hist = [env.pos.copy()], [], []
    targets = env.targets.copy()
    while True:
        adj = env._adjacency()
        obs, r, done, info = env.step(policy(obs, adj))
        pos_hist.append(env.pos.copy())
        sinr_hist.append(info["sinr"]); jam_hist.append(info["jammed_frac"])
        if done:
            break
    return np.array(pos_hist), np.array(sinr_hist), np.array(jam_hist), targets, env


def fig_trajectory(pos, targets, env, name="fig4_trajectory"):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    L = env.cfg.area_size
    N = pos.shape[1]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, N))
    for i in range(N):
        ax.plot(pos[:, i, 0], pos[:, i, 1], "-", color=cmap[i], lw=1.1, alpha=0.85, zorder=2)
        ax.plot(pos[0, i, 0], pos[0, i, 1], "o", color=cmap[i], ms=4, zorder=3)          # start
        ax.plot(pos[-1, i, 0], pos[-1, i, 1], "s", color=cmap[i], ms=5,
                markeredgecolor="k", markeredgewidth=0.4, zorder=4)                       # end
    ax.scatter(targets[:, 0], targets[:, 1], marker="*", s=150, color="#D55E00",
               edgecolor="k", linewidth=0.5, zorder=5, label="targets")
    ax.scatter(env.jammer_pos[:, 0], env.jammer_pos[:, 1], marker="X", s=70,
               color="#333", zorder=5, label="jammers")
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("UAV swarm trajectories")
    ax.legend(loc="upper right", fontsize=7, markerscale=0.8)
    ax.set_aspect("equal"); F.finalize(ax)
    fig.tight_layout()
    for d in ("figures", "paper/figures"):
        fig.savefig(f"{d}/{name}.png", dpi=300)
    plt.close(fig)


def _smooth(y, w=11):
    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    w = min(w, len(y) | 1); half = w // 2
    out = np.convolve(y, np.ones(w) / w, mode="same")
    for i in range(half):
        out[i] = y[:i + half + 1].mean(); out[-(i + 1)] = y[-(i + half + 1):].mean()
    return out


def fig_sinr_compare(curves, name="fig6_sinr_jamming"):
    """curves: dict method -> sinr array. Smoothed comparison under jamming."""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for m, sinr in curves.items():
        t = np.arange(len(sinr)); c = F.color(m)
        lw = 2.2 if m.startswith("XM") else 1.5
        ax.plot(t, _smooth(sinr), color=c, lw=lw, label=f"{m} ({np.mean(sinr):.1f} dB)",
                zorder=3 if m.startswith("XM") else 2)
    ax.set_xlabel("Time step"); ax.set_ylabel("SINR (dB, smoothed)")
    ax.set_title("SINR under adaptive jamming")
    ax.legend(loc="lower right", fontsize=7); F.finalize(ax)
    fig.tight_layout()
    for d in ("figures", "paper/figures"):
        fig.savefig(f"{d}/{name}.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="XM-MADRL")   # method for the trajectory figure
    ap.add_argument("--compare", default="MADDPG")    # 2nd method for SINR comparison
    args = ap.parse_args()

    def wpath(m):
        c = sorted(glob.glob(f"results/{m}_seed*.pt"))
        return c[0] if c else ""

    env0 = UAVSwarmEWEnv(TaskConfig(seed=1234))
    # --- trajectory (proposed method if available, else fall back) --------- #
    traj_m = args.method if wpath(args.method) else "PPO"
    if wpath(traj_m):
        pos, sinr_a, jam, targets, env = rollout(load_policy(traj_m, wpath(traj_m), env0))
        fig_trajectory(pos, targets, env)
    # --- SINR comparison across methods ----------------------------------- #
    curves = {}
    for m in [args.method, args.compare]:
        if wpath(m):
            _, sinr, _, _, _ = rollout(load_policy(m, wpath(m), UAVSwarmEWEnv(TaskConfig(seed=1234))))
            curves[m] = sinr
    if curves:
        fig_sinr_compare(curves)
    print(f"wrote fig4_trajectory.png (from {traj_m}), fig6_sinr_jamming.png ({list(curves)})")


if __name__ == "__main__":
    main()
