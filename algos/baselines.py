"""Baseline reinforcement-learning algorithms.

Two families, both trained on the *same* environment with the *same* evaluation
protocol as the proposed method:

  * on-policy  : ``PPO`` and ``A2C`` (Actor-Critic) -- built on the shared MAPPO
                 machinery with a plain MLP backbone (no GNN, no meta-learning);
  * off-policy : ``DDPG`` (independent, decentralised Q) and ``MADDPG``
                 (centralised joint Q) -- genuine deterministic actor-critic with
                 replay buffers, target networks and exploration noise.

``make_baseline`` returns a trainer exposing a common interface:
    ``learn(total_steps, log_every)`` -> history list
    ``evaluate(env, episodes)``       -> metrics dict
so ``train.py`` can drive every algorithm identically.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate import evaluate_policy


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class DetActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim], out_act=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)                 # in (-1, 1); env decodes movement+channel


class QCritic(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden, 1])

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# Replay buffer (stores full joint transitions)
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque = deque(maxlen=capacity)

    def add(self, obs, act, rew, nobs, done):
        self.buf.append((obs, act, rew, nobs, done))

    def sample(self, batch: int):
        idx = np.random.randint(0, len(self.buf), size=batch)
        o, a, r, no, d = zip(*[self.buf[i] for i in idx])
        return (np.stack(o), np.stack(a), np.stack(r), np.stack(no), np.stack(d))

    def __len__(self):
        return len(self.buf)


@dataclass
class DDPGConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.01
    batch: int = 256
    warmup: int = 2000
    noise: float = 0.2
    capacity: int = 100_000
    centralised: bool = False


class DDPGTrainer:
    """Parameter-shared DDPG / MADDPG for a homogeneous UAV swarm."""

    def __init__(self, env_fn: Callable, cfg: DDPGConfig, device: str = "cpu", name: str = "DDPG"):
        self.env = env_fn()
        self.env_fn = env_fn
        self.cfg = cfg
        self.device = device
        self.name = name
        o, a, n = self.env.obs_dim, self.env.act_dim, self.env.n

        self.actor = DetActor(o, a).to(device)
        self.actor_t = DetActor(o, a).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        # decentralised critic: Q(o, a) ; centralised: Q(O_all, A_all)
        q_in = (o + a) * n if cfg.centralised else (o + a)
        self.critic = QCritic(q_in).to(device)
        self.critic_t = QCritic(q_in).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.capacity)
        self.history: List[Dict] = []

    def _q_input(self, obs, act):
        # obs,act: (B, N, .) tensors
        if self.cfg.centralised:
            B = obs.size(0)
            return torch.cat([obs.reshape(B, -1), act.reshape(B, -1)], dim=-1)
        return torch.cat([obs, act], dim=-1)     # (B, N, o+a) -> critic broadcasts

    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def learn(self, total_steps: int, log_every: int = 2000):
        env = self.env
        obs = env.reset()
        ep_ret = np.zeros(env.n)
        recent = deque(maxlen=20)
        for step in range(total_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                act = self.actor(obs_t).cpu().numpy()
            if step < self.cfg.warmup:
                act = np.random.uniform(-1, 1, size=act.shape)
            else:
                act = np.clip(act + np.random.normal(0, self.cfg.noise, act.shape), -1, 1)
            nobs, rew, done, info = env.step(act)
            self.replay.add(obs, act, rew, nobs, np.full(env.n, float(done)))
            ep_ret += rew
            obs = nobs
            if done:
                if "episode" in info:
                    recent.append(info["episode"]["mission_success"])
                obs = env.reset(); ep_ret = np.zeros(env.n)

            if len(self.replay) >= max(self.cfg.batch, self.cfg.warmup):
                self._train_step()

            if (step + 1) % log_every == 0:
                self.history.append({
                    "step": step + 1,
                    "mission_success": float(np.mean(recent)) if recent else 0.0,
                })
        return self.history

    def _train_step(self):
        o, a, r, no, d = self.replay.sample(self.cfg.batch)
        o = torch.as_tensor(o, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        no = torch.as_tensor(no, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            na = self.actor_t(no)
            q_next = self.critic_t(self._q_input(no, na))
            if self.cfg.centralised:
                r_c = r.mean(dim=1); d_c = d[:, 0]
                target = r_c + self.cfg.gamma * (1 - d_c) * q_next
            else:
                target = r + self.cfg.gamma * (1 - d) * q_next
        q = self.critic(self._q_input(o, a))
        loss_c = F.mse_loss(q, target)
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()

        # actor: maximise Q under current policy
        pi = self.actor(o)
        loss_a = -self.critic(self._q_input(o, pi)).mean()
        self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()

        self._soft_update(self.actor, self.actor_t)
        self._soft_update(self.critic, self.critic_t)

    def evaluate(self, env, episodes: int = 20):
        def policy(obs, adj=None):
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                return self.actor(o).cpu().numpy()
        return evaluate_policy(policy, env, episodes)

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}


def make_baseline(name: str, env_fn: Callable, device: str = "cpu", total_steps: int = 200_000):
    """Factory used by train.py. Returns (trainer, kind) where kind in {'onpolicy','offpolicy'}."""
    name = name.upper()
    if name in ("DDPG", "MADDPG"):
        cfg = DDPGConfig(centralised=(name == "MADDPG"))
        return DDPGTrainer(env_fn, cfg, device, name=name), "offpolicy"
    raise ValueError(f"Off-policy factory got unknown baseline {name}")
