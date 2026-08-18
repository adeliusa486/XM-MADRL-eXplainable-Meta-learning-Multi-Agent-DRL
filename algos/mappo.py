"""Multi-Agent PPO trainer with optional MAML meta-learning.

The same class trains:
  * the proposed XM-MADRL policy (transformer + GNN backbone), and
  * the PPO / A2C / MADDPG / DDPG-style baselines (plain MLP backbone),
so every method shares an identical optimisation pipeline and only the network
architecture and the meta-learning switch differ -- a fair comparison.

Meta-learning uses first-order MAML (FOMAML): for each meta-iteration we sample
several tasks from the environment distribution, take an inner PPO adaptation
step per task, and apply the averaged adapted gradients to the meta-parameters.
Setting ``meta=False`` recovers standard MAPPO.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Rollout storage
# --------------------------------------------------------------------------- #
class RolloutBuffer:
    def __init__(self):
        self.obs, self.adj, self.act, self.logp = [], [], [], []
        self.rew, self.val, self.done = [], [], []

    def add(self, obs, adj, act, logp, rew, val, done):
        self.obs.append(obs); self.adj.append(adj); self.act.append(act)
        self.logp.append(logp); self.rew.append(rew); self.val.append(val)
        self.done.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Generalised Advantage Estimation. All inputs are (T, N) numpy arrays."""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 256
    # meta-learning
    meta: bool = False
    meta_tasks: int = 4
    inner_lr: float = 0.01
    inner_steps: int = 1


class MAPPO:
    def __init__(self, policy: nn.Module, env_fn: Callable, cfg: PPOConfig, device: str = "cpu",
                 make_task_env: Optional[Callable] = None):
        self.policy = policy.to(device)
        self.env_fn = env_fn                     # () -> env (fixed eval/train task)
        self.make_task_env = make_task_env       # (rng) -> env (random meta task)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.history: List[Dict] = []

    # -- rollout ------------------------------------------------------------ #
    def collect(self, env, buffer: RolloutBuffer, steps: int, obs: np.ndarray):
        for _ in range(steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            adj_np = env._adjacency() if hasattr(env, "_adjacency") else None
            adj_t = torch.as_tensor(adj_np, device=self.device) if adj_np is not None else None
            with torch.no_grad():
                action, logp, value = self.policy.act(obs_t, adj_t)
            a_np = action.cpu().numpy()
            nobs, rew, done, info = env.step(a_np)
            buffer.add(obs, adj_np, a_np, logp.cpu().numpy(), rew,
                       value.cpu().numpy(), np.full(env.n, float(done)))
            obs = nobs
            if done:
                obs = env.reset()
                if "episode" in info:
                    self._last_ep = info["episode"]
        return obs

    # -- one PPO update on a buffer ----------------------------------------- #
    def _ppo_loss(self, policy, batch):
        obs, adj, act, old_logp, adv, ret = batch
        logp, entropy, value = self._eval_batch(policy, obs, adj, act)
        ratio = torch.exp(logp - old_logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        s1 = ratio * adv
        s2 = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv
        policy_loss = -torch.min(s1, s2).mean()
        value_loss = ((value - ret) ** 2).mean()
        loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy.mean()
        return loss, policy_loss.item(), value_loss.item()

    def _eval_batch(self, policy, obs, adj, act):
        """Vectorised evaluation of a (T, N) rollout in a single forward pass."""
        obs_TN = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=self.device)
        act_TN = torch.as_tensor(np.stack(act), dtype=torch.float32, device=self.device)
        if adj[0] is not None:
            adj_TNN = torch.as_tensor(np.stack(adj), dtype=torch.float32, device=self.device)
        else:
            T, N = obs_TN.shape[0], obs_TN.shape[1]
            adj_TNN = torch.eye(N, device=self.device).expand(T, N, N)
        return policy.evaluate_actions_batch(obs_TN, adj_TNN, act_TN)

    def _prep_batch(self, buffer: RolloutBuffer, last_value):
        rew = np.array(buffer.rew, dtype=np.float32)
        val = np.array(buffer.val, dtype=np.float32)
        done = np.array(buffer.done, dtype=np.float32)
        adv, ret = compute_gae(rew, val, done, last_value, self.cfg.gamma, self.cfg.lam)
        old_logp = torch.as_tensor(np.concatenate(buffer.logp), device=self.device)
        adv_t = torch.as_tensor(adv.reshape(-1), device=self.device)
        ret_t = torch.as_tensor(ret.reshape(-1), device=self.device)
        return (buffer.obs, buffer.adj, buffer.act, old_logp, adv_t, ret_t)

    def update(self, buffer: RolloutBuffer, last_value):
        batch = self._prep_batch(buffer, last_value)
        stats = {}
        for _ in range(self.cfg.epochs):
            loss, pl, vl = self._ppo_loss(self.policy, batch)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
            self.opt.step()
            stats = {"policy_loss": pl, "value_loss": vl}
        return stats

    # -- MAML meta-update --------------------------------------------------- #
    def meta_update(self, rng):
        """First-order MAML step across sampled tasks."""
        meta_grads = [torch.zeros_like(p) for p in self.policy.parameters()]
        task_returns = []
        for _ in range(self.cfg.meta_tasks):
            env = self.make_task_env(rng)
            obs = env.reset()
            # inner adaptation on a fast copy
            fast = copy.deepcopy(self.policy)
            inner_opt = torch.optim.SGD(fast.parameters(), lr=self.cfg.inner_lr)
            for _ in range(self.cfg.inner_steps):
                buf = RolloutBuffer()
                obs = self.collect_with(fast, env, buf, self.cfg.rollout_steps, obs)
                last_v = self._bootstrap_value(fast, env, obs)
                batch = self._prep_batch(buf, last_v)
                loss, _, _ = self._ppo_loss(fast, batch)
                inner_opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(fast.parameters(), self.cfg.max_grad_norm)
                inner_opt.step()
            # query set: evaluate adapted policy, accumulate meta-gradient
            qbuf = RolloutBuffer()
            obs = self.collect_with(fast, env, qbuf, self.cfg.rollout_steps, obs)
            last_v = self._bootstrap_value(fast, env, obs)
            batch = self._prep_batch(qbuf, last_v)
            loss, _, _ = self._ppo_loss(fast, batch)
            fast.zero_grad(); loss.backward()
            for mg, fp in zip(meta_grads, fast.parameters()):
                if fp.grad is not None:
                    mg += fp.grad.detach() / self.cfg.meta_tasks
            task_returns.append(float(np.sum(qbuf.rew)))
        # apply averaged meta-gradient to the meta-parameters
        self.opt.zero_grad()
        for p, g in zip(self.policy.parameters(), meta_grads):
            p.grad = g.clone()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        return {"meta_return": float(np.mean(task_returns))}

    def collect_with(self, policy, env, buffer, steps, obs):
        for _ in range(steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            adj_np = env._adjacency()
            adj_t = torch.as_tensor(adj_np, device=self.device)
            with torch.no_grad():
                action, logp, value = policy.act(obs_t, adj_t)
            nobs, rew, done, info = env.step(action.cpu().numpy())
            buffer.add(obs, adj_np, action.cpu().numpy(), logp.cpu().numpy(),
                       rew, value.cpu().numpy(), np.full(env.n, float(done)))
            obs = nobs
            if done:
                obs = env.reset()
        return obs

    def _bootstrap_value(self, policy, env, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        adj_t = torch.as_tensor(env._adjacency(), device=self.device)
        with torch.no_grad():
            _, _, v = policy.act(obs_t, adj_t)
        return v.cpu().numpy()
