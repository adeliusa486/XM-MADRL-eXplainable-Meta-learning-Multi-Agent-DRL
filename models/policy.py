"""Policy / value networks.

* :class:`XMActorCritic` -- the proposed XM-MADRL network:
      multimodal token embedding -> transformer encoder -> GCN swarm
      communication -> Gaussian actor + centralised critic.
  Ablation flags (`use_transformer`, `use_gnn`) let the same class reproduce the
  "-Transformer" and "-GNN" rows of the ablation study.

* :class:`MLPActorCritic` -- a plain per-agent actor-critic used by the PPO /
  A2C / MADDPG / DDPG baselines so every method is trained with the same PPO/AC
  machinery and only the architecture differs.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from .gnn import DenseGCN


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class MultiModalTokenizer(nn.Module):
    """Project each sensing modality into a shared embedding token."""

    def __init__(self, groups: Dict[str, slice], dim: int):
        super().__init__()
        self.groups = groups
        self.proj = nn.ModuleDict(
            {name: nn.Linear(sl.stop - sl.start, dim) for name, sl in groups.items()}
        )
        self.dim = dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_dim) -> tokens: (B, n_tokens, dim)
        toks = [self.proj[name](obs[:, sl]) for name, sl in self.groups.items()]
        return torch.stack(toks, dim=1)


class XMActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_agents: int,
        feature_groups: Dict[str, slice],
        hidden: int = 128,
        n_heads: int = 4,
        transformer_layers: int = 2,
        gnn_layers: int = 2,
        use_transformer: bool = True,
        use_gnn: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.use_transformer = use_transformer
        self.use_gnn = use_gnn

        self.tokenizer = MultiModalTokenizer(feature_groups, hidden)
        if use_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
                batch_first=True, dropout=0.0,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        else:
            self.fuse = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())

        if use_gnn:
            self.gnn = DenseGCN(hidden, layers=gnn_layers)

        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

        # centralised critic: consumes mean-pooled swarm feature
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    # -- feature extraction ------------------------------------------------- #
    def encode(self, obs: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """obs: (N, obs_dim) for a single swarm snapshot -> (N, hidden)."""
        tokens = self.tokenizer(obs)                       # (N, T, H)
        if self.use_transformer:
            fused = self.transformer(tokens).mean(dim=1)   # (N, H)
        else:
            fused = self.fuse(tokens.mean(dim=1))
        if self.use_gnn:
            if adj is None:
                adj = torch.ones(obs.size(0), obs.size(0), device=obs.device)
            fused = self.gnn(fused, adj)
        return fused

    # -- actor / critic ----------------------------------------------------- #
    def forward(self, obs: torch.Tensor, adj: Optional[torch.Tensor] = None):
        feat = self.encode(obs, adj)
        mu = self.mu(self.actor(feat))
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        dist = Normal(mu, std)
        value = self.critic(feat.mean(dim=0, keepdim=True)).expand(feat.size(0), 1)
        return dist, value.squeeze(-1)

    def act(self, obs: torch.Tensor, adj: Optional[torch.Tensor] = None, deterministic: bool = False):
        dist, value = self.forward(obs, adj)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value

    def evaluate_actions(self, obs, adj, actions):
        dist, value = self.forward(obs, adj)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value

    # -- batched evaluation over a whole rollout (T timesteps, N agents) ----- #
    def evaluate_actions_batch(self, obs_TN, adj_TNN, act_TN):
        """obs_TN:(T,N,obs) adj_TNN:(T,N,N) act_TN:(T,N,act) -> flat (T*N,)."""
        T, N, _ = obs_TN.shape
        flat = obs_TN.reshape(T * N, -1)
        tokens = self.tokenizer(flat)                      # (T*N, tok, H)
        if self.use_transformer:
            fused = self.transformer(tokens).mean(dim=1)   # (T*N, H)
        else:
            fused = self.fuse(tokens.mean(dim=1))
        fused = fused.reshape(T, N, -1)
        if self.use_gnn:
            fused = self.gnn(fused, adj_TNN)               # (T, N, H)
        mu = self.mu(self.actor(fused))                    # (T, N, act)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        dist = Normal(mu, std)
        logp = dist.log_prob(act_TN).sum(-1)               # (T, N)
        entropy = dist.entropy().sum(-1)                   # (T, N)
        value = self.critic(fused.mean(dim=1)).squeeze(-1) # (T,)
        value = value.unsqueeze(1).expand(T, N)            # (T, N)
        return logp.reshape(-1), entropy.reshape(-1), value.reshape(-1)


class MLPActorCritic(nn.Module):
    """Plain per-agent Gaussian actor-critic (baseline backbone)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, centralised_critic: bool = False, n_agents: int = 1):
        super().__init__()
        self.centralised = centralised_critic
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        critic_in = obs_dim * n_agents if centralised_critic else obs_dim
        self.critic = nn.Sequential(
            nn.Linear(critic_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.n_agents = n_agents

    def _value(self, obs):
        if self.centralised:
            flat = obs.reshape(1, -1).expand(obs.size(0), -1)
            return self.critic(flat).squeeze(-1)
        return self.critic(obs).squeeze(-1)

    def forward(self, obs, adj=None):
        h = self.actor(obs)
        mu = self.mu(h)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        return Normal(mu, std), self._value(obs)

    def act(self, obs, adj=None, deterministic: bool = False):
        dist, value = self.forward(obs)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value

    def evaluate_actions(self, obs, adj, actions):
        dist, value = self.forward(obs)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value

    def evaluate_actions_batch(self, obs_TN, adj_TNN, act_TN):
        T, N, _ = obs_TN.shape
        flat_o = obs_TN.reshape(T * N, -1)
        flat_a = act_TN.reshape(T * N, -1)
        h = self.actor(flat_o)
        mu = self.mu(h)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        dist = Normal(mu, std)
        logp = dist.log_prob(flat_a).sum(-1)
        entropy = dist.entropy().sum(-1)
        if self.centralised:
            flat = obs_TN.reshape(T, -1).unsqueeze(1).expand(T, N, -1).reshape(T * N, -1)
            value = self.critic(flat).squeeze(-1)
        else:
            value = self.critic(flat_o).squeeze(-1)
        return logp, entropy, value
