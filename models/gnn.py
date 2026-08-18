"""Dense graph-convolution layers for inter-UAV communication.

A small, self-contained GCN implementation (no torch-geometric dependency) that
operates on a dense adjacency matrix.  For a swarm of 8-20 agents this is both
simpler and faster than building sparse edge indices every step, and it is a
genuine graph neural network: H' = sigma( D^-1/2 (A+I) D^-1/2  H  W ).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalise_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation of a (possibly batched) adjacency matrix."""
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
    deg = adj.sum(-1).clamp(min=1e-6)
    dinv = deg.pow(-0.5)
    return dinv.unsqueeze(-1) * adj * dinv.unsqueeze(-2)


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F) ; adj_norm: (B, N, N)
        return self.lin(torch.bmm(adj_norm, x))


class DenseGCN(nn.Module):
    """Two-layer GCN used as the swarm communication module."""

    def __init__(self, dim: int, layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList([GraphConv(dim, dim) for _ in range(layers)])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)
        adj_norm = _normalise_adj(adj)
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, adj_norm)) + h          # residual message passing
        return h.squeeze(0) if single else h
