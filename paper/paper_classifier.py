"""Constellation-histogram classifier for modulation recognition.

Modulation identity lives in the *2D I/Q constellation density*, not in the raw
time series (whose per-axis mean is ~0 for symmetric constellations). We convert
each I/Q snippet to a normalized 2D histogram and classify it with a small 2D CNN
-- the standard, effective recipe.
"""
import numpy as np
import torch
import torch.nn as nn


def to_hist(X, bins=24, rng=2.5):
    """X: (n, 2, length) raw I/Q -> (n, 1, bins, bins) normalized density images."""
    out = np.empty((len(X), 1, bins, bins), dtype=np.float32)
    edges = [[-rng, rng], [-rng, rng]]
    for k, s in enumerate(X):
        h, _, _ = np.histogram2d(s[0], s[1], bins=bins, range=edges)
        m = h.max()
        out[k, 0] = (h / m) if m > 0 else h
    return out


class ConstellationCNN(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(48, n_classes)

    def forward(self, x):
        return self.head(self.net(x))
