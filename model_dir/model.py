"""
Model architecture — AGENT-EDITABLE.

Contract (keep it): build_model(n_features:int) -> torch.nn.Module whose
forward(x) returns raw logits of shape (B,) or (B, 1) for the positive class.

Baseline: a deliberately small single-hidden-layer MLP. Lots of headroom —
try depth, width, normalisation, dropout, residual connections, etc.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, hidden2: int = 32, dropout: float = 0.2):
        super().__init__()
        # Two-layer MLP with LayerNorm and dropout for stability/regularisation
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        # Return shape (B,) for compatibility with existing loss/metrics code
        return logits.view(-1)


def build_model(n_features: int) -> nn.Module:
    return MLP(n_features, hidden=64, hidden2=32, dropout=0.2)
