"""
Training loss — AGENT-EDITABLE.

Contract (keep it): build_loss(y_train:np.ndarray) -> callable(logits, targets)
where logits and targets are both shape (B,) float tensors, returning a scalar
loss. y_train is passed so the loss can adapt to class imbalance (e.g. pos_weight
or focal loss).

Baseline: plain BCEWithLogitsLoss — ignores the 2% imbalance entirely, so this
is a prime lever for the agent (class weighting, focal loss, ...).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_loss(y_train: np.ndarray):
    y_train = np.asarray(y_train)
    # count positives/negatives (guard against zero positives)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0:
        pos_weight = 1.0
    else:
        # Use class-balanced weighting (effective number of samples) to compute
        # a more stable positive-class weight instead of raw n_neg/n_pos.
        # Beta close to 1 emphasises the effective-number correction for rare classes.
        beta = 0.9999
        # Effective number: 1 - beta^n (guard against exact zeros)
        effective_pos = 1.0 - (beta ** n_pos)
        effective_neg = 1.0 - (beta ** n_neg)
        # Avoid degenerate tiny denominators
        if effective_pos <= 0.0:
            effective_pos = 1e-8
        if effective_neg <= 0.0:
            effective_neg = 1e-8
        weight_pos = (1.0 - beta) / effective_pos
        weight_neg = (1.0 - beta) / effective_neg
        # pos_weight for BCEWithLogits expects weight_pos / weight_neg
        pos_weight = float(weight_pos / weight_neg)
    # Cap extremely large pos_weights to avoid instability on very tiny positive counts
    pos_weight = float(min(pos_weight, 50.0))
    # store as a tensor; we'll move it to the device of the inputs at call time
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
    # Maintain the focal focusing that previously helped (gamma=2.0)
    gamma = 2.0

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Accept logits/targets shaped (B,) or (B,1). Make them compatible.
        if logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.view(-1)
        if targets.dim() == 2 and targets.shape[1] == 1:
            targets = targets.view(-1)
        # Ensure float dtype for targets
        targets = targets.to(dtype=logits.dtype)
        # Compute per-sample BCE with pos_weight moved to the correct device
        pw = pos_weight_tensor.to(logits.device)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pw
        )
        # probability of the true class: pt = p if y==1 else (1-p)
        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        # focal factor (stronger focusing)
        focal_factor = (1.0 - pt).clamp(min=0.0) ** gamma
        loss = focal_factor * bce
        return loss.mean()

    return loss_fn
