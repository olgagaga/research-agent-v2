"""
Optimizer + LR schedule — AGENT-EDITABLE.

Contract (keep it): build_optimizer(params) -> (optimizer, scheduler_or_None).
The scheduler (if any) is stepped once per epoch by the harness.

Baseline: plain SGD, no schedule, modest LR. Try Adam/AdamW, weight decay,
cosine/step schedules, different LRs.
"""

from __future__ import annotations

import torch


def build_optimizer(params):
    # Use AdamW with a smaller learning rate and lighter weight decay.
    # Rationale: previous AdamW at lr=1e-3, wd=1e-4 gave good gains; lowering the lr
    # and reducing weight decay often improves stability and minority-class
    # performance for small tabular networks.
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-5)
    # Keep cosine annealing stepped once per epoch by the harness. We intentionally
    # keep T_max=40 (the prior change to T_max=120 was tried and reverted).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    return optimizer, scheduler
