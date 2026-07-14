#!/usr/bin/env python3
"""
Training harness — FIXED (the agent must not edit this).

It wires together the four agent-editable seams and trains on CPU:

    transforms.Preprocessor  — fit/transform features        (transforms.py)
    transforms.resample      — optional train-set resampling (transforms.py)
    model.build_model        — the network                   (model.py)
    loss.build_loss          — the training loss             (loss.py)
    optimizer.build_optimizer— optimiser + LR schedule       (optimizer.py)

Contracts the seams MUST keep:
    Preprocessor().fit(X_train, y_train); Preprocessor().transform(X) -> 2D array
    resample(X, y) -> (X, y)
    build_model(n_features:int) -> nn.Module, forward(x)->logits shape (B,) or (B,1)
    build_loss(y_train:np.ndarray) -> callable(logits:(B,), targets:(B,) float)->scalar
    build_optimizer(params) -> (torch.optim.Optimizer, scheduler_or_None)

Output (read by the pluggable Tracker):
    runs/<run_id>/metrics.json  and  a line "RUN_ID=<run_id>" on stdout.

The metric is AUPRC (average precision) on the rare positive class — the right
metric for ~2% prevalence, where plain accuracy is meaningless.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

import data as data_mod
import loss as loss_mod
import model as model_mod
import optimizer as optim_mod
import transforms as transforms_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run")

HERE = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config() -> dict:
    cfg_path = HERE / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    # Defaults (config.yaml may override; harness clamps epochs for cost safety).
    cfg.setdefault("epochs", 40)
    cfg.setdefault("batch_size", 256)
    cfg.setdefault("seed", 0)
    cfg["epochs"] = int(min(cfg["epochs"], 200))  # cheap-agent guardrail
    return cfg


@torch.no_grad()
def predict_proba(net: torch.nn.Module, X: torch.Tensor) -> np.ndarray:
    net.eval()
    logits = net(X).reshape(-1)
    return torch.sigmoid(logits).cpu().numpy()


def main() -> int:
    cfg = load_config()
    set_seed(int(cfg["seed"]))
    run_id = uuid.uuid4().hex[:12]
    log.info("run_id=%s config=%s", run_id, cfg)

    # --- data (fixed split) --------------------------------------------------
    ds = data_mod.load_dataset()

    # --- features (agent-editable) ------------------------------------------
    pre = transforms_mod.Preprocessor()
    pre.fit(ds.X_train, ds.y_train)
    Xtr = np.asarray(pre.transform(ds.X_train), dtype=np.float32)
    Xva = np.asarray(pre.transform(ds.X_val), dtype=np.float32)
    Xte = np.asarray(pre.transform(ds.X_test), dtype=np.float32)
    ytr = ds.y_train

    # optional resampling of the TRAIN set only
    Xtr, ytr = transforms_mod.resample(Xtr, ytr)
    Xtr = np.asarray(Xtr, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)

    n_features = Xtr.shape[1]
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    Xva_t = torch.from_numpy(Xva)
    Xte_t = torch.from_numpy(Xte)

    # --- model / loss / optimizer (agent-editable) --------------------------
    net = model_mod.build_model(n_features)
    loss_fn = loss_mod.build_loss(ds.y_train)
    optimizer, scheduler = optim_mod.build_optimizer(net.parameters())

    # --- training loop -------------------------------------------------------
    n = Xtr_t.shape[0]
    bs = int(cfg["batch_size"])
    best_val = -1.0
    best_test = float("nan")
    series: list[list[float]] = []
    last_val_loss = float("nan")

    for epoch in range(int(cfg["epochs"])):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            optimizer.zero_grad()
            logits = net(xb).reshape(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        val_p = predict_proba(net, Xva_t)
        val_auprc = float(average_precision_score(ds.y_val, val_p))
        series.append([epoch, val_auprc])
        with torch.no_grad():
            last_val_loss = float(
                loss_fn(net(Xva_t).reshape(-1), torch.from_numpy(ds.y_val.astype(np.float32)))
            )

        if val_auprc > best_val:
            best_val = val_auprc
            best_test = float(average_precision_score(ds.y_test, predict_proba(net, Xte_t)))
        log.info("epoch %3d  val/auprc=%.4f  best=%.4f", epoch, val_auprc, best_val)

    # --- write metrics (tracker-agnostic) -----------------------------------
    metrics = {
        "val/auprc": best_val,
        "test/auprc": best_test,
        "val/loss": last_val_loss,
        "pos_rate": ds.pos_rate,
    }
    run_dir = HERE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {"run_id": run_id, "metrics": metrics, "series": {"val/auprc": series}},
            indent=2,
        )
    )

    # --- Future ClearML hook (kept inert until TRACKER=clearml) -------------
    # from clearml import Task
    # task = Task.init(project_name="autoresearch", task_name=run_id)
    # for e, v in series: task.get_logger().report_scalar("val", "auprc", v, e)
    # print(f"CLEARML_TASK_ID={task.id}")

    print(f"RUN_ID={run_id}")
    print(f"val/auprc={best_val:.4f} test/auprc={best_test:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
