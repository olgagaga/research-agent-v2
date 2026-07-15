"""
Proposers — where an experiment comes from.

The loop (validate → apply → train → score → keep/revert) is held fixed; only the
*source of proposals* varies. That makes the control arm a config value
(``PROPOSER=random``) rather than a fork of the codebase — RESEARCH.md §4
("Controls") and §6.3.

Why this matters: every "the agent works" claim so far is confounded with the
harness. The fixed epoch budget, the keep/revert ratchet, and the 4-lever action
space might be doing the work, not the model's *choices*. The only way to know is
to hold the loop constant and swap the LLM for uniform sampling of the same
action space.

* :class:`LLMProposer`  — the real agent (default).
* :class:`RandomProposer` — the control: pick a random candidate edit from a
  fixed menu. No LLM call, zero cost, reproducible from ``AGENT_SEED``.

The menu deliberately spans the *same* space the LLM can reach (the four levers,
good options and bad ones alike), so the comparison isolates the model's choice —
not its access. With only 4 levers and one dominant (loss), random may do well;
that is itself the finding, and a low bar the LLM should be expected to clear.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from agent import config
from agent.schemas import EditOp, ExperimentPlan, FileEdits

log = logging.getLogger(__name__)

_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cost": 0.0}


class Proposer(Protocol):
    """Anything that can propose the next experiment."""

    last_usage: Dict[str, Any]

    def propose(self, model_dir: Path, feedback: Optional[str]) -> ExperimentPlan:
        ...


# --------------------------------------------------------------------------- LLM


class LLMProposer:
    """The real agent: build context, ask the LLM for a structured plan."""

    def __init__(self, llm):
        self.llm = llm
        self.last_usage: Dict[str, Any] = {}

    def propose(self, model_dir: Path, feedback: Optional[str]) -> ExperimentPlan:
        # Imported lazily: orchestrator imports this module, so a module-level
        # import here would be circular.
        from agent.orchestrator import build_context

        messages = build_context(model_dir, feedback)
        plan = self.llm.chat_structured(
            messages=messages,
            response_format=ExperimentPlan,
            model=config.LLM_MODEL,
            reasoning_effort=config.REASONING_EFFORT,
        )
        self.last_usage = getattr(self.llm, "last_usage", {}) or {}
        return plan


# --------------------------------------------------------------------------- random

def _fe(filename: str, ops: List[EditOp]) -> FileEdits:
    return FileEdits(filename=filename, edit_list=ops)


def _body(target: str, content: str) -> EditOp:
    return EditOp(operation="replace_function_body", target=target, content=content)


# Candidate experiments. Each entry is (lever, description, [FileEdits]).
# Bodies use ONLY names already imported by the baseline file (loss.py: np/torch/nn,
# model.py: torch/nn, optimizer.py: torch, transforms.py: np) so they apply to any
# state of the file — replace_function_body targets by name, not by line.
def _menu() -> List[tuple]:
    m: List[tuple] = []

    # ---- loss.py -----------------------------------------------------------
    m.append(("loss.py", "plain BCEWithLogitsLoss (no imbalance handling)", [
        _fe("loss.py", [_body("build_loss",
            "criterion = nn.BCEWithLogitsLoss()\n"
            "def loss_fn(logits, targets):\n    return criterion(logits, targets)\n"
            "return loss_fn")])]))
    m.append(("loss.py", "BCE with pos_weight = n_neg/n_pos", [
        _fe("loss.py", [_body("build_loss",
            "y = np.asarray(y_train)\n"
            "pos = float(y.sum()); neg = float(len(y) - pos)\n"
            "pw = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)\n"
            "criterion = nn.BCEWithLogitsLoss(pos_weight=pw)\n"
            "def loss_fn(logits, targets):\n    return criterion(logits, targets)\n"
            "return loss_fn")])]))
    m.append(("loss.py", "pos-weighted focal loss (gamma=2)", [
        _fe("loss.py", [_body("build_loss",
            "y = np.asarray(y_train)\n"
            "pos = float(y.sum()); neg = float(len(y) - pos)\n"
            "pw = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)\n"
            "def loss_fn(logits, targets):\n"
            "    bce = nn.functional.binary_cross_entropy_with_logits(\n"
            "        logits, targets, pos_weight=pw, reduction='none')\n"
            "    p = torch.sigmoid(logits)\n"
            "    pt = targets * p + (1 - targets) * (1 - p)\n"
            "    return (((1 - pt) ** 2.0) * bce).mean()\n"
            "return loss_fn")])]))
    m.append(("loss.py", "focal loss (gamma=1, no pos_weight)", [
        _fe("loss.py", [_body("build_loss",
            "def loss_fn(logits, targets):\n"
            "    bce = nn.functional.binary_cross_entropy_with_logits(\n"
            "        logits, targets, reduction='none')\n"
            "    p = torch.sigmoid(logits)\n"
            "    pt = targets * p + (1 - targets) * (1 - p)\n"
            "    return (((1 - pt) ** 1.0) * bce).mean()\n"
            "return loss_fn")])]))

    # ---- optimizer.py ------------------------------------------------------
    for desc, code in [
        ("SGD lr=0.01", "optimizer = torch.optim.SGD(params, lr=0.01)\nscheduler = None\nreturn optimizer, scheduler"),
        ("SGD lr=0.1 + momentum 0.9", "optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)\nscheduler = None\nreturn optimizer, scheduler"),
        ("Adam lr=1e-3", "optimizer = torch.optim.Adam(params, lr=1e-3)\nscheduler = None\nreturn optimizer, scheduler"),
        ("Adam lr=1e-2", "optimizer = torch.optim.Adam(params, lr=1e-2)\nscheduler = None\nreturn optimizer, scheduler"),
        ("AdamW lr=1e-3 wd=1e-4 + cosine", "optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)\nscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)\nreturn optimizer, scheduler"),
    ]:
        m.append(("optimizer.py", desc, [_fe("optimizer.py", [_body("build_optimizer", code)])]))

    # ---- model.py ----------------------------------------------------------
    for desc, code in [
        ("tiny MLP (16)", "return nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, 1))"),
        ("MLP 64", "return nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 1))"),
        ("MLP 64-32", "return nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))"),
        ("MLP 128-64 + LayerNorm + dropout",
         "return nn.Sequential(nn.Linear(n_features, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),\n"
         "                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))"),
    ]:
        m.append(("model.py", desc, [_fe("model.py", [_body("build_model", code)])]))

    # ---- transforms.py (fit+transform must move together) ------------------
    m.append(("transforms.py", "identity preprocessing", [
        _fe("transforms.py", [
            _body("Preprocessor.fit", "pass"),
            _body("Preprocessor.transform", "return np.asarray(X, dtype=np.float32)")])]))
    m.append(("transforms.py", "per-feature standardisation", [
        _fe("transforms.py", [
            _body("Preprocessor.fit",
                  "X = np.asarray(X_train, dtype=np.float32)\n"
                  "self.mean_ = X.mean(axis=0)\n"
                  "self.std_ = X.std(axis=0)\n"
                  "self.std_[self.std_ == 0] = 1.0"),
            _body("Preprocessor.transform",
                  "X = np.asarray(X, dtype=np.float32)\n"
                  "return ((X - self.mean_) / self.std_).astype(np.float32)")])]))
    m.append(("transforms.py", "standardisation + quadratic features", [
        _fe("transforms.py", [
            _body("Preprocessor.fit",
                  "X = np.asarray(X_train, dtype=np.float32)\n"
                  "self.mean_ = X.mean(axis=0)\n"
                  "self.std_ = X.std(axis=0)\n"
                  "self.std_[self.std_ == 0] = 1.0"),
            _body("Preprocessor.transform",
                  "X = np.asarray(X, dtype=np.float32)\n"
                  "Z = ((X - self.mean_) / self.std_).astype(np.float32)\n"
                  "return np.concatenate([Z, Z ** 2], axis=1).astype(np.float32)")])]))
    m.append(("transforms.py", "oversample the positive class 5x", [
        _fe("transforms.py", [
            _body("resample",
                  "X = np.asarray(X); y = np.asarray(y)\n"
                  "pos = np.where(y == 1)[0]\n"
                  "if len(pos) == 0:\n    return X, y\n"
                  "rep = np.repeat(pos, 4)\n"
                  "Xr = np.concatenate([X, X[rep]], axis=0)\n"
                  "yr = np.concatenate([y, y[rep]], axis=0)\n"
                  "return Xr, yr")])]))
    m.append(("transforms.py", "no resampling", [
        _fe("transforms.py", [_body("resample", "return X, y")])]))

    # ---- config.yaml (same atomic group as transforms.py) ------------------
    for desc, epochs, bs in [("epochs=40, batch=256", 40, 256),
                             ("epochs=80, batch=128", 80, 128),
                             ("epochs=120, batch=64", 120, 64)]:
        m.append(("config.yaml", desc, [
            _fe("config.yaml", [EditOp(operation="replace_file", content=(
                f"epochs: {epochs}\nbatch_size: {bs}\nseed: 0\n"))])]))
    return m


MENU = _menu()


class RandomProposer:
    """Control arm: uniform choice from a fixed menu of candidate edits.

    No LLM call, no cost. Reproducible from ``AGENT_SEED`` so a trial can be
    replayed exactly. Ignores feedback — it has no ability to learn, which is the
    point: it isolates how much of the loop's performance is the *loop*.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.last_usage: Dict[str, Any] = dict(_ZERO_USAGE)

    def propose(self, model_dir: Path, feedback: Optional[str]) -> ExperimentPlan:
        lever, desc, edits = self.rng.choice(MENU)
        self.last_usage = dict(_ZERO_USAGE)
        return ExperimentPlan(
            reasoning=("Random control arm: uniform draw from a fixed menu of "
                       f"{len(MENU)} candidate edits spanning the same four levers "
                       "the LLM can reach. No reasoning involved by construction."),
            short_description=f"[random] {lever}: {desc}",
            edits=edits,
        )


# --------------------------------------------------------------------------- factory


def make_proposer(llm=None) -> Proposer:
    """Pick a proposer: an injected client (tests) → LLM; else ``config.PROPOSER``."""
    if llm is not None:
        return LLMProposer(llm)
    if config.PROPOSER == "random":
        log.info("Using RandomProposer (control arm, seed=%d, menu=%d)",
                 config.AGENT_SEED, len(MENU))
        return RandomProposer(seed=config.AGENT_SEED)
    from agent.llm import LLMClient
    return LLMProposer(LLMClient(api_key=None))
