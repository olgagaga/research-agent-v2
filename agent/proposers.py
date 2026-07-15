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
import math
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


# --------------------------------------------------------------------- wide space
#
# The curated MENU above encodes *our* knowledge of what is worth trying — which
# is precisely the knowledge the LLM is supposed to supply. Handing it to the
# random arm gives away the thing under test, and the first control duly found no
# LLM advantage (n=5, t=0.33).
#
# The wide space keeps the same four levers but makes them genuinely large:
# hyperparameters are *sampled* and whole regions are junk (lr=1e-6, sigmoid
# activations, dropout 0.8, MSE on a classification task, feature dropping).
# It is not "random made dumber" — it is the same search problem without the
# expert prior baked in. Varying curated→wide is the independent variable; the
# LLM arm is untouched. If the gap opens up, the LLM's value is *filtering*.


def _loguniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _wide_loss(rng: random.Random) -> tuple:
    kind = rng.choice(["bce", "bce_pw", "focal", "mse", "l1", "smooth"])
    if kind == "bce":
        return "plain BCE", ("criterion = nn.BCEWithLogitsLoss()\n"
                             "def loss_fn(logits, targets):\n    return criterion(logits, targets)\n"
                             "return loss_fn")
    if kind == "bce_pw":
        mult = round(_loguniform(rng, 0.05, 100.0), 3)
        return f"BCE pos_weight×{mult}", (
            f"y = np.asarray(y_train)\npos = float(y.sum()); neg = float(len(y) - pos)\n"
            f"pw = torch.tensor([{mult} * neg / max(pos, 1.0)], dtype=torch.float32)\n"
            "criterion = nn.BCEWithLogitsLoss(pos_weight=pw)\n"
            "def loss_fn(logits, targets):\n    return criterion(logits, targets)\nreturn loss_fn")
    if kind == "focal":
        g = round(rng.uniform(0.0, 5.0), 2)
        use_pw = rng.random() < 0.5
        pw = ("y = np.asarray(y_train)\npos = float(y.sum()); neg = float(len(y) - pos)\n"
              "pw = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)\n") if use_pw else "pw = None\n"
        return f"focal γ={g}{' +pw' if use_pw else ''}", (
            pw + "def loss_fn(logits, targets):\n"
            "    bce = nn.functional.binary_cross_entropy_with_logits(\n"
            "        logits, targets, pos_weight=pw, reduction='none')\n"
            "    p = torch.sigmoid(logits)\n"
            "    pt = targets * p + (1 - targets) * (1 - p)\n"
            f"    return (((1 - pt) ** {g}) * bce).mean()\nreturn loss_fn")
    if kind == "mse":  # junk: regression loss on a detection task
        return "MSE on sigmoid (junk)", (
            "def loss_fn(logits, targets):\n"
            "    return ((torch.sigmoid(logits) - targets) ** 2).mean()\nreturn loss_fn")
    if kind == "l1":  # junk
        return "L1 on sigmoid (junk)", (
            "def loss_fn(logits, targets):\n"
            "    return (torch.sigmoid(logits) - targets).abs().mean()\nreturn loss_fn")
    s = round(rng.uniform(0.0, 0.4), 2)  # heavy smoothing on 2% positives = junk
    return f"BCE label-smoothing {s}", (
        "criterion = nn.BCEWithLogitsLoss()\n"
        "def loss_fn(logits, targets):\n"
        f"    t = targets * (1 - {s}) + 0.5 * {s}\n    return criterion(logits, t)\nreturn loss_fn")


def _wide_optimizer(rng: random.Random) -> tuple:
    opt = rng.choice(["SGD", "Adam", "AdamW", "RMSprop", "Adagrad"])
    lr = _loguniform(rng, 1e-6, 1.0)          # spans useless→divergent
    wd = rng.choice([0.0, _loguniform(rng, 1e-6, 1e-1)])
    sched = rng.choice([None, "cosine", "step"])
    if opt == "SGD":
        mom = round(rng.uniform(0.0, 0.99), 2)
        mk = f"torch.optim.SGD(params, lr={lr:.3g}, momentum={mom}, weight_decay={wd:.3g})"
    elif opt == "RMSprop":
        mk = f"torch.optim.RMSprop(params, lr={lr:.3g}, weight_decay={wd:.3g})"
    elif opt == "Adagrad":
        mk = f"torch.optim.Adagrad(params, lr={lr:.3g}, weight_decay={wd:.3g})"
    else:
        mk = f"torch.optim.{opt}(params, lr={lr:.3g}, weight_decay={wd:.3g})"
    if sched == "cosine":
        sc = "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)"
    elif sched == "step":
        sc = f"scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size={rng.choice([5,10,20])}, gamma=0.5)"
    else:
        sc = "scheduler = None"
    return (f"{opt} lr={lr:.2g} wd={wd:.2g} sched={sched}",
            f"optimizer = {mk}\n{sc}\nreturn optimizer, scheduler")


def _wide_model(rng: random.Random) -> tuple:
    depth = rng.randint(1, 4)
    width = rng.choice([2, 4, 8, 16, 32, 64, 128, 256, 512])
    act = rng.choice(["ReLU", "Tanh", "Sigmoid", "SiLU"])   # Sigmoid/Tanh = vanishing junk
    drop = rng.choice([0.0, 0.1, 0.3, 0.5, 0.8])            # 0.8 = junk
    norm = rng.choice([None, "LayerNorm", "BatchNorm1d"])
    layers, ind = [], "n_features"
    for _ in range(depth):
        layers.append(f"nn.Linear({ind}, {width})")
        if norm:
            layers.append(f"nn.{norm}({width})")
        layers.append(f"nn.{act}()")
        if drop > 0:
            layers.append(f"nn.Dropout({drop})")
        ind = str(width)
    layers.append(f"nn.Linear({ind}, 1)")
    return (f"MLP d={depth} w={width} {act} drop={drop} norm={norm}",
            "return nn.Sequential(\n    " + ",\n    ".join(layers) + ",\n)")


def _wide_transforms(rng: random.Random) -> tuple:
    kind = rng.choice(["identity", "standardize", "quadratic", "drop_features",
                       "noise_features", "resample", "undersample"])
    if kind == "identity":
        return "identity", [_body("Preprocessor.fit", "pass"),
                            _body("Preprocessor.transform", "return np.asarray(X, dtype=np.float32)")]
    fit_std = ("X = np.asarray(X_train, dtype=np.float32)\n"
               "self.mean_ = X.mean(axis=0)\nself.std_ = X.std(axis=0)\n"
               "self.std_[self.std_ == 0] = 1.0")
    if kind == "standardize":
        return "standardise", [_body("Preprocessor.fit", fit_std),
                               _body("Preprocessor.transform",
                                     "X = np.asarray(X, dtype=np.float32)\n"
                                     "return ((X - self.mean_) / self.std_).astype(np.float32)")]
    if kind == "quadratic":
        return "standardise + quadratic", [_body("Preprocessor.fit", fit_std),
            _body("Preprocessor.transform",
                  "X = np.asarray(X, dtype=np.float32)\n"
                  "Z = ((X - self.mean_) / self.std_).astype(np.float32)\n"
                  "return np.concatenate([Z, Z ** 2], axis=1).astype(np.float32)")]
    if kind == "drop_features":  # junk: throw away signal
        k = rng.randint(1, 3)
        return f"drop to first {k} features (junk)", [_body("Preprocessor.fit", "pass"),
            _body("Preprocessor.transform",
                  f"return np.asarray(X, dtype=np.float32)[:, :{k}]")]
    if kind == "noise_features":  # junk: pad with pure noise
        k = rng.choice([4, 8, 16])
        return f"append {k} noise features (junk)", [_body("Preprocessor.fit", "pass"),
            _body("Preprocessor.transform",
                  "X = np.asarray(X, dtype=np.float32)\n"
                  f"N = np.random.RandomState(0).randn(X.shape[0], {k}).astype(np.float32)\n"
                  "return np.concatenate([X, N], axis=1).astype(np.float32)")]
    if kind == "resample":
        k = rng.randint(1, 20)
        return f"oversample positives {k}x", [_body("resample",
            "X = np.asarray(X); y = np.asarray(y)\n"
            "pos = np.where(y == 1)[0]\n"
            "if len(pos) == 0:\n    return X, y\n"
            f"rep = np.repeat(pos, {k})\n"
            "return np.concatenate([X, X[rep]], axis=0), np.concatenate([y, y[rep]], axis=0)")]
    frac = round(rng.uniform(0.05, 1.0), 2)  # aggressive undersampling = junk
    return f"undersample negatives to {frac}", [_body("resample",
        "X = np.asarray(X); y = np.asarray(y)\n"
        "rs = np.random.RandomState(0)\n"
        "pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]\n"
        f"keep = rs.choice(neg, max(1, int(len(neg) * {frac})), replace=False)\n"
        "idx = np.concatenate([pos, keep])\n"
        "return X[idx], y[idx]")]


def _wide_config(rng: random.Random) -> tuple:
    # Bounded so a junk draw can't blow the wall-clock budget.
    epochs = rng.randint(5, 80)
    batch = rng.choice([64, 128, 256, 512])
    return (f"epochs={epochs}, batch={batch}",
            f"epochs: {epochs}\nbatch_size: {batch}\nseed: 0\n")


def wide_proposal(rng: random.Random) -> tuple:
    """Sample one edit from the large, un-curated action space."""
    lever = rng.choice(["loss.py", "optimizer.py", "model.py", "transforms.py", "config.yaml"])
    if lever == "loss.py":
        desc, code = _wide_loss(rng)
        return lever, desc, [_fe("loss.py", [_body("build_loss", code)])]
    if lever == "optimizer.py":
        desc, code = _wide_optimizer(rng)
        return lever, desc, [_fe("optimizer.py", [_body("build_optimizer", code)])]
    if lever == "model.py":
        desc, code = _wide_model(rng)
        return lever, desc, [_fe("model.py", [_body("build_model", code)])]
    if lever == "transforms.py":
        desc, ops = _wide_transforms(rng)
        return lever, desc, [_fe("transforms.py", ops)]
    desc, text = _wide_config(rng)
    return lever, desc, [_fe("config.yaml", [EditOp(operation="replace_file", content=text)])]


class RandomProposer:
    """Control arm: random draw from the action space. No LLM call, no cost.

    ``menu="curated"`` draws uniformly from the 21 expert-chosen edits;
    ``menu="wide"`` samples the large space (incl. junk regions). Reproducible
    from ``AGENT_SEED``. Ignores feedback — it cannot learn, which is the point:
    it isolates how much of the loop's performance is the *loop*.
    """

    def __init__(self, seed: int = 0, menu: str = "curated"):
        self.rng = random.Random(seed)
        self.menu = menu
        self.last_usage: Dict[str, Any] = dict(_ZERO_USAGE)

    def propose(self, model_dir: Path, feedback: Optional[str]) -> ExperimentPlan:
        self.last_usage = dict(_ZERO_USAGE)
        if self.menu == "wide":
            lever, desc, edits = wide_proposal(self.rng)
            why = ("Random control arm (WIDE space): uniform sample over the same four "
                   "levers with sampled hyperparameters and no expert prior — junk "
                   "regions included. No reasoning involved by construction.")
        else:
            lever, desc, edits = self.rng.choice(MENU)
            why = (f"Random control arm (CURATED menu of {len(MENU)}): uniform draw from "
                   "expert-chosen edits over the same four levers. No reasoning "
                   "involved by construction.")
        return ExperimentPlan(
            reasoning=why,
            short_description=f"[random/{self.menu}] {lever}: {desc}",
            edits=edits,
        )


# --------------------------------------------------------------------------- factory


def make_proposer(llm=None) -> Proposer:
    """Pick a proposer: an injected client (tests) → LLM; else ``config.PROPOSER``."""
    if llm is not None:
        return LLMProposer(llm)
    if config.PROPOSER == "random":
        log.info("Using RandomProposer (control arm, seed=%d, menu=%s)",
                 config.AGENT_SEED, config.RANDOM_MENU)
        return RandomProposer(seed=config.AGENT_SEED, menu=config.RANDOM_MENU)
    from agent.llm import LLMClient
    return LLMProposer(LLMClient(api_key=None))
