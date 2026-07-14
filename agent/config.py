
"""
Centralized configuration for the optimization agent.

All paths and knobs come from environment variables with sensible defaults, so
the agent works out of the box after ``uv sync``.  A ``.env`` file (loaded via
python-dotenv in :mod:`agent.__init__`) is honoured.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the mutable model project (model.py, loss.py, run.py, …).
MODEL_DIR = Path(
    os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "model_dir")
).resolve()

# Wiki describing the problem to the LLM (lives inside the model dir).
WIKI_FILE = MODEL_DIR / os.environ.get("WIKI_FILE", "wiki.md")

# Long-term experiment memory (agent-owned, inside the model dir).
LOGS_FILE = MODEL_DIR / os.environ.get("LOGS_FILE", "logs.md")

# Where the training script writes per-run metrics (LocalTracker reads these).
RUNS_DIR = MODEL_DIR / os.environ.get("RUNS_DIR", "runs")

# PERSISTENT, append-only run archive — lives OUTSIDE model_dir so git resets,
# `git clean`, and reverts never touch it. This is the durable record for
# analysis / papers. Every experiment (kept, reverted, or crashed) is appended
# here with its edits, score, cost, and metrics. Default: <project>/history.
ARCHIVE_DIR = Path(
    os.environ.get("AUTORESEARCH_HISTORY", MODEL_DIR.parent / "history")
).resolve()

# Python interpreter that runs training — the model dir's own venv if synced,
# else fall back to the ambient ``python3`` (works where torch/sklearn are
# already installed). Overridable via EXEC_COMMAND (space-separated).
_MODEL_PYTHON = MODEL_DIR / ".venv" / "bin" / "python"
_model_python = str(_MODEL_PYTHON) if _MODEL_PYTHON.exists() else "python3"
_default_exec = f"{_model_python} run.py"
EXEC_COMMAND = os.environ.get("EXEC_COMMAND", _default_exec).split()

# Per-run wall-clock cap (seconds) — a cheap-agent guardrail against a mutation
# that accidentally makes training loop forever.
RUN_TIMEOUT_SEC = int(os.environ.get("RUN_TIMEOUT_SEC", "900"))

# ---------------------------------------------------------------------------
# Experiment tracking (pluggable — see agent/tracker.py)
# ---------------------------------------------------------------------------

# "local" (default, zero-dep) or "clearml" (future).
TRACKER = os.environ.get("TRACKER", "local")

# Metric to optimise and its direction.
TARGET_METRIC = os.environ.get("TARGET_METRIC", "val/auprc")
TARGET_DIRECTION = os.environ.get("TARGET_DIRECTION", "max")  # "max" or "min"

# ---------------------------------------------------------------------------
# Scoring thresholds (Section 4 of TASK.md)
# ---------------------------------------------------------------------------

STATISTICAL_DELTA = float(os.environ.get("STATISTICAL_DELTA", "0.03"))  # 3 %

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_MODEL = os.environ.get("MAIN_MODEL", "anthropic/claude-sonnet-4.6")
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "medium")
MAX_VALIDATION_RETRIES = int(os.environ.get("MAX_VALIDATION_RETRIES", "3"))

# Cap how much conversation history we carry — token-cost control.  Older
# turns are dropped; the durable memory lives in logs.md, which is always
# re-summarised into the context.
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "8"))

# ---------------------------------------------------------------------------
# Mutable files & atomic target groups (Section 2 of TASK.md)
# ---------------------------------------------------------------------------

# Each experiment must touch files from exactly ONE group.  For this tabular
# task each lever is a single file; the transforms/config pair is a tandem.
TARGET_GROUPS: list[set[str]] = [
    {"model.py"},       # architecture
    {"loss.py"},        # loss function (key lever for class imbalance)
    {"optimizer.py"},   # optimiser + LR schedule
    {"transforms.py", "config.yaml"},  # feature engineering + hyperparameters
]

# Flattened set for quick membership tests.
ALLOWED_FILES: set[str] = {f for g in TARGET_GROUPS for f in g}
