"""
Persistent, append-only run archive.

The working ``logs.md`` and ``runs/`` live *inside* ``model_dir`` and are
gitignored, so they get wiped by ``git reset --hard`` / ``git clean`` on reverts
(and by anything that resets the model dir). That's fine for the agent's
short-term memory, but it means run history is **not durable**.

This module writes an immutable, append-only copy of every experiment to a
directory *outside* ``model_dir`` (``config.ARCHIVE_DIR``, default
``<project>/history``). Nothing in the loop ever deletes it. Each experiment —
kept, reverted, or crashed — becomes one JSON line in ``experiments.jsonl`` with
its full provenance: the proposed edits, score, status, cost/tokens, and the
metrics/training-curve. This is the source of truth for later analysis, papers,
and debugging.

Format: newline-delimited JSON (JSONL) — append-only, corruption-resistant
(a torn final line loses one record, not the file), trivially loadable with
``pandas.read_json(path, lines=True)``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def new_session_id() -> str:
    """A sortable, ~unique session id, e.g. ``20260714T230501Z-a1b2``.

    The random suffix prevents collisions when several agents in a population
    start within the same second.
    """
    import uuid
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]


class RunArchive:
    """Append-only writer for experiment records."""

    def __init__(self, root: Path, session: str, model: str = "", task: str = "",
                 agent: str = "solo", config_snapshot: Optional[Dict[str, Any]] = None,
                 variant: str = "default", trial: int = 0):
        self.root = Path(root)
        self.session = session
        self.model = model
        self.task = task
        self.agent = agent
        # The independent variables of this run (RESEARCH.md §6): without them the
        # archive says what the agent did but not what it was, so sessions can't
        # be compared and no A/B question is decidable.
        self.config_snapshot = config_snapshot or {}
        self.variant = variant
        self.trial = trial
        self.root.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.root / "experiments.jsonl"
        # Per-session metadata file (human-scannable index of sessions).
        self._write_session_header()

    def _write_session_header(self) -> None:
        idx = self.root / "sessions.jsonl"
        rec = {
            "session": self.session,
            "agent": self.agent,
            "variant": self.variant,
            "trial": self.trial,
            "started": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "task": self.task,
            "config": self.config_snapshot,
        }
        try:
            with open(idx, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:
            log.debug("Could not write session header", exc_info=True)

    def record(self, rec: Dict[str, Any]) -> None:
        """Append one experiment record. Never raises into the caller."""
        full = {
            "session": self.session,
            "agent": self.agent,
            "variant": self.variant,
            "trial": self.trial,
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            **rec,
        }
        try:
            with open(self.jsonl, "a") as fh:
                fh.write(json.dumps(full, default=str) + "\n")
        except Exception:
            log.exception("Failed to write archive record (continuing)")

    # Convenience builder used by the orchestrator ------------------------------

    def record_experiment(
        self,
        *,
        iteration: int,
        target: str,
        status: str,
        plan: Any,  # ExperimentPlan
        score: Optional[float] = None,
        best_before: Optional[float] = None,
        run_id: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        series: Optional[Dict[str, list]] = None,
        cost: Optional[float] = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        tokens_cached: int = 0,
        kept: bool = False,
        error: Optional[str] = None,
    ) -> None:
        try:
            edits = [fe.model_dump() for fe in getattr(plan, "edits", [])]
        except Exception:
            edits = []
        self.record({
            "iteration": iteration,
            "target": target,
            "status": status,
            "kept": kept,
            "score": score,
            "best_before": best_before,
            "run_id": run_id,
            "short_description": getattr(plan, "short_description", None),
            "reasoning": getattr(plan, "reasoning", None),
            "edits": edits,
            "metrics": metrics,
            "series": series,
            "cost": cost,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_cached": tokens_cached,
            "error": error,
        })
