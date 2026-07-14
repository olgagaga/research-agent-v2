"""
Experiment-tracking abstraction.

The orchestrator must not care *where* experiment metrics come from.  It gets a
``run handle`` (a short string the training subprocess prints on stdout) and asks
a :class:`Tracker` to turn that handle into a :class:`RunResult`.

Two backends:

* :class:`LocalTracker` — the default, zero-dependency backend.  The training
  script writes ``runs/<run_id>/metrics.json`` and prints ``RUN_ID=<run_id>``.
  We read the JSON.  No external service, no cost.

* :class:`ClearMLTracker` — a stub wired for the future.  ClearML is an optional
  dependency; nothing imports it unless this backend is selected.  The training
  script would print ``CLEARML_TASK_ID=<id>`` and this backend would fetch
  scalars via the ClearML SDK.

Swapping backends is a one-line config change (``TRACKER=clearml``); the
orchestrator, editor, and model code are untouched.  This is the "integrate
ClearML later without a rewrite" requirement.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Normalised result of one training run, backend-agnostic."""

    run_id: str
    # metric name -> best value over the run (e.g. {"val/auprc": 0.62})
    metrics: Dict[str, float] = field(default_factory=dict)
    # full time series if the backend has it: name -> [(step, value), ...]
    series: Dict[str, list] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)

    def score(self, metric: str, direction: str = "max") -> Optional[float]:
        """Best value of *metric* (already reduced by the backend)."""
        val = self.metrics.get(metric)
        if val is None:
            log.warning(
                "Metric %r not in run %s. Available: %s",
                metric, self.run_id, list(self.metrics),
            )
        return val


# ---------------------------------------------------------------------------
# Handle parsing — how a run identifies itself on stdout
# ---------------------------------------------------------------------------

# Each backend advertises the stdout marker its training script prints.
_HANDLE_PATTERNS = {
    "local": re.compile(r"RUN_ID[=:]\s*(\S+)"),
    "clearml": re.compile(r"CLEARML_TASK_ID[=:]\s*(\S+)"),
}


class Tracker(Protocol):
    """Protocol every tracking backend implements."""

    #: regex-key used to pull the run handle out of subprocess stdout
    handle_key: str

    def parse_handle(self, stdout: str) -> Optional[str]:
        ...

    def fetch(self, handle: str) -> RunResult:
        ...


# ---------------------------------------------------------------------------
# Local (default) backend
# ---------------------------------------------------------------------------


class LocalTracker:
    """Reads metrics the training script wrote to ``runs/<id>/metrics.json``.

    Expected JSON shape (written by ``model_dir/run.py``)::

        {
          "run_id": "ab12cd",
          "metrics": {"val/auprc": 0.61, "test/auprc": 0.58, "val/loss": 0.12},
          "series":  {"val/auprc": [[0, 0.1], [1, 0.4], ...]}   # optional
        }
    """

    handle_key = "local"

    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)

    def parse_handle(self, stdout: str) -> Optional[str]:
        m = _HANDLE_PATTERNS["local"].search(stdout or "")
        return m.group(1) if m else None

    def fetch(self, handle: str) -> RunResult:
        metrics_path = self.runs_dir / handle / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"No metrics.json for run {handle}: {metrics_path}")
        data = json.loads(metrics_path.read_text())
        return RunResult(
            run_id=data.get("run_id", handle),
            metrics={k: float(v) for k, v in (data.get("metrics") or {}).items()},
            series=data.get("series") or {},
            extra=data.get("extra") or {},
        )


# ---------------------------------------------------------------------------
# ClearML backend (future — stub only, not wired to a live server here)
# ---------------------------------------------------------------------------


class ClearMLTracker:
    """Fetches metrics from ClearML. Kept import-clean until actually selected.

    Not exercised by the local test harness.  When you're ready to integrate:
      1. ``uv sync --extra clearml``
      2. have ``model_dir/run.py`` create a ``clearml.Task`` and print
         ``CLEARML_TASK_ID=<task.id>``
      3. set ``TRACKER=clearml`` in the environment.
    The orchestrator needs no changes — it only talks to the Tracker protocol.
    """

    handle_key = "clearml"

    def __init__(self, target_metric: str = "val/auprc", direction: str = "max"):
        self.target_metric = target_metric
        self.direction = direction

    def parse_handle(self, stdout: str) -> Optional[str]:
        m = _HANDLE_PATTERNS["clearml"].search(stdout or "")
        return m.group(1) if m else None

    def fetch(self, handle: str) -> RunResult:
        # Lazy import: clearml is an optional dependency.
        from clearml import Task  # noqa: F401  (import-time cost only when used)

        task = Task.get_task(task_id=handle)
        raw = task.get_reported_scalars() or {}
        series: Dict[str, list] = {}
        metrics: Dict[str, float] = {}
        for metric, variants in raw.items():
            if metric == "summary":
                continue
            for variant, payload in variants.items():
                key = f"{metric}/{variant}" if variant != metric else metric
                ys = list(payload.get("y", []))
                xs = list(payload.get("x", range(len(ys))))
                if not ys:
                    continue
                series[key] = list(zip(xs, ys))
                metrics[key] = max(ys) if self.direction == "max" else min(ys)
        return RunResult(run_id=handle, metrics=metrics, series=series)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_tracker(name: str, *, runs_dir: Path, target_metric: str, direction: str) -> Tracker:
    """Instantiate a tracker by name (``"local"`` | ``"clearml"``)."""
    name = (name or "local").strip().lower()
    if name == "local":
        return LocalTracker(runs_dir=runs_dir)
    if name == "clearml":
        return ClearMLTracker(target_metric=target_metric, direction=direction)
    raise ValueError(f"Unknown tracker backend: {name!r} (use 'local' or 'clearml')")
