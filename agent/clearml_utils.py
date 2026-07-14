"""
ClearML data retrieval helpers (Section 3 of the spec).

Test IDs (from the spec):
  * ``c86c0fe7af78403da21dc13ec4eff489`` — successful run
  * ``186b9a48a0d4431e841e8b0c6d09743c`` — crashed run
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

log = logging.getLogger(__name__)

# Type aliases for readability
Scalars = Dict[str, list[Tuple[int, float]]]  # metric_name -> [(step, value), ...]
Plots = Dict[str, Any]  # plot_name -> plot_data


def fetch_clearml_data(task_id: str) -> Tuple[Scalars, Plots]:
    """Retrieve scalars (excluding ``summary``) and plots for a ClearML task.

    Args:
        task_id: ClearML task UUID.

    Returns:
        ``(filtered_scalars, plots)`` tuple.
    """
    from clearml import Task

    task = Task.get_task(task_id=task_id)

    # 1. Retrieve and filter scalars -------------------------------------------
    raw_scalars: Dict[str, Any] = task.get_reported_scalars()
    filtered_scalars: Scalars = {
        metric: list(values) if not isinstance(values, list) else values
        for metric, values in raw_scalars.items()
        if metric != "summary"
    }

    # 2. Retrieve additional metrics / tables ----------------------------------
    plots: Plots = task.get_reported_plots()

    log.info(
        "Fetched ClearML task %s: %d scalar series, %d plots",
        task_id,
        len(filtered_scalars),
        len(plots),
    )
    return filtered_scalars, plots


def extract_best_score(
    scalars: Scalars,
    metric: str = "validation/mIoU",
    direction: str = "max",
) -> float | None:
    """Extract the best (max / min) value for a given metric from ClearML scalars.

    Args:
        scalars: Filtered scalars dict as returned by :func:`fetch_clearml_data`.
        metric: Metric name to look for.
        direction: ``"max"`` (higher is better) or ``"min"`` (lower is better).

    Returns:
        The best value or ``None`` if the metric wasn't found.
    """
    if metric not in scalars:
        log.warning("Metric %r not found in scalars. Available: %s", metric, list(scalars.keys()))
        return None

    values = [v for _, v in scalars[metric]]
    if not values:
        return None

    if direction == "min":
        return min(values)
    return max(values)
