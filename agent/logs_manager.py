"""
State management via ``logs.md`` (Section 4 of TASK.md).

``logs.md`` is the agent's durable, human-readable memory.  Each entry records
one resolved experiment with the exact status string required by the spec:

* ``statistically better`` — score improved ≥ 3%
* ``better`` — score improved < 3%
* ``lower`` — score decreased < 3%
* ``statistically lower`` — score decreased ≥ 3%
* ``crushed`` — run failed/crashed twice in a row
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional

log = logging.getLogger(__name__)

VALID_STATUSES = frozenset({
    "statistically better",
    "better",
    "lower",
    "statistically lower",
    "crushed",
})

# Statuses that count as a successful (score-improving) run when computing best.
_NON_SCORING = {"crushed"}


class LogEntry(NamedTuple):
    number: int
    timestamp: str
    target: str
    description: str
    reasoning_summary: str
    score: Optional[float]
    best_score: Optional[float]
    status: str
    # Cost accounting (trailing columns; older rows may omit them).
    tokens: str = "—"       # "in→out" LLM tokens for this experiment
    cost: Optional[float] = None       # $ spent on this experiment
    cum_cost: Optional[float] = None   # cumulative $ so far


_HEADER = (
    "| # | Timestamp | Target | Description | Reasoning | Score | Best | Status "
    "| Tokens(in→out) | Cost $ | Cum $ |\n"
    "|---|-----------|--------|-------------|-----------|-------|------|--------"
    "|----------------|--------|-------|\n"
)

_SEP = re.compile(r"\s*\|\s*")


def _float(s: str) -> Optional[float]:
    s = s.strip()
    if s in ("", "-", "—", "N/A", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_table_line(line: str) -> Optional[LogEntry]:
    line = line.strip()
    if not line.startswith("|") or not line.endswith("|"):
        return None
    parts = _SEP.split(line.strip("|"))
    if len(parts) < 8:
        return None
    try:
        num = int(parts[0].strip())
    except ValueError:
        return None
    return LogEntry(
        number=num,
        timestamp=parts[1].strip(),
        target=parts[2].strip(),
        description=parts[3].strip(),
        reasoning_summary=parts[4].strip(),
        score=_float(parts[5]),
        best_score=_float(parts[6]),
        status=parts[7].strip(),
        tokens=parts[8].strip() if len(parts) > 8 else "—",
        cost=_float(parts[9]) if len(parts) > 9 else None,
        cum_cost=_float(parts[10]) if len(parts) > 10 else None,
    )


def read_logs(logs_path: Path) -> tuple[list[LogEntry], Optional[float]]:
    """Parse ``logs.md`` → ``(entries, best_score)``.

    *best_score* is the best score among non-crashed entries.  Direction is not
    known here, so we assume higher-is-better for the running max; the
    orchestrator owns the authoritative best via its own comparison.  (For
    lower-is-better metrics, read_logs is only used to seed context display.)
    """
    entries: list[LogEntry] = []
    best: Optional[float] = None
    if not logs_path.exists():
        return entries, best
    with open(logs_path, "r") as fh:
        for line in fh:
            entry = _parse_table_line(line)
            if entry is None:
                continue
            entries.append(entry)
            if entry.score is not None and entry.status not in _NON_SCORING:
                if best is None or entry.score > best:
                    best = entry.score
    return entries, best


def append_log_entry(
    logs_path: Path,
    description: str,
    reasoning_summary: str,
    status: str,
    score: Optional[float] = None,
    best_score: Optional[float] = None,
    target: str = "—",
    tokens: str = "—",
    cost: Optional[float] = None,
    cum_cost: Optional[float] = None,
) -> LogEntry:
    """Append one experiment entry to ``logs.md``."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status {status!r}. Must be one of: {', '.join(sorted(VALID_STATUSES))}"
        )
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    entries, _ = read_logs(logs_path)
    next_number = max((e.number for e in entries), default=0) + 1
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Escape pipes so descriptions don't break the markdown table.
    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    score_str = f"{score:.6f}" if score is not None else "—"
    best_str = f"{best_score:.6f}" if best_score is not None else "—"
    cost_str = f"{cost:.4f}" if cost is not None else "—"
    cum_str = f"{cum_cost:.4f}" if cum_cost is not None else "—"

    row = (
        f"| {next_number} | {timestamp} | {esc(target)} | {esc(description)} "
        f"| {esc(reasoning_summary)} | {score_str} | {best_str} | {status} "
        f"| {tokens} | {cost_str} | {cum_str} |\n"
    )

    if next_number == 1:
        with open(logs_path, "w") as fh:
            fh.write("# Experiment Log\n\n")
            fh.write(_HEADER)
            fh.write(row)
    else:
        with open(logs_path, "a") as fh:
            fh.write(row)

    entry = LogEntry(
        number=next_number, timestamp=timestamp, target=target,
        description=description, reasoning_summary=reasoning_summary,
        score=score, best_score=best_score, status=status,
        tokens=tokens, cost=cost, cum_cost=cum_cost,
    )
    log.info("Logged experiment #%d [%s] (%s)", next_number, target, status)
    return entry
