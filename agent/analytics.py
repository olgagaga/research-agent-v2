"""
Read agent run data into plain dicts — the single source of truth for the
dashboard (and any other consumer). No HTML, no framework: just functions that
turn the durable archive (``history/``), parallel runs (``parallel_runs/``), and
the ephemeral ``logs.md`` into JSON-serialisable structures.

Solo run  -> :func:`collect_solo`
Population -> :func:`collect_parallel`, :func:`list_parallel_runs`
Sessions  -> :func:`list_sessions`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agent import config
from agent.logs_manager import read_logs

_SUCCESS = {"statistically better", "better"}


# --------------------------------------------------------------------------- io


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, tolerating a torn final line."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _tok_str(rec: dict) -> str:
    ti, to, tc = rec.get("tokens_in", 0), rec.get("tokens_out", 0), rec.get("tokens_cached", 0)
    if not (ti or to):
        return "—"
    return f"{ti}→{to}" + (f" ({tc} cached)" if tc else "")


# --------------------------------------------------------------------------- sessions


def _archive_sessions(archive_dir: Optional[Path] = None) -> dict[str, list[dict]]:
    archive = (archive_dir or config.ARCHIVE_DIR) / "experiments.jsonl"
    grouped: dict[str, list[dict]] = {}
    for rec in _read_jsonl(archive):
        grouped.setdefault(rec.get("session", "?"), []).append(rec)
    return grouped


def list_sessions(archive_dir: Optional[Path] = None) -> list[dict]:
    """Summaries of every session in the archive, newest first."""
    grouped = _archive_sessions(archive_dir)
    out = []
    for sid, recs in grouped.items():
        scored = [r for r in recs if r.get("score") is not None]
        best = max((r["score"] for r in scored), default=None)
        out.append({
            "session": sid,
            "agent": recs[0].get("agent", "solo") if recs else "solo",
            "model": recs[0].get("model") if recs else None,
            "experiments": len(scored),
            "attempts": len(recs),
            "best": best,
            "cost": round(sum(float(r.get("cost") or 0) for r in recs), 6),
            "started": recs[0].get("ts") if recs else None,
        })
    return sorted(out, key=lambda s: s["session"], reverse=True)


# --------------------------------------------------------------------------- solo


def collect_solo(session: Optional[str] = None, all_sessions: bool = False,
                 archive_dir: Optional[Path] = None) -> dict:
    """Solo-run data. Prefers the durable archive; falls back to logs.md."""
    archive = (archive_dir or config.ARCHIVE_DIR) / "experiments.jsonl"
    if archive.exists() and archive.stat().st_size > 0:
        return _from_archive(archive, session, all_sessions)
    return _from_logs()


def _from_archive(path: Path, session: Optional[str], all_sessions: bool) -> dict:
    grouped: dict[str, list[dict]] = {}
    for rec in _read_jsonl(path):
        grouped.setdefault(rec.get("session", "?"), []).append(rec)
    session_ids = sorted(grouped)

    if all_sessions:
        records = [r for sid in session_ids for r in grouped[sid]]
        chosen = f"all ({len(session_ids)} sessions)"
    else:
        sid = session if session in grouped else (session_ids[-1] if session_ids else None)
        records = grouped.get(sid, [])
        chosen = sid or "—"

    experiments, best, baseline = [], None, None
    best_series, best_run_id, best_val = [], None, -1.0
    metric = config.TARGET_METRIC
    cum = 0.0
    for i, r in enumerate(records, 1):
        score = r.get("score")
        if score is not None:
            best = score if best is None else max(best, score)
        cum += float(r.get("cost") or 0.0)
        metrics = r.get("metrics") or {}
        baseline = baseline if baseline is not None else metrics.get("pos_rate")
        va = metrics.get(metric)
        if va is not None and va > best_val:
            best_val = va
            best_series = (r.get("series") or {}).get(metric, [])
            best_run_id = r.get("run_id")
        experiments.append({
            "n": i, "iteration": r.get("iteration"),
            "target": r.get("target") or "—",
            "description": r.get("short_description") or r.get("status", ""),
            "reasoning": r.get("reasoning"),
            "status": r.get("status", ""), "kept": bool(r.get("kept")),
            "score": score, "best_so_far": best,
            "cost": r.get("cost"), "cum_cost": round(cum, 6),
            "tokens": _tok_str(r), "error": r.get("error"),
        })

    scored = [e for e in experiments if e["score"] is not None]
    kept = sum(1 for e in scored if e["kept"])
    return {
        "meta": {
            "metric": metric, "direction": config.TARGET_DIRECTION,
            "model": config.LLM_MODEL, "model_dir": config.MODEL_DIR.name,
            "task": "mammography · rare-event medical detection (~2.3% positive)",
            "source": "archive", "session": chosen, "n_sessions": len(session_ids),
        },
        "experiments": experiments,
        "best_series": best_series, "best_run_id": best_run_id, "baseline": baseline,
        "summary": {
            "best": best, "count": len(scored), "kept": kept,
            "reverted": len(scored) - kept, "attempts": len(experiments),
            "total_cost": round(cum, 6),
            "first_score": scored[0]["score"] if scored else None,
        },
    }


def _from_logs() -> dict:
    entries, _ = read_logs(config.LOGS_FILE)
    metric = config.TARGET_METRIC
    experiments, best = [], None
    for e in entries:
        if e.score is not None:
            best = e.score if best is None else max(best, e.score)
        experiments.append({
            "n": e.number, "target": e.target, "description": e.description,
            "status": e.status, "kept": e.status in _SUCCESS, "score": e.score,
            "best_so_far": best, "cost": e.cost, "cum_cost": e.cum_cost,
            "tokens": e.tokens, "reasoning": None, "error": None,
        })
    scored = [e for e in experiments if e["score"] is not None]
    kept = sum(1 for e in experiments if e["kept"])
    total = next((e["cum_cost"] for e in reversed(experiments) if e["cum_cost"] is not None), 0.0)
    return {
        "meta": {"metric": metric, "direction": config.TARGET_DIRECTION,
                 "model": config.LLM_MODEL, "model_dir": config.MODEL_DIR.name,
                 "task": "mammography · rare-event medical detection (~2.3% positive)",
                 "source": "logs.md", "session": "—", "n_sessions": 0},
        "experiments": experiments, "best_series": [], "best_run_id": None, "baseline": None,
        "summary": {"best": best, "count": len(experiments), "kept": kept,
                    "reverted": len(experiments) - kept, "attempts": len(experiments),
                    "total_cost": total,
                    "first_score": scored[0]["score"] if scored else None},
    }


# --------------------------------------------------------------------------- parallel

_HUES = ["#4c9be8", "#e8804c", "#3ec98a", "#c264d6", "#e0b13a",
         "#e8607a", "#5ec5c0", "#9a86e0", "#7bbf3a", "#d98cc0"]


def parallel_runs_dir() -> Path:
    return config.MODEL_DIR.parent / "parallel_runs"


def list_parallel_runs() -> list[dict]:
    root = parallel_runs_dir()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        lb = d / "leaderboard.json"
        info = {"name": d.name, "n_agents": None, "best": None, "metric": None}
        if lb.exists():
            try:
                data = json.loads(lb.read_text())
                info.update(n_agents=data.get("n_agents"),
                            best=(data.get("best") or {}).get("best") if data.get("best") else None,
                            metric=data.get("metric"),
                            total_cost=data.get("total_cost"))
            except Exception:
                pass
        out.append(info)
    return out


def collect_parallel(name: str) -> dict:
    run_dir = parallel_runs_dir() / name
    lb_path = run_dir / "leaderboard.json"
    leaderboard = json.loads(lb_path.read_text()) if lb_path.exists() else {}
    direction = leaderboard.get("direction", config.TARGET_DIRECTION)
    metric = leaderboard.get("metric", config.TARGET_METRIC)

    archive_root = run_dir / "archives"
    by_agent: dict[str, list] = {}
    if archive_root.exists():
        for adir in sorted(archive_root.glob("agent_*")):
            by_agent[adir.name] = _read_jsonl(adir / "experiments.jsonl")

    lb = {b["agent"]: b for b in leaderboard.get("leaderboard", [])}
    agents, best_overall = [], None
    for i, (aname, recs) in enumerate(sorted(by_agent.items())):
        scored = [r for r in recs if r.get("score") is not None]
        pts, best = [], None
        for k, r in enumerate(scored, 1):
            s = r["score"]
            best = s if best is None else (max(best, s) if direction == "max" else min(best, s))
            pts.append([k, best])
        cost = sum(float(r.get("cost") or 0) for r in recs)
        agents.append({
            "label": aname, "color": _HUES[i % len(_HUES)],
            "points": pts, "best": best, "done": len(scored),
            "attempts": len(recs), "kept": sum(1 for r in scored if r.get("kept")),
            "cost": round(cost, 6),
            "effort": lb.get(aname, {}).get("effort"),
            "hint": lb.get(aname, {}).get("hint", ""),
            "experiments": [{
                "n": k + 1, "target": r.get("target"), "status": r.get("status"),
                "score": r.get("score"), "kept": bool(r.get("kept")),
                "description": r.get("short_description"),
            } for k, r in enumerate(recs)],
        })
        if best is not None and (best_overall is None or
                                 (best > best_overall if direction == "max" else best < best_overall)):
            best_overall = best

    maxlen = max((len(a["points"]) for a in agents), default=0)
    pop_curve, running = [], None
    for k in range(1, maxlen + 1):
        vals = [a["points"][k - 1][1] for a in agents if len(a["points"]) >= k]
        if not vals:
            continue
        cand = max(vals) if direction == "max" else min(vals)
        running = cand if running is None else (max(running, cand) if direction == "max" else min(running, cand))
        pop_curve.append([k, running])

    return {
        "meta": {"metric": metric, "direction": direction, "run": name,
                 "n_agents": len(agents),
                 "total_cost": round(sum(a["cost"] for a in agents), 6),
                 "best": best_overall},
        "agents": agents, "population": pop_curve,
    }
