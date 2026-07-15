#!/usr/bin/env python3
"""
Run a POPULATION of agents on the same task, each in its own git worktree.

Turns the single greedy hill-climb into a population search (RESEARCH.md §3):
N agents explore independently from the same baseline, and we keep the best.

Isolation
---------
* Each agent gets a **git worktree** of ``model_dir`` on its own branch — shared
  object store, separate working tree + commits, zero file/commit collisions.
* Each agent runs as its **own OS process** (``main.py``) with its own
  ``MODEL_DIR`` and archive dir (config is process-global, so process isolation
  is the clean boundary).
* Diversity comes from per-agent ``REASONING_EFFORT`` + an ``AGENT_HINT`` nudge.

Tracking
--------
* Each agent appends to its own durable archive
  (``<run>/archives/agent_XX/experiments.jsonl``).
* This orchestrator polls those archives and prints a **live leaderboard**, then
  writes ``<run>/leaderboard.json`` + merged ``all_experiments.jsonl`` and points
  you at the dashboard app (see dashboard/README.md).

Usage
-----
    python parallel.py --agents 4 --iterations 6
    python parallel.py --agents 8 --iterations 10 --base <baseline_sha> --name sweep1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from agent import config  # noqa: E402

# Per-agent exploration nudges (cycled). Keep them broad — they bias, not force.
DEFAULT_HINTS = [
    "Prioritise the loss function and class-imbalance handling.",
    "Prioritise model architecture (depth, width, normalisation, regularisation).",
    "Prioritise the optimiser and learning-rate schedule.",
    "Prioritise feature engineering and preprocessing in transforms.py.",
    "Balance exploration across all levers; avoid repeating recent ideas.",
]
DEFAULT_EFFORTS = ["low", "medium", "low", "medium", "high"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_records(archive: Path) -> list[dict]:
    """Read an agent's experiments.jsonl, tolerating a torn final line."""
    f = archive / "experiments.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue  # skip a partially-written trailing line
    return out


def agent_stats(records: list[dict], direction: str = "max") -> dict:
    scored = [r for r in records if r.get("score") is not None]
    best = None
    for r in scored:
        s = r["score"]
        best = s if best is None else (max(best, s) if direction == "max" else min(best, s))
    kept = sum(1 for r in scored if r.get("kept"))
    cost = sum(float(r.get("cost") or 0.0) for r in records)
    last = records[-1]["status"] if records else "—"
    return {"best": best, "done": len(scored), "attempts": len(records),
            "kept": kept, "cost": cost, "last": last}


# --------------------------------------------------------------------------- setup


def setup_worktrees(model_dir: Path, run_dir: Path, n: int, base: str, tag: str) -> list[Path]:
    worktrees = []
    for i in range(n):
        wt = run_dir / f"agent_{i:02d}"
        branch = f"par/{tag}/agent-{i:02d}"
        r = subprocess.run(
            ["git", "-C", str(model_dir), "worktree", "add", "-b", branch, str(wt), base],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"worktree add failed for agent {i}: {r.stderr.strip()}")
        worktrees.append(wt)
    return worktrees


def launch(worktrees: list[Path], run_dir: Path, iterations: int, model: str,
           hints: list[str], efforts: list[str], threads: int) -> list[dict]:
    agents = []
    for i, wt in enumerate(worktrees):
        archive = run_dir / "archives" / f"agent_{i:02d}"
        archive.mkdir(parents=True, exist_ok=True)
        logf = (run_dir / "logs" / f"agent_{i:02d}.out")
        logf.parent.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.update(
            MODEL_DIR=str(wt),
            AUTORESEARCH_HISTORY=str(archive),
            AGENT_LABEL=f"agent_{i:02d}",
            AGENT_HINT=hints[i % len(hints)],
            REASONING_EFFORT=efforts[i % len(efforts)],
            EXEC_COMMAND="python3 run.py",
            MAIN_MODEL=model,
            PROMPT_CACHE="1",
            # keep CPU fair across the population
            OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
        )
        fh = open(logf, "w")
        proc = subprocess.Popen(
            [sys.executable, "main.py", "-n", str(iterations)],
            cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
        )
        agents.append({"i": i, "wt": wt, "archive": archive, "proc": proc,
                       "log": logf, "fh": fh, "hint": env["AGENT_HINT"],
                       "effort": env["REASONING_EFFORT"]})
    return agents


# --------------------------------------------------------------------------- monitor


def monitor(agents: list[dict], direction: str, poll: float = 5.0) -> None:
    metric = config.TARGET_METRIC
    while True:
        rows = []
        alive = 0
        for a in agents:
            st = agent_stats(load_records(a["archive"]), direction)
            running = a["proc"].poll() is None
            alive += running
            rows.append((a["i"], running, a["effort"], st))
        # render
        print("\033[2J\033[H", end="")  # clear screen
        print(f"=== Parallel run · {metric} ({direction}) · {_now()} ===")
        print(f"{'agent':6} {'state':5} {'effort':7} {'best':>9} {'done':>5} "
              f"{'kept':>5} {'attempts':>9} {'cost$':>9}  last")
        best_overall, best_agent = None, None
        for i, running, effort, st in rows:
            b = st["best"]
            if b is not None and (best_overall is None or
                                  (b > best_overall if direction == "max" else b < best_overall)):
                best_overall, best_agent = b, i
            bstr = f"{b:.4f}" if b is not None else "—"
            print(f"a{i:<5} {'RUN' if running else 'done':5} {effort:7} {bstr:>9} "
                  f"{st['done']:>5} {st['kept']:>5} {st['attempts']:>9} {st['cost']:>9.4f}  {st['last']}")
        tot = sum(agent_stats(load_records(a['archive']), direction)['cost'] for a in agents)
        lead = f"agent_{best_agent:02d}" if best_agent is not None else "—"
        print(f"\nleader: {lead}  best {metric}={best_overall}  |  total spend ${tot:.4f}  |  {alive} running")
        if alive == 0:
            break
        time.sleep(poll)


# --------------------------------------------------------------------------- aggregate


def aggregate(agents: list[dict], run_dir: Path, direction: str) -> dict:
    merged = run_dir / "all_experiments.jsonl"
    board = []
    with open(merged, "w") as out:
        for a in agents:
            recs = load_records(a["archive"])
            for r in recs:
                out.write(json.dumps(r) + "\n")
            st = agent_stats(recs, direction)
            board.append({"agent": f"agent_{a['i']:02d}", "hint": a["hint"],
                          "effort": a["effort"], **st,
                          "exit_code": a["proc"].returncode})
    for a in agents:  # close subprocess log handles
        if not a["fh"].closed:
            a["fh"].close()
    board.sort(key=lambda x: (x["best"] is not None,
                              x["best"] if x["best"] is not None else 0),
               reverse=(direction == "max"))
    summary = {
        "metric": config.TARGET_METRIC, "direction": direction,
        "generated": _now(), "n_agents": len(agents),
        "total_cost": sum(b["cost"] for b in board),
        "best": board[0] if board else None,
        "leaderboard": board,
    }
    (run_dir / "leaderboard.json").write_text(json.dumps(summary, indent=2))
    return summary


def teardown(model_dir: Path, worktrees: list[Path], keep: bool) -> None:
    if keep:
        return
    for wt in worktrees:
        subprocess.run(["git", "-C", str(model_dir), "worktree", "remove", "--force", str(wt)],
                       capture_output=True, text=True)
    subprocess.run(["git", "-C", str(model_dir), "worktree", "prune"], capture_output=True)


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a population of agents in worktrees")
    ap.add_argument("--agents", "-a", type=int, default=4)
    ap.add_argument("--iterations", "-n", type=int, default=6)
    ap.add_argument("--base", default=None, help="base commit for all worktrees (default: model_dir HEAD)")
    ap.add_argument("--model", default=os.environ.get("MAIN_MODEL", "openai/gpt-5-mini"))
    ap.add_argument("--name", default=None, help="run name (default: timestamp)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads per agent (0=auto)")
    ap.add_argument("--keep-worktrees", action="store_true", help="don't delete worktrees after")
    args = ap.parse_args()

    model_dir = config.MODEL_DIR
    if not (model_dir / ".git").exists():
        sys.exit(f"model_dir is not a git repo: {model_dir}")

    base = args.base or subprocess.run(
        ["git", "-C", str(model_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    tag = args.name or _now()
    run_dir = ROOT / "parallel_runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    cores = os.cpu_count() or 4
    threads = args.threads or max(1, (cores - 2) // max(1, args.agents))

    print(f"Population run '{tag}': {args.agents} agents × {args.iterations} iters")
    print(f"  base={base[:10]}  model={args.model}  threads/agent={threads}")
    print(f"  run dir: {run_dir}")

    worktrees = setup_worktrees(model_dir, run_dir, args.agents, base, tag)
    agents = launch(worktrees, run_dir, args.iterations, args.model,
                    DEFAULT_HINTS, DEFAULT_EFFORTS, threads)
    try:
        monitor(agents, config.TARGET_DIRECTION)
    except KeyboardInterrupt:
        print("\nInterrupted — terminating agents…")
        for a in agents:
            a["proc"].terminate()
    summary = aggregate(agents, run_dir, config.TARGET_DIRECTION)
    teardown(model_dir, worktrees, args.keep_worktrees)

    print("\n=== FINAL LEADERBOARD ===")
    for b in summary["leaderboard"]:
        print(f"  {b['agent']}  best={b['best']}  kept={b['kept']}  ${b['cost']:.4f}  ({b['effort']}, {b['hint'][:40]})")
    print(f"\nwinner: {summary['best']['agent'] if summary['best'] else '—'}  "
          f"best {summary['metric']}={summary['best']['best'] if summary['best'] else '—'}")
    print(f"total spend: ${summary['total_cost']:.4f}")
    print(f"\nrun dir: {run_dir}")
    print("dashboard: python dashboard/backend/app.py  ->  http://localhost:8000 (Population tab)")


if __name__ == "__main__":
    main()
