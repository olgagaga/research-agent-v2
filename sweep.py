#!/usr/bin/env python3
"""
Run a GRID of system variants × repeat trials — the apparatus for studying the
agent itself (RESEARCH.md §6).

`parallel.py` answers "run N agents at one config, keep the best" (a *search*
strategy). This answers a different question: **"does system variant A beat
variant B?"** — which needs (a) the config recorded, (b) repeated trials so an
outcome is a distribution rather than an anecdote, and (c) equal budget per arm.

Each (variant × trial) is one isolated agent: its own git worktree, its own
process, its own archive dir, tagged with `VARIANT` + `TRIAL` in every record.

Usage
-----
    # does reasoning effort matter? 3 trials per arm, 6 experiments each
    python sweep.py --vary REASONING_EFFORT=low,medium,high --trials 3 -n 6

    # LLM vs the non-LLM control arm (once agent/proposers.py lands)
    python sweep.py --vary PROPOSER=llm,random --trials 5 -n 8

    # two dimensions (cartesian product)
    python sweep.py --vary REASONING_EFFORT=low,high --vary MAIN_MODEL=openai/gpt-5-mini,openai/gpt-5 --trials 3

Results: sweep_runs/<name>/summary.json  (per-variant mean ± σ, cost, n)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent import config  # noqa: E402
# Reuse the isolation + polling machinery rather than duplicating it.
from parallel import agent_stats, load_records, _now  # noqa: E402


def parse_vary(specs: list[str]) -> dict[str, list[str]]:
    """``["EFFORT=low,high", "MODEL=a,b"]`` → ``{"EFFORT": ["low","high"], ...}``"""
    grid: dict[str, list[str]] = {}
    for s in specs:
        if "=" not in s:
            sys.exit(f"--vary needs KEY=v1,v2 (got {s!r})")
        k, vs = s.split("=", 1)
        grid[k.strip()] = [v.strip() for v in vs.split(",") if v.strip()]
    return grid


def variant_name(combo: dict[str, str]) -> str:
    """Stable, filesystem-safe label for one point in the grid."""
    if not combo:
        return "default"
    return "__".join(f"{k.lower()}-{v.replace('/', '-')}" for k, v in sorted(combo.items()))


def make_arms(grid: dict[str, list[str]], trials: int, offset: int = 0) -> list[dict]:
    keys = sorted(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))] or [{}]
    arms = []
    for combo in combos:
        for t in range(offset, offset + trials):
            arms.append({"combo": combo, "variant": variant_name(combo), "trial": t})
    return arms


def parse_arms(specs: list[str], trials: int, offset: int = 0) -> list[dict]:
    """Explicit arms: ``NAME:KEY=V,KEY2=V2``.

    Needed whenever the grid isn't a clean product — e.g. comparing the LLM
    against two *random* menus: a `--vary PROPOSER=llm,random --vary
    RANDOM_MENU=curated,wide` product would run the LLM arm twice (it ignores
    RANDOM_MENU), paying for a duplicate.
    """
    arms = []
    for spec in specs:
        if ":" not in spec:
            sys.exit(f"--arm needs NAME:KEY=V[,KEY=V] (got {spec!r})")
        name, kvs = spec.split(":", 1)
        combo = {}
        for kv in kvs.split(","):
            if "=" not in kv:
                sys.exit(f"--arm bad assignment {kv!r} in {spec!r}")
            k, v = kv.split("=", 1)
            combo[k.strip()] = v.strip()
        for t in range(offset, offset + trials):
            arms.append({"combo": combo, "variant": name.strip(), "trial": t})
    return arms


def launch_arm(arm: dict, i: int, run_dir: Path, model_dir: Path, base: str,
               iterations: int, tag: str, threads: int) -> dict:
    wt = run_dir / "work" / f"{arm['variant']}__t{arm['trial']}"
    archive = run_dir / "archives" / f"{arm['variant']}__t{arm['trial']}"
    archive.mkdir(parents=True, exist_ok=True)
    logf = run_dir / "logs" / f"{arm['variant']}__t{arm['trial']}.out"
    logf.parent.mkdir(parents=True, exist_ok=True)
    wt.parent.mkdir(parents=True, exist_ok=True)

    branch = f"sweep/{tag}/{arm['variant']}-t{arm['trial']}"
    r = subprocess.run(["git", "-C", str(model_dir), "worktree", "add", "-b", branch, str(wt), base],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"worktree add failed: {r.stderr.strip()}")

    env = dict(os.environ)
    env.update(
        MODEL_DIR=str(wt), AUTORESEARCH_HISTORY=str(archive),
        VARIANT=arm["variant"], TRIAL=str(arm["trial"]),
        AGENT_LABEL=f"{arm['variant']}__t{arm['trial']}",
        AGENT_SEED=str(arm["trial"]),          # trial index = the harness seed
        EXEC_COMMAND="python3 run.py", PROMPT_CACHE="1",
        OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
    )
    env.update(arm["combo"])                    # the independent variable(s)

    fh = open(logf, "w")
    proc = subprocess.Popen([sys.executable, "main.py", "-n", str(iterations)],
                            cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    return {**arm, "wt": wt, "archive": archive, "proc": proc, "fh": fh}


def monitor(arms: list[dict], direction: str, poll: float = 5.0) -> None:
    while True:
        alive = 0
        rows = []
        for a in arms:
            st_ = agent_stats(load_records(a["archive"]), direction)
            running = a["proc"].poll() is None
            alive += running
            rows.append((a, running, st_))
        print("\033[2J\033[H", end="")
        print(f"=== Sweep · {config.TARGET_METRIC} ({direction}) · {_now()} ===")
        print(f"{'variant':38} {'trial':>5} {'state':>5} {'best':>8} {'done':>5} {'cost$':>8}")
        for a, running, s in rows:
            b = f"{s['best']:.4f}" if s["best"] is not None else "—"
            print(f"{a['variant'][:38]:38} {a['trial']:>5} {'RUN' if running else 'done':>5} "
                  f"{b:>8} {s['done']:>5} {s['cost']:>8.4f}")
        print(f"\n{alive} running")
        if alive == 0:
            break
        time.sleep(poll)


def aggregate(arms: list[dict], run_dir: Path, direction: str) -> dict:
    by_variant: dict[str, list[dict]] = {}
    merged = run_dir / "all_experiments.jsonl"
    with open(merged, "w") as out:
        for a in arms:
            recs = load_records(a["archive"])
            for r in recs:
                out.write(json.dumps(r) + "\n")
            s = agent_stats(recs, direction)
            by_variant.setdefault(a["variant"], []).append(
                {"trial": a["trial"], **s, "exit_code": a["proc"].returncode})
    for a in arms:
        if not a["fh"].closed:
            a["fh"].close()

    summary = []
    for variant, trials in sorted(by_variant.items()):
        bests = [t["best"] for t in trials if t["best"] is not None]
        costs = [t["cost"] for t in trials]
        summary.append({
            "variant": variant, "n_trials": len(trials),
            "best_mean": st.mean(bests) if bests else None,
            # population σ over trials — the point of repeats (RESEARCH.md §6.2)
            "best_std": st.pstdev(bests) if len(bests) > 1 else 0.0,
            "best_min": min(bests) if bests else None,
            "best_max": max(bests) if bests else None,
            "cost_mean": st.mean(costs) if costs else 0.0,
            "cost_total": sum(costs),
            "trials": trials,
        })
    summary.sort(key=lambda v: (v["best_mean"] is not None, v["best_mean"] or 0),
                 reverse=(direction == "max"))
    out = {"metric": config.TARGET_METRIC, "direction": direction,
           "generated": _now(), "variants": summary,
           "total_cost": sum(v["cost_total"] for v in summary)}
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2))
    return out


def teardown(model_dir: Path, arms: list[dict], keep: bool) -> None:
    if keep:
        return
    for a in arms:
        subprocess.run(["git", "-C", str(model_dir), "worktree", "remove", "--force", str(a["wt"])],
                       capture_output=True, text=True)
    subprocess.run(["git", "-C", str(model_dir), "worktree", "prune"], capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep system variants × trials")
    ap.add_argument("--vary", action="append", default=[],
                    help="KEY=v1,v2 (repeatable → cartesian product). KEY is any agent env knob.")
    ap.add_argument("--arm", action="append", default=[],
                    help="explicit arm NAME:KEY=V[,KEY=V] (repeatable). Use instead of --vary "
                         "when the grid isn't a clean product.")
    ap.add_argument("--trials", "-t", type=int, default=3, help="repeats per variant")
    ap.add_argument("--trial-offset", type=int, default=0,
                    help="start trial index here (AGENT_SEED=trial, so an offset gives "
                         "FRESH seeds — required to pool with an earlier sweep)")
    ap.add_argument("--iterations", "-n", type=int, default=6, help="experiments per run (equal budget)")
    ap.add_argument("--base", default=None, help="base commit (default: model_dir HEAD)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-parallel", type=int, default=4, help="concurrent agents")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--keep-worktrees", action="store_true")
    args = ap.parse_args()

    model_dir = config.MODEL_DIR
    if not (model_dir / ".git").exists():
        sys.exit(f"model_dir is not a git repo: {model_dir}")
    base = args.base or subprocess.run(["git", "-C", str(model_dir), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip()
    tag = args.name or _now()
    run_dir = ROOT / "sweep_runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.arm:
        grid = {"__arms__": [a.split(":", 1)[0] for a in args.arm]}
        arms = parse_arms(args.arm, args.trials, args.trial_offset)
    else:
        grid = parse_vary(args.vary)
        arms = make_arms(grid, args.trials, args.trial_offset)
    cores = os.cpu_count() or 4
    threads = args.threads or max(1, (cores - 2) // max(1, min(args.max_parallel, len(arms))))

    print(f"Sweep '{tag}': {len(arms)} runs "
          f"({len(arms)//max(args.trials,1)} variants × {args.trials} trials × {args.iterations} iters)")
    print(f"  grid: {grid or '{} (single default arm)'}")
    print(f"  base={base[:10]}  max_parallel={args.max_parallel}  threads/agent={threads}")
    (run_dir / "spec.json").write_text(json.dumps(
        {"grid": grid, "trials": args.trials, "iterations": args.iterations,
         "base": base, "created": _now()}, indent=2))

    launched: list[dict] = []
    try:
        pending = list(enumerate(arms))
        while pending or any(a["proc"].poll() is None for a in launched):
            while pending and sum(1 for a in launched if a["proc"].poll() is None) < args.max_parallel:
                i, arm = pending.pop(0)
                launched.append(launch_arm(arm, i, run_dir, model_dir, base,
                                           args.iterations, tag, threads))
            if pending:
                time.sleep(3)
            else:
                monitor(launched, config.TARGET_DIRECTION)
    except KeyboardInterrupt:
        print("\nInterrupted — terminating…")
        for a in launched:
            a["proc"].terminate()

    summary = aggregate(launched, run_dir, config.TARGET_DIRECTION)
    teardown(model_dir, launched, args.keep_worktrees)

    print("\n=== SWEEP SUMMARY (mean ± σ over trials) ===")
    for v in summary["variants"]:
        mean = f"{v['best_mean']:.4f}" if v["best_mean"] is not None else "—"
        print(f"  {v['variant'][:44]:44} {mean} ± {v['best_std']:.4f}  "
              f"(n={v['n_trials']}, ${v['cost_total']:.4f})")
    print(f"\ntotal spend: ${summary['total_cost']:.4f}")
    print(f"results: {run_dir}/summary.json")


if __name__ == "__main__":
    main()
