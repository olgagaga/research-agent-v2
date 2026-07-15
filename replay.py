#!/usr/bin/env python3
"""
Replay an agent trajectory under a pinned harness and fresh seeds.

This is a measurement instrument, not an agent: it makes **no LLM calls** and
costs nothing but CPU. It exists because two numbers that every comparison in
RESEARCH.md silently depends on have never been measured:

1. **The noise floor.** Re-run one pipeline under R seeds and you get the
   run-to-run sigma of the target metric. The loop commits an experiment when
   ``improve > 0`` (``_classify`` in agent/orchestrator.py — ``STATISTICAL_DELTA``
   only *labels* the result, it does not gate the commit). So for an edit that
   truly changes nothing, noise alone gets it committed ~50% of the time, and the
   incumbent only ever ratchets *up* on val. Sigma tells you how big those free
   upward steps are, and therefore how wide the error bars on any A-vs-B claim
   must be.

2. **Self-report inflation.** The agent optimises ``val/auprc`` and never sees
   ``test/auprc`` (only TARGET_METRIC reaches its context, via the orchestrator's
   feedback string and logs.md). Replaying every commit gives val and test side by
   side at each step, so the val→test gap can be tracked *as the agent climbs* —
   which separates a constant split offset (already present at the baseline) from
   progressive overfitting of the val split (grows with iterations).

Method
------
One trajectory step = one commit in the model repo. For each (commit, seed):

* check out **only the agent-editable seams** from that commit
  (``config.TARGET_GROUPS`` — model.py, loss.py, optimizer.py, transforms.py,
  config.yaml);
* keep the **harness pinned** to the current working tree (run.py, data.py), so
  every step is scored by byte-identical evaluation code rather than by whatever
  harness happened to exist at that commit;
* run training with ``SEED=<seed>`` (run.py lets the env override config.yaml,
  which matters because ``seed`` lives in an agent-editable seam).

All work happens in throwaway git worktrees — one per parallel job — so the
source model_dir is never modified.

Task-agnostic by construction: it knows only "a git repo + editable seams + a
run command that prints RUN_ID and writes runs/<id>/metrics.json". Pointing it at
a second task requires no changes here.

Usage
-----
    python replay.py                              # whole trajectory, 5 seeds
    python replay.py --seeds 10 --jobs 8
    python replay.py --range 8e1f1e5..HEAD --seeds 5
    python replay.py --commits HEAD --seeds 20    # noise floor of one pipeline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from agent import config  # noqa: E402

# Files that define *evaluation*, pinned to the working tree so that every commit
# in the trajectory is judged by identical code. Everything else in the model dir
# is either an agent-editable seam or irrelevant to scoring.
HARNESS_FILES = ("run.py", "data.py")

_RUN_ID_RE = re.compile(r"^RUN_ID=(\S+)", re.MULTILINE)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)


def _git_ok(args: list[str], cwd: Path) -> str:
    r = _git(args, cwd)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


# --------------------------------------------------------------------------- setup


def resolve_commits(model_dir: Path, rng: str | None, explicit: list[str] | None) -> list[dict]:
    """Trajectory steps, oldest first, as [{sha, subject}]."""
    if explicit:
        shas = [_git_ok(["rev-parse", c], model_dir) for c in explicit]
    else:
        out = _git_ok(["rev-list", "--reverse", rng or "HEAD"], model_dir)
        shas = [s for s in out.splitlines() if s]
    steps = []
    for sha in shas:
        subject = _git_ok(["log", "-1", "--format=%s", sha], model_dir)
        steps.append({"sha": sha, "short": sha[:10], "subject": subject})
    return steps


def harness_fingerprint(model_dir: Path) -> dict:
    """Hash the pinned harness so a replay's provenance is self-describing."""
    h = hashlib.sha256()
    present = []
    for name in HARNESS_FILES:
        p = model_dir / name
        if p.exists():
            h.update(p.read_bytes())
            present.append(name)
    return {"files": present, "sha256": h.hexdigest()[:16]}


class WorktreePool:
    """A worktree per parallel job. Threads borrow one for the duration of a task.

    Worktrees are detached at HEAD; each task then swaps in that commit's seams.
    The harness is copied from the source working tree once, at creation.
    """

    def __init__(self, model_dir: Path, root: Path, n: int):
        self.model_dir = model_dir
        self.paths: list[Path] = []
        self._q: queue.Queue[Path] = queue.Queue()
        root.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            wt = root / f"wt_{i:02d}"
            r = _git(["worktree", "add", "--detach", str(wt), "HEAD"], model_dir)
            if r.returncode != 0:
                self.close()
                raise RuntimeError(f"worktree add failed: {r.stderr.strip()}")
            for name in HARNESS_FILES:
                src = model_dir / name
                if src.exists():
                    shutil.copy2(src, wt / name)
            self.paths.append(wt)
            self._q.put(wt)

    def acquire(self) -> Path:
        return self._q.get()

    def release(self, wt: Path) -> None:
        self._q.put(wt)

    def close(self) -> None:
        for wt in self.paths:
            _git(["worktree", "remove", "--force", str(wt)], self.model_dir)
        _git(["worktree", "prune"], self.model_dir)


# --------------------------------------------------------------------------- run


def checkout_seams(wt: Path, sha: str, seams: list[str]) -> list[str]:
    """Swap in one commit's seams; leave the pinned harness alone.

    Returns the seams that commit actually had (early commits may lack some).
    """
    got = []
    for f in seams:
        if _git(["checkout", sha, "--", f], wt).returncode == 0:
            got.append(f)
    return got


def run_once(wt: Path, seed: int, threads: int, timeout: int) -> dict:
    env = dict(os.environ)
    env.update(
        SEED=str(seed),
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
    )
    r = subprocess.run(
        config.EXEC_COMMAND, cwd=str(wt), env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        return {"ok": False, "error": (r.stderr or r.stdout or "")[-1500:]}
    m = _RUN_ID_RE.search(r.stdout or "")
    if not m:
        return {"ok": False, "error": "no RUN_ID in stdout"}
    run_id = m.group(1)
    mp = wt / "runs" / run_id / "metrics.json"
    if not mp.exists():
        return {"ok": False, "error": f"no metrics.json for {run_id}"}
    data = json.loads(mp.read_text())
    return {"ok": True, "run_id": run_id, "metrics": data.get("metrics") or {}}


# --------------------------------------------------------------------------- stats


def summarize(rows: list[dict], metric: str) -> list[dict]:
    """Per-commit mean/sd of every metric, ordered by trajectory step."""
    by_step: dict[int, list[dict]] = {}
    for r in rows:
        by_step.setdefault(r["step"], []).append(r)

    out = []
    for step in sorted(by_step):
        rs = by_step[step]
        ok = [r for r in rs if r["ok"]]
        keys = sorted({k for r in ok for k in r["metrics"]})
        stats = {}
        for k in keys:
            vals = [r["metrics"][k] for r in ok if r["metrics"].get(k) is not None]
            if not vals:
                continue
            stats[k] = {
                "mean": statistics.mean(vals),
                "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
        out.append({
            "step": step, "sha": rs[0]["sha"], "short": rs[0]["short"],
            "subject": rs[0]["subject"],
            "n_ok": len(ok), "n_fail": len(rs) - len(ok),
            "metrics": stats,
        })
    return out


def _fmt(st: dict | None, key: str) -> str:
    if not st or key not in st:
        return "—"
    return f"{st[key]['mean']:.4f}±{st[key]['sd']:.4f}"


def report(summary: list[dict], metric: str, held_out: str | None) -> None:
    print(f"\n{'step':>4}  {'commit':10}  {metric+' (val)':>17}  "
          f"{(held_out or 'held-out'):>17}  {'gap':>8}  subject")
    print("-" * 110)
    for s in summary:
        m = s["metrics"]
        gap = "—"
        if held_out and metric in m and held_out in m:
            gap = f"{m[metric]['mean'] - m[held_out]['mean']:+.4f}"
        print(f"{s['step']:>4}  {s['short']:10}  {_fmt(m, metric):>17}  "
              f"{_fmt(m, held_out or ''):>17}  {gap:>8}  {s['subject'][:44]}")

    # --- headline 1: the noise floor -------------------------------------
    sds = [s["metrics"][metric]["sd"] for s in summary
           if metric in s["metrics"] and s["metrics"][metric]["n"] > 1]
    print()
    if sds:
        pooled = statistics.mean(sds)
        print(f"NOISE FLOOR   mean within-commit sd of {metric} = {pooled:.4f}  "
              f"(worst commit {max(sds):.4f})")
        print("              the loop commits on any improve > 0, so an edit that "
              "truly changes nothing")
        print("              is committed ~50% of the time, and the incumbent only "
              "ratchets up on val.")
        final = summary[-1]["metrics"].get(metric)
        if pooled > 0 and final:
            delta_abs = config.STATISTICAL_DELTA * abs(final["mean"])
            print(f"              for scale: the 'statistically better' label needs "
                  f"{config.STATISTICAL_DELTA:.0%} relative = {delta_abs:.4f} "
                  f"absolute here = {delta_abs / pooled:.1f} sd")

    # --- headline 2: does the self-report inflate as it climbs? -----------
    if not held_out:
        return

    def gap_at(s: dict) -> tuple[float, float] | None:
        """(gap, standard error of the gap) for one trajectory step."""
        m = s["metrics"]
        if metric not in m or held_out not in m:
            return None
        a, b = m[metric], m[held_out]
        if a["n"] < 2 or b["n"] < 2:
            return (a["mean"] - b["mean"], float("nan"))
        # SE of a difference of means. Treating the two as independent is
        # conservative: they share a seed (same trained net), so they covary
        # positively, which would only shrink this.
        se = ((a["sd"] ** 2) / a["n"] + (b["sd"] ** 2) / b["n"]) ** 0.5
        return (a["mean"] - b["mean"], se)

    pts = [(s["step"], gap_at(s)) for s in summary]
    pts = [(st, g) for st, g in pts if g is not None]
    if len(pts) < 2:
        return
    (s0, (g0, se0)), (s1, (g1, se1)) = pts[0], pts[-1]
    drift = g1 - g0
    se_drift = (se0 ** 2 + se1 ** 2) ** 0.5

    print(f"\nSELF-REPORT   {metric} − {held_out}   (how inflated is what the agent tells us?)")
    print(f"              step {s0:>2} (baseline) = {g0:+.4f} ± {se0:.4f}")
    print(f"              step {s1:>2} (final)    = {g1:+.4f} ± {se1:.4f}")
    if se_drift > 0 and drift > 2 * se_drift:
        verdict = "gap GROWS as the agent climbs → the agent overfits its own val signal"
    elif se_drift > 0:
        verdict = ("gap is flat within noise → the offset is a property of the SPLIT, "
                   "present before the agent ran")
    else:
        verdict = "need ≥2 seeds to judge"
    print(f"              drift      = {drift:+.4f} ± {se_drift:.4f}   → {verdict}")
    print(f"              NB: a baseline gap ≠ 0 means val is simply easier than "
          f"{held_out}; only the DRIFT is attributable to the agent.")

    audit_decisions(summary, metric, held_out)


def audit_decisions(summary: list[dict], metric: str, held_out: str | None) -> None:
    """Re-run the loop's own keep/revert rule with n seeds instead of 1.

    Every step here is a commit, i.e. an experiment the agent decided to KEEP.
    It made that call from a single seed, comparing against its best-so-far. We
    replay the same comparison against the replicated means, so each kept
    experiment gets a verdict: was it a real improvement, or a noise draw the
    ratchet locked in?
    """
    rows = [s for s in summary if metric in s["metrics"] and s["metrics"][metric]["n"] > 1]
    if len(rows) < 2:
        return

    print(f"\nDECISION AUDIT   the loop kept every one of these. Which were real?")
    print(f"{'step':>4}  {'Δ vs incumbent':>16}  {'verdict':<12}  subject")
    print("-" * 90)

    incumbent = None      # best replicated mean among earlier steps
    inc_stat = None
    tally = {"REAL": 0, "NOISE": 0, "REGRESSION": 0}
    for i, s in enumerate(rows):
        st = s["metrics"][metric]
        if incumbent is None:
            incumbent, inc_stat = st["mean"], st
            print(f"{s['step']:>4}  {'—  (baseline)':>16}  {'':<12}  {s['subject'][:40]}")
            continue
        delta = st["mean"] - incumbent
        se = ((st["sd"] ** 2) / st["n"] + (inc_stat["sd"] ** 2) / inc_stat["n"]) ** 0.5
        if se > 0 and delta > 2 * se:
            verdict = "REAL"
        elif se > 0 and delta < -2 * se:
            verdict = "REGRESSION"
        else:
            verdict = "NOISE"
        tally[verdict] += 1
        print(f"{s['step']:>4}  {delta:>+9.4f}±{se:.4f}  {verdict:<12}  {s['subject'][:40]}")
        if st["mean"] > incumbent:
            incumbent, inc_stat = st["mean"], st

    n = sum(tally.values())
    print(f"\n              of {n} kept experiments: {tally['REAL']} real, "
          f"{tally['NOISE']} indistinguishable from noise, "
          f"{tally['REGRESSION']} outright regressions — all committed.")

    # Where did the gain actually stop? Compare the peak to the final answer, on
    # the held-out metric the agent never saw.
    for key, label in ((metric, "self-reported"), (held_out, "held-out")):
        if not key or any(key not in s["metrics"] for s in rows):
            continue
        means = [(s["step"], s["metrics"][key]["mean"]) for s in rows]
        peak_step, peak = max(means, key=lambda t: t[1])
        fin_step, fin = means[-1]
        tail = "" if peak_step == fin_step else (
            f"  → the agent walked away from its best pipeline "
            f"({peak - fin:+.4f} worse at the end)")
        print(f"              {label:<14} peak {peak:.4f} @ step {peak_step}, "
              f"final {fin:.4f} @ step {fin_step}{tail}")


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Replay an agent trajectory under a pinned harness and fresh seeds (no LLM).")
    ap.add_argument("--seeds", "-s", type=int, default=5, help="replicates per commit")
    ap.add_argument("--range", "-r", default=None, help="git range (default: all of HEAD)")
    ap.add_argument("--commits", nargs="*", default=None, help="explicit commits instead of a range")
    ap.add_argument("--jobs", "-j", type=int, default=0, help="parallel workers (0=auto)")
    ap.add_argument("--out", default=None, help="output dir (default: replays/<timestamp>)")
    ap.add_argument("--metric", default=config.TARGET_METRIC, help="the agent's optimisation target")
    ap.add_argument("--held-out", default=None,
                    help="metric the agent never sees (default: metric with val→test swapped)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads per job (0=auto)")
    args = ap.parse_args()

    model_dir = config.MODEL_DIR
    if not (model_dir / ".git").exists():
        sys.exit(f"model_dir is not a git repo: {model_dir}")

    held_out = args.held_out or (args.metric.replace("val/", "test/")
                                 if args.metric.startswith("val/") else None)
    seams = sorted(config.ALLOWED_FILES)
    steps = resolve_commits(model_dir, args.range, args.commits)
    if not steps:
        sys.exit("no commits to replay")

    cores = os.cpu_count() or 4
    jobs = args.jobs or max(1, min(len(steps) * args.seeds, (cores - 2) // 2))
    threads = args.threads or max(1, (cores - 2) // jobs)

    tag = _now()
    out_dir = Path(args.out) if args.out else ROOT / "replays" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fp = harness_fingerprint(model_dir)
    total = len(steps) * args.seeds
    print(f"Replay '{tag}': {len(steps)} commits × {args.seeds} seeds = {total} runs")
    print(f"  metric={args.metric}  held-out={held_out or '—'}")
    print(f"  harness pinned to working tree: {fp['files']} sha256={fp['sha256']}")
    print(f"  seams swapped per commit: {seams}")
    print(f"  jobs={jobs} threads/job={threads}   out={out_dir}")

    tasks = [(i, st, seed) for i, st in enumerate(steps) for seed in range(args.seeds)]
    rows: list[dict] = []
    lock = threading.Lock()
    done = 0

    pool = WorktreePool(model_dir, out_dir / "worktrees", jobs)
    jsonl = open(out_dir / "runs.jsonl", "w")

    def work(task) -> None:
        nonlocal done
        step, st, seed = task
        wt = pool.acquire()
        try:
            got = checkout_seams(wt, st["sha"], seams)
            try:
                res = run_once(wt, seed, threads, config.RUN_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                res = {"ok": False, "error": "timeout"}
            row = {
                "step": step, "sha": st["sha"], "short": st["short"],
                "subject": st["subject"], "seed": seed, "seams": got,
                "ok": res["ok"], "metrics": res.get("metrics", {}),
                "run_id": res.get("run_id"), "error": res.get("error"),
            }
        finally:
            pool.release(wt)
        with lock:
            rows.append(row)
            jsonl.write(json.dumps(row) + "\n")
            jsonl.flush()
            done += 1
            score = row["metrics"].get(args.metric)
            note = f"{args.metric}={score:.4f}" if score is not None else f"FAIL {row['error'][:40]}"
            print(f"  [{done:>3}/{total}] step {step:>2} {st['short']} seed {seed}  {note}")

    try:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(work, tasks))
    except KeyboardInterrupt:
        print("\ninterrupted — writing what we have")
    finally:
        jsonl.close()
        pool.close()
        shutil.rmtree(out_dir / "worktrees", ignore_errors=True)

    summary = summarize(rows, args.metric)
    (out_dir / "summary.json").write_text(json.dumps({
        "generated": tag, "metric": args.metric, "held_out": held_out,
        "seeds": args.seeds, "harness": fp, "seams": seams,
        "model_dir": str(model_dir), "steps": summary,
    }, indent=2))
    report(summary, args.metric, held_out)
    print(f"\nrows: {out_dir/'runs.jsonl'}   summary: {out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
