"""
Main orchestration loop (Sections 5 & 7 of TASK.md).

Each iteration:

1.  Build a *fresh, bounded* context: system + wiki + current code + log memory.
2.  Ask the LLM for an :class:`ExperimentPlan` (surgical edits, one target group).
3.  Validate atomicity + that every edited file still parses (dry-run in memory).
4.  Write the edits to ``model_dir``.
5.  Run training as a subprocess (with a wall-clock cap).
6.  On non-zero exit → two-strike crash handling.
7.  On success → fetch metrics via the pluggable Tracker.
8.  Classify the score delta vs. best.
9.  Git commit (success) or hard-reset (regression) the model dir.
10. Append the outcome to ``logs.md``.
11. Update best score and loop.

Token-cost design
-----------------
* The LLM emits **surgical edits**, not whole files → fewer output tokens.
* Context is **rebuilt fresh each turn** (not accumulated) → per-call tokens are
  O(codebase + recent log) instead of O(iterations x codebase).
* The stable system prompt is marked for **prompt caching** on Anthropic.
* Durable memory lives in ``logs.md``, always summarised back into context.
"""

from __future__ import annotations

import logging
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent import config
from agent.archive import RunArchive, new_session_id
from agent.editor import EditError, apply_edits
from agent.git_manager import execute_git_protocol, reset_on_failure
from agent.llm import LLMClient
from agent.logs_manager import append_log_entry, read_logs
from agent.proposers import make_proposer
from agent.schemas import ExperimentPlan, FileEdits
from agent.tracker import RunResult, Tracker, make_tracker

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation (Section 2)
# ---------------------------------------------------------------------------


def _target_group_for(file_name: str) -> frozenset[str]:
    for group in config.TARGET_GROUPS:
        if file_name in group:
            return frozenset(group)
    return frozenset({file_name})


def validate_plan(
    model_dir: Path, plan: ExperimentPlan
) -> Tuple[Optional[str], Dict[str, str]]:
    """Validate a plan and dry-run its edits in memory.

    Returns ``(error_or_None, {filename: new_content})``.  On success the second
    element holds the fully-edited file contents ready to write; on failure it is
    empty and the string explains why (fed back to the LLM).
    """
    if not plan.edits:
        return "Plan has no edits — propose at least one file edit.", {}

    # --- atomicity: all edited files in exactly one target group -------------
    groups = {
        _target_group_for(fe.filename) for fe in plan.edits
    }
    if len(groups) != 1:
        touched = sorted({f for g in groups for f in g})
        return (
            f"Edits span multiple atomic targets ({touched}). Pick exactly ONE "
            "target group: model.py, loss.py, optimizer.py, or the "
            "transforms.py + config.yaml tandem.",
            {},
        )

    # --- only allow known mutable files --------------------------------------
    for fe in plan.edits:
        if fe.filename not in config.ALLOWED_FILES:
            return (
                f"File {fe.filename!r} is not editable. Allowed: "
                f"{sorted(config.ALLOWED_FILES)}.",
                {},
            )

    # --- dry-run each file's edits (catches syntax errors before training) ---
    new_contents: Dict[str, str] = {}
    for fe in plan.edits:
        path = model_dir / fe.filename
        original = path.read_text() if path.exists() else ""
        try:
            new_contents[fe.filename] = apply_edits(original, fe.edit_list)
        except Exception as exc:
            # Any editor failure (EditError or an unexpected one) → treat as a
            # rejected plan and feed it back, never crash the loop.
            kind = "" if isinstance(exc, EditError) else f" [{type(exc).__name__}]"
            return (
                f"Edit to {fe.filename} could not be applied{kind}: {exc}. "
                "Re-check the target name / operation and try again.",
                {},
            )
    return None, new_contents


def write_files(model_dir: Path, new_contents: Dict[str, str]) -> None:
    for fname, content in new_contents.items():
        (model_dir / fname).write_text(content)
        log.info("Applied edits to %s (%d chars)", fname, len(content))


# ---------------------------------------------------------------------------
# Context builders (Step 1) — fresh & bounded each iteration
# ---------------------------------------------------------------------------


def _read_code_block(model_dir: Path, numbered: bool = True) -> str:
    """Render current mutable files, optionally with line numbers.

    Line numbers help the LLM reason about structure, but the surgical editor
    targets by *name*, not line, so numbers are advisory only.
    """
    blocks: List[str] = []
    for fname in sorted(config.ALLOWED_FILES):
        path = model_dir / fname
        lang = "yaml" if fname.endswith((".yaml", ".yml")) else "python"
        blocks.append(f"### {fname}")
        if not path.exists():
            blocks.append("```\n(file not found)\n```")
            continue
        text = path.read_text()
        if numbered:
            text = "\n".join(
                f"{i:4d} | {ln}" for i, ln in enumerate(text.splitlines(), 1)
            )
        blocks.append(f"```{lang}\n{text}\n```")
    return "\n\n".join(blocks)


def _log_memory() -> str:
    entries, best = read_logs(config.LOGS_FILE)
    if not entries:
        return "_(no previous experiments)_"
    rows = ["| # | Target | Description | Status | Score |",
            "|---|--------|-------------|--------|-------|"]
    for e in entries[-15:]:
        score = f"{e.score:.4f}" if e.score is not None else "—"
        rows.append(f"| {e.number} | {e.target or '—'} | {e.description} | {e.status} | {score} |")
    tail = f"\n\n**Best score so far: {best:.4f}**" if best is not None else ""
    return "\n".join(rows) + tail


_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an autonomous ML research agent. Each turn you propose ONE atomic
    experiment to improve a target metric on a machine-learning task, by emitting
    *surgical edits* to the code (not whole-file rewrites).

    ## Atomic target groups (edit files from exactly ONE per turn)
    1. model.py       — model architecture (build_model)
    2. loss.py        — loss function (build_loss); the key lever for imbalance
    3. optimizer.py   — optimiser & LR schedule (build_optimizer)
    4. transforms.py + config.yaml — feature engineering / resampling + hyperparams

    ## Edit operations (prefer the smallest that does the job)
    - replace_function_body: new statements for an existing function (no def line)
    - replace_definition:    full new function/class incl. signature
    - insert_definition:     add a new function/class (needs position)
    - delete_definition:     remove a function/class
    - add_import / replace_imports: manage imports
    - replace_global:        replace a module-level assignment
    - replace_file:          LAST RESORT (whole file); required for config.yaml

    ## Rules
    - Exactly ONE target group per experiment.
    - Keep the function *contracts* intact (same names/signatures run.py calls).
    - Ground each change in the experiment history — don't repeat failed ideas.
    - Be economical: the fewer/smaller the edits, the cheaper the run.

    Return a JSON object matching the ExperimentPlan schema.
    """
)


def build_context(
    model_dir: Path, feedback: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Assemble the message list for one LLM call (fresh, bounded)."""
    # System message — marked for prompt caching on Anthropic (it never changes).
    system_content: Any = _SYSTEM_PROMPT
    if config.LLM_MODEL.startswith("anthropic/") and _cache_enabled():
        system_content = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]

    if config.WIKI_FILE.exists():
        messages.append(
            {"role": "user", "content": f"# Problem\n\n{config.WIKI_FILE.read_text()}"}
        )

    messages.append(
        {"role": "user", "content": f"# Current code\n\n{_read_code_block(model_dir)}"}
    )
    messages.append(
        {"role": "user", "content": f"# Experiment history\n\n{_log_memory()}"}
    )

    # Per-agent exploration nudge (diversifies a population of agents).
    if config.AGENT_HINT:
        messages.append({
            "role": "user",
            "content": f"# Exploration focus for this agent\n{config.AGENT_HINT}",
        })

    if feedback:
        messages.append({"role": "user", "content": feedback})

    messages.append({"role": "user", "content": "Propose the next experiment."})
    return messages


def _cache_enabled() -> bool:
    import os

    return os.environ.get("PROMPT_CACHE", "1") not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# Scoring (Section 4)
# ---------------------------------------------------------------------------


def _classify(score: Optional[float], best: Optional[float], direction: str) -> str:
    if score is None:
        log.warning("No score available; classifying as 'lower'")
        return "lower"
    if best is None:
        return "statistically better"  # first successful run sets the bar

    # Signed relative improvement (positive = better, for either direction).
    raw = (score - best) / (abs(best) if best != 0 else 1.0)
    improve = raw if direction == "max" else -raw

    if improve >= config.STATISTICAL_DELTA:
        return "statistically better"
    if improve > 0:
        return "better"
    if improve >= -config.STATISTICAL_DELTA:
        return "lower"
    return "statistically lower"


_SUCCESS = {"statistically better", "better"}


# ---------------------------------------------------------------------------
# Main loop (Section 7)
# ---------------------------------------------------------------------------


def run_loop(
    model_dir: Path | str | None = None,
    max_iterations: int | None = None,
    llm: Optional[LLMClient] = None,
) -> None:
    """Run the optimisation agent.

    Args:
        model_dir: root of the model project (defaults to ``config.MODEL_DIR``).
        max_iterations: stop after N iterations (``None`` = run forever).
        llm: injected LLM client (tests pass a mock); defaults to a real client.
    """
    md = Path(model_dir or config.MODEL_DIR).resolve()
    if not md.exists():
        raise FileNotFoundError(f"Model directory does not exist: {md}")

    # Where experiments come from. The loop below is held FIXED across proposers,
    # so PROPOSER=random is a true control arm (RESEARCH.md §4 "Controls").
    # An injected client (tests) always wins over config.
    proposer = make_proposer(llm)
    tracker: Tracker = make_tracker(
        config.TRACKER,
        runs_dir=config.RUNS_DIR,
        target_metric=config.TARGET_METRIC,
        direction=config.TARGET_DIRECTION,
    )

    _entries, best_score = read_logs(config.LOGS_FILE)
    consecutive_crashes = 0
    feedback: Optional[str] = None  # transient message for the next turn
    iteration = 0

    # --- persistent archive (durable; lives outside model_dir) ---------------
    session = new_session_id()
    archive = RunArchive(
        config.ARCHIVE_DIR, session, model=config.LLM_MODEL,
        task=f"{config.TARGET_METRIC} ({config.TARGET_DIRECTION}) on {md.name}",
        agent=config.AGENT_LABEL,
        config_snapshot=config.snapshot(),  # what the agent WAS (RESEARCH.md §6)
        variant=config.VARIANT, trial=config.TRIAL,
    )
    log.info("Archiving to %s (session %s, agent %s, variant %s/trial %d)",
             archive.jsonl, session, config.AGENT_LABEL, config.VARIANT, config.TRIAL)

    # --- cost accounting -----------------------------------------------------
    # `pending_*` accumulate spend since the last logged experiment (so LLM
    # retries and crash-fix turns roll into the experiment they belong to);
    # `last` is just the most recent call, used for per-attempt archive records;
    # `total_cost` is cumulative across the whole run.
    total_cost = float(_entries[-1].cum_cost) if _entries and _entries[-1].cum_cost else 0.0
    pending = {"cost": 0.0, "in": 0, "out": 0, "cached": 0}
    last = {"cost": 0.0, "in": 0, "out": 0, "cached": 0}

    def capture_usage() -> None:
        u = getattr(proposer, "last_usage", {}) or {}
        c = float(u.get("cost", 0.0) or 0.0)
        i = int(u.get("prompt_tokens", 0) or 0)
        o = int(u.get("completion_tokens", 0) or 0)
        ca = int(u.get("cached_tokens", 0) or 0)
        pending["cost"] += c; pending["in"] += i; pending["out"] += o; pending["cached"] += ca
        last.update(cost=c, **{"in": i, "out": o, "cached": ca})

    def archive_record(plan, target, status, score, best_before, *,
                       run=None, kept=False, error=None) -> None:
        archive.record_experiment(
            iteration=iteration, target=target, status=status, plan=plan,
            score=score, best_before=best_before,
            run_id=getattr(run, "run_id", None),
            metrics=getattr(run, "metrics", None),
            series=getattr(run, "series", None),
            cost=last["cost"], tokens_in=last["in"], tokens_out=last["out"],
            tokens_cached=last["cached"], kept=kept, error=error,
        )

    def log_experiment(plan: ExperimentPlan, target: str, status: str,
                       score: Optional[float], best: Optional[float]) -> None:
        nonlocal total_cost
        total_cost += pending["cost"]
        tok = f"{pending['in']}→{pending['out']}"
        if pending["cached"]:
            tok += f" ({pending['cached']} cached)"
        _log(plan, target, status, score, best,
             tokens=tok, cost=pending["cost"], cum_cost=total_cost)
        pending.update({"cost": 0.0, "in": 0, "out": 0, "cached": 0})

    log.info("=== Agent starting ===")
    log.info("model_dir=%s  model=%s  tracker=%s  metric=%s(%s)  best=%s",
             md, config.LLM_MODEL, config.TRACKER,
             config.TARGET_METRIC, config.TARGET_DIRECTION, best_score)

    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        log.info("--- Iteration %d ---", iteration)
        # Whole-iteration guard: an unexpected error in ANY step reverts the
        # working tree and continues, so one bad turn never aborts the run
        # (max_iterations is always honoured).
        try:
            # Step 1-2: propose the next experiment (LLM, or the random control)
            try:
                plan: ExperimentPlan = proposer.propose(md, feedback)
            except Exception as exc:
                log.exception("Proposer failed")
                # NOTE: no `finally: feedback = None` here — finally runs after
                # except and would wipe this error feedback before the retry.
                feedback = f"The previous call failed ({exc}). Return valid ExperimentPlan JSON."
                continue
            feedback = None  # consumed by the proposal above
            capture_usage()  # accumulate token/cost spend for this call

            target = plan.edits[0].filename if plan.edits else "?"
            log.info("Plan: [%s] %s", target, plan.short_description)

            # Step 3: validate + dry-run edits -------------------------------
            error, new_contents = validate_plan(md, plan)
            if error is not None:
                log.warning("Plan rejected: %s", error)
                archive_record(plan, target, "rejected", None, best_score, error=error)
                feedback = f"Your plan was REJECTED: {error}"
                continue

            # Step 4: write edits --------------------------------------------
            write_files(md, new_contents)

            # Step 5: run training -------------------------------------------
            result = _run_training(md)

            # Step 6: crash handling -----------------------------------------
            if result.returncode != 0:
                consecutive_crashes += 1
                tb = (result.stderr or result.stdout or "(no output)")[-4000:]
                log.error("Training crashed (rc=%s, strike %d)", result.returncode, consecutive_crashes)

                if consecutive_crashes >= 2:
                    # Strike 2: give up this vector, log, revert, reset context.
                    archive_record(plan, target, "crushed", None, best_score, error=tb)
                    log_experiment(plan, target, "crushed", None, best_score)
                    reset_on_failure(md)
                    consecutive_crashes = 0
                    feedback = None
                else:
                    # Strike 1: revert broken files, feed traceback back.
                    archive_record(plan, target, "crashed", None, best_score, error=tb)
                    reset_on_failure(md)
                    feedback = (
                        f"Your edit to `{target}` crashed training:\n```\n{tb}\n```\n"
                        "The files were reverted to the last working state. "
                        "Propose a corrected experiment (fix the bug)."
                    )
                continue

            # Step 7: fetch metrics via tracker ------------------------------
            handle = tracker.parse_handle(result.stdout)
            if handle is None:
                consecutive_crashes += 1
                log.warning("No run handle in stdout (strike %d)", consecutive_crashes)
                if consecutive_crashes >= 2:
                    archive_record(plan, target, "crushed", None, best_score,
                                   error="no RUN_ID in stdout")
                    log_experiment(plan, target, "crushed", None, best_score)
                    reset_on_failure(md)
                    consecutive_crashes = 0
                else:
                    archive_record(plan, target, "no_run_id", None, best_score,
                                   error="no RUN_ID in stdout")
                    reset_on_failure(md)
                    feedback = (
                        "Training ran but printed no RUN_ID. Ensure run.py writes "
                        "metrics and prints 'RUN_ID=<id>'. Stdout tail:\n```\n"
                        f"{result.stdout[-1500:]}\n```"
                    )
                continue

            try:
                run: RunResult = tracker.fetch(handle)
            except Exception as exc:
                log.exception("Tracker fetch failed")
                reset_on_failure(md)
                feedback = f"Could not read metrics for run {handle}: {exc}."
                continue

            score = run.score(config.TARGET_METRIC, config.TARGET_DIRECTION)

            # Step 8: classify -----------------------------------------------
            status = _classify(score, best_score, config.TARGET_DIRECTION)
            log.info("score=%s  best=%s  status=%s", score, best_score, status)

            # Step 9: git protocol -------------------------------------------
            execute_git_protocol(md, status, plan.short_description, success_set=_SUCCESS)

            # Step 10: archive (durable) + log (working memory) --------------
            archive_record(plan, target, status, score, best_score,
                           run=run, kept=status in _SUCCESS)
            log_experiment(plan, target, status, score, best_score)

            # Step 11: update state ------------------------------------------
            consecutive_crashes = 0
            if score is not None and _is_better(score, best_score, config.TARGET_DIRECTION):
                best_score = score
            feedback = (
                f"Experiment on `{target}` → **{status}** "
                f"(score={score:.4f}, best={best_score:.4f}). "
                + ("Kept." if status in _SUCCESS else "Reverted.")
            )

        except Exception as exc:
            # Catch-all so a bug in one iteration can't end the whole run.
            import traceback as _tb
            log.exception("Unexpected error in iteration %d — reverting, continuing", iteration)
            try:
                archive.record({"iteration": iteration, "status": "internal_error",
                                "error": _tb.format_exc(), "detail": str(exc)})
            except Exception:
                pass
            try:
                reset_on_failure(md)
            except Exception:
                log.exception("Revert after error also failed")
            consecutive_crashes = 0
            feedback = ("The previous iteration hit an internal error and was skipped. "
                        "Propose a fresh, simple experiment on one target group.")
            continue

    log.info("=== Agent finished after %d iterations ===", iteration)
    log.info("Total LLM spend this run: $%.4f  (best %s = %s)",
             total_cost, config.TARGET_METRIC, best_score)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_training(md: Path) -> subprocess.CompletedProcess:
    log.info("Running: %s (cwd=%s)", " ".join(config.EXEC_COMMAND), md)
    try:
        return subprocess.run(
            config.EXEC_COMMAND,
            cwd=str(md),
            capture_output=True,
            text=True,
            timeout=config.RUN_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        log.error("Training timed out after %ss", config.RUN_TIMEOUT_SEC)
        return subprocess.CompletedProcess(
            config.EXEC_COMMAND, returncode=124,
            stdout=exc.stdout or "", stderr=f"TIMEOUT after {config.RUN_TIMEOUT_SEC}s",
        )


def _is_better(score: float, best: Optional[float], direction: str) -> bool:
    if best is None:
        return True
    return score > best if direction == "max" else score < best


def _log(plan: ExperimentPlan, target: str, status: str,
         score: Optional[float], best: Optional[float],
         tokens: str = "—", cost: Optional[float] = None,
         cum_cost: Optional[float] = None) -> None:
    summary = plan.reasoning[:200] + ("…" if len(plan.reasoning) > 200 else "")
    append_log_entry(
        config.LOGS_FILE, plan.short_description, summary, status,
        score=score, best_score=best, target=target,
        tokens=tokens, cost=cost, cum_cost=cum_cost,
    )
