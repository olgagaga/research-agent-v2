"""
Git protocol (Section 6 of the spec).

* On **success** (``statistically better`` / ``better``): add → commit → push.
* On **failure** (``lower`` / ``statistically lower`` / ``crushed``):
  ``git reset --hard HEAD`` + ``git clean -fd`` to revert to the last good state.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# Statuses that signal a *successful* experiment
_SUCCESS_STATUSES = frozenset({"statistically better", "better"})


def _run_git(args: list[str], cwd: Path, capture: bool = False) -> subprocess.CompletedProcess:
    """Thin wrapper around ``subprocess.run`` for git commands."""
    cmd = ["git"] + args
    log.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=capture,
        text=True,
    )


def commit_on_success(model_dir: Path, message: str) -> None:
    """Stage all changes, commit, and push.

    Args:
        model_dir: Root of the model git repository.
        message: Commit message (the LLM ``short_description``).
    """
    log.info("Committing successful experiment: %s", message)

    result = _run_git(["add", "."], cwd=model_dir)
    if result.returncode != 0:
        log.error("git add failed (rc=%d)", result.returncode)
        return

    result = _run_git(["commit", "-m", message], cwd=model_dir)
    if result.returncode != 0:
        log.error("git commit failed (rc=%d)", result.returncode)
        return

    # Only push if an 'origin' remote is configured (avoids noisy failures on a
    # purely local repo). Disable entirely with GIT_PUSH=0.
    if os.environ.get("GIT_PUSH", "1") in ("0", "false", "False"):
        return
    remotes = _run_git(["remote"], cwd=model_dir, capture=True)
    if "origin" not in (remotes.stdout or ""):
        log.info("No 'origin' remote — skipping push (local-only repo)")
        return
    result = _run_git(["push", "origin", "HEAD"], cwd=model_dir)
    if result.returncode != 0:
        log.warning("git push failed (rc=%d) — continuing anyway", result.returncode)


def reset_on_failure(model_dir: Path) -> None:
    """Hard-reset tracked files and remove untracked artifacts.

    Args:
        model_dir: Root of the model git repository.
    """
    log.warning("Resetting model_dir to last committed state (git reset --hard HEAD)")

    result = _run_git(["reset", "--hard", "HEAD"], cwd=model_dir)
    if result.returncode != 0:
        log.error("git reset --hard failed (rc=%d)", result.returncode)

    result = _run_git(["clean", "-fd"], cwd=model_dir)
    if result.returncode != 0:
        log.error("git clean -fd failed (rc=%d)", result.returncode)


def execute_git_protocol(
    model_dir: Path,
    status: str,
    message: str,
    success_set: frozenset[str] | set[str] | None = None,
) -> None:
    """Single entry-point: commit on success, reset on failure.

    Args:
        model_dir: Root of the model git repository.
        status: The experiment status string.
        message: Commit message (LLM ``short_description``).
        success_set: Statuses treated as success (defaults to the module set).
    """
    success = success_set if success_set is not None else _SUCCESS_STATUSES
    if status in success:
        commit_on_success(model_dir, message)
    else:
        reset_on_failure(model_dir)
