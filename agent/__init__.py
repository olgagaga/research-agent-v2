"""
LLM-driven ML optimization agent.

Public surface::

    from agent import run_loop
    from agent import config

``.env`` (if present in the project root) is loaded on import so API keys and
overrides are available everywhere.
"""

from __future__ import annotations

from pathlib import Path

# Load .env early so config picks up env vars.
try:
    from dotenv import load_dotenv

    _here = Path(__file__).resolve().parent
    # Project-root .env first, then a co-located agent/.env (which overrides).
    for _env in (_here.parent / ".env", _here / ".env"):
        if _env.exists():
            load_dotenv(_env, override=True)
except Exception:  # dotenv optional — never fatal
    pass

from agent import config  # noqa: E402
from agent.orchestrator import run_loop  # noqa: E402

__all__ = ["run_loop", "config"]
