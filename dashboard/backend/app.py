"""
FastAPI backend for the autoresearch dashboard.

Serves the durable run archive (``history/``) and population runs
(``parallel_runs/``) as JSON, and — in production — the built React frontend.
All data reading lives in :mod:`agent.analytics`; this module is just routing.

Run (dev, API only):
    python dashboard/backend/app.py            # http://localhost:8000
    #   frontend dev server (vite) runs separately on :5173 and calls this API

Run (production, single server):
    (build the frontend first: cd dashboard/frontend && npm run build)
    python dashboard/backend/app.py            # serves API + built SPA at :8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the repo importable (agent package) regardless of CWD.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from agent import analytics, config  # noqa: E402

app = FastAPI(title="Autoresearch Dashboard", version="1.0")

# Dev: the Vite dev server (:5173) calls this API cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------------- API


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/config")
def get_config() -> dict:
    return {
        "metric": config.TARGET_METRIC,
        "direction": config.TARGET_DIRECTION,
        "model": config.LLM_MODEL,
        "model_dir": config.MODEL_DIR.name,
        "task": "mammography · rare-event medical detection (~2.3% positive)",
        "archive": str(config.ARCHIVE_DIR),
    }


@app.get("/api/sessions")
def sessions() -> list[dict]:
    return analytics.list_sessions()


@app.get("/api/solo")
def solo(session: str | None = Query(default=None),
         all: bool = Query(default=False)) -> dict:
    return analytics.collect_solo(session=session, all_sessions=all)


@app.get("/api/parallel")
def parallel() -> list[dict]:
    return analytics.list_parallel_runs()


@app.get("/api/parallel/{name}")
def parallel_run(name: str) -> dict:
    run_dir = analytics.parallel_runs_dir() / name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"no parallel run {name!r}")
    return analytics.collect_parallel(name)


# --------------------------------------------------------------------------- static

# Serve the built SPA at "/" if it exists (production single-server mode).
_DIST = ROOT / "dashboard" / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DASHBOARD_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
