# Dashboard — FastAPI + React

Web UI for the autoresearch agent. Reads the **durable archive** (`history/`) and
population runs (`parallel_runs/`) — never the ephemeral `logs.md`, so it
survives `model_dir` resets.

```
dashboard/
├── backend/     FastAPI — serves JSON API + (in prod) the built SPA
│   └── app.py
└── frontend/    React + Vite + TypeScript
    └── src/  App.tsx · views.tsx · charts.tsx · api.ts · types.ts · styles.css
```

All data reading lives in `agent/analytics.py` (single source of truth); the
backend is only routing.

## Run

**Production (one server, no node needed after building once):**
```bash
cd dashboard/frontend && npm install && npm run build && cd ../..
python dashboard/backend/app.py            # http://localhost:8000  (API + UI)
```

**Development (hot reload):**
```bash
python dashboard/backend/app.py            # terminal 1 — API on :8000
cd dashboard/frontend && npm run dev       # terminal 2 — UI on :5173 (proxies /api)
```

Deps: `uv sync --extra dashboard` (fastapi + uvicorn). Override the port with
`DASHBOARD_PORT`.

## API

| Endpoint | Returns |
|----------|---------|
| `GET /api/health` | liveness |
| `GET /api/config` | metric, direction, model, task, archive path |
| `GET /api/sessions` | every session in the archive (best, cost, counts) |
| `GET /api/solo?session=<id>&all=<bool>` | one session's experiments, curves, summary |
| `GET /api/parallel` | list of population runs |
| `GET /api/parallel/{name}` | per-agent trajectories, best-of-N, leaderboard |

## UI

- **Solo run** — KPIs, optimization curve (best-so-far + per-experiment points,
  kept vs reverted, naive baseline), spend-per-experiment bars, best run's
  learning curve, full experiment log. Session picker.
- **Population** — per-agent best-so-far curves + the bold **best-of-N** line,
  and a ranked leaderboard (effort, focus hint, best, kept, cost). Run picker.
- Live polling every 5 s (pausable), light/dark theme, no chart library
  (hand-rolled SVG → ~51 kB gzipped).
