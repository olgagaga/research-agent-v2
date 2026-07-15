# mysearch — a cheap autoresearch agent

A small, cost-optimized take on "autoresearch": instead of handing markdown specs
to a heavyweight coding agent, this is a **purpose-built loop** that generates an
idea, implements it as a *surgical* code edit, launches training, reads the
result, and repeats — keeping the model in its own bounded context so token spend
stays flat per iteration.

```
read context ─▶ propose ONE atomic experiment ─▶ apply surgical edit
     ▲                                                    │
     │                                              run training
  update logs.md ◀─ commit / revert ◀─ score ◀─ fetch metrics
```

## Layout

```
agent/              the agent library (no ML deps)
  orchestrator.py     the loop (validate → edit → train → track → git → log)
  editor.py           surgical AST-guided source editor (the token-cost lever)
  schemas.py          ExperimentPlan / FileEdits / EditOp (structured LLM output)
  tracker.py          pluggable metrics backend: LocalTracker | ClearMLTracker
  archive.py          durable append-only run archive  → history/
  analytics.py        reads history/ + parallel_runs/ (single source for the UI)
  llm.py              LLM client (OpenAI direct + OpenRouter), cost accounting
  git_manager.py      commit-on-success / hard-reset-on-failure
  logs_manager.py     logs.md working memory
  config.py           env-driven knobs
dashboard/          web UI (see dashboard/README.md)
  backend/            FastAPI — JSON API + serves the built SPA
  frontend/           React + Vite + TypeScript
history/            DURABLE archive — every experiment, ever (survives resets)
model_dir/          the research TARGET / working dir (its own git repo + venv)
  data.py, run.py     FIXED harness — the agent must not edit these
  model.py loss.py optimizer.py transforms.py config.yaml   AGENT-EDITABLE seams
  wiki.md             problem briefing for the LLM
parallel_runs/      population runs (worktrees + per-agent archives; runtime)
replays/            replay results — noise floor + decision audits (runtime)
docs/               RESEARCH.md · LAB_NOTEBOOK.md · TASK.md
tests/              editor unit tests + isolated end-to-end loop test
main.py             run one agent          parallel.py   run a population
replay.py           re-run a trajectory under R seeds — no LLM, $0 (see below)
```

**Separation of concerns:** `agent/` never imports the dashboard; the dashboard
never imports the loop. They meet only at `history/` (append-only JSONL) — the
agent writes it, `agent/analytics.py` reads it, the API serves it. `model_dir/`
is pure working directory: its own git repo, mutated and reverted by the agent,
and *nothing* durable lives there (which is why the archive is outside it).

## The task: rare-event medical detection

OpenML **mammography** — flag breast-cancer microcalcifications from 6 numeric
features. **~2.3% positive**, so it's a genuine imbalanced-detection problem with
no easy solution (plain accuracy is useless; predicting all-negative scores
97.7%). We optimize **AUPRC** (average precision) on a fixed, seeded split.

Baseline (tiny MLP + SGD + unweighted BCE) is deliberately under-trained, so
there is lots of headroom. Quote it as **val/AUPRC = 0.363 ± 0.219** (10 seeds,
`replay.py`) — *not* as a point value: across seeds the identical baseline commit
scores anywhere from **0.046 to 0.676**. Past the baseline the pipeline settles
down to σ ≈ 0.015. Strong pipelines reach ~0.6–0.75 on val.

**Read val scores against test.** val runs ~0.12 higher than the held-out test
split — a fixed offset of *this split*, present at the baseline before the agent
runs, not agent overfitting (`replay.py` measures the drift at +0.009 ± 0.086,
i.e. flat). Val is the agent's optimization signal; test is never shown to it and
is the number to believe.

A full training run is **~4 seconds on CPU** — cheap enough to loop many times,
and cheap enough that seed-averaging an evaluation costs far less than the LLM
call that proposed it.

## Setup

```bash
# 1) orchestrator env (light — no torch)
uv sync

# 2) training env (CPU torch), inside model_dir
cd model_dir && uv sync && cd ..

# 3) credentials — put your key in .env (see "LLM provider" below)
#    API_KEY=sk-...              # works for either provider
#    MAIN_MODEL=openai/gpt-5-mini
```

## LLM provider (OpenAI or OpenRouter)

`agent/llm.py` is **provider-aware** and auto-detects from `MAIN_MODEL`:

| `MAIN_MODEL` | Provider | API | Cost source |
|--------------|----------|-----|-------------|
| `openai/gpt-5-mini`, `gpt-5-mini`, `o3`, `gpt-4o-mini` | **OpenAI** | native `.parse()` structured output, automatic prompt caching | computed from tokens × pricing table |
| `anthropic/claude-…`, `google/…` (any `vendor/model`) | **OpenRouter** | `json_schema` strict, Anthropic prompt caching | read from OpenRouter usage/generation API |

- Key resolution: `OPENAI_API_KEY` / `OPENROUTER_API_KEY`, else the generic
  `API_KEY` (used for whichever provider is active). Force a provider with
  `LLM_PROVIDER=openai|openrouter`.
- **OpenAI cost**: OpenAI responses carry no cost field, so cost is computed from
  token counts, **billing cached input tokens at the discounted rate** (e.g.
  gpt-5-mini cached input is 10× cheaper than fresh input). Prices live in
  `_OPENAI_PRICING` in `agent/llm.py` as `(input, cached_input, output)` per 1M
  tokens (they change — verify/edit), or override per-run with `LLM_PRICE_IN` /
  `LLM_PRICE_CACHED` / `LLM_PRICE_OUT`. Unknown model → cost shown as 0.
- Reasoning models (`gpt-5*`, `o3`, `o4-mini`) automatically use
  `reasoning_effort` (`REASONING_EFFORT`, default `medium`) + `max_completion_tokens`.

Verified working end-to-end on `openai/gpt-5-mini`: iteration 1 chose
`loss.py pos_weight` for ~$0.0025 (single-seed val 0.32 → 0.69; replayed over 10
seeds it is **+0.328 ± 0.069**, the one unambiguously large win in the whole
trajectory — see `replay.py`).

The training subprocess is launched via `MODEL_DIR/.venv/bin/python run.py`.
On a machine that already has `torch`+`scikit-learn`+`pandas`, you can skip the
model venv and set `EXEC_COMMAND="python3 run.py"`.

## Run

```bash
python main.py --max-iterations 10        # 10 experiments then stop
python main.py                            # run until interrupted
python main.py -n 5 --verbose             # debug logging
```

Watch progress in `model_dir/logs.md` (the agent's memory) and `git log` inside
`model_dir` (every kept experiment is one commit).

Each `logs.md` row records **`Tokens(in→out)`**, per-experiment **`Cost $`**, and
running **`Cum $`** — so the exact LLM spend per experiment and cumulative total
are always visible. Cost comes from the provider's usage (OpenAI: tokens x pricing incl. cached discount); crash
retries roll their cost into the experiment they belong to. The final cumulative
spend is also logged at the end of the run.

## Dashboard (FastAPI + React)

A real web app — see [dashboard/README.md](dashboard/README.md).

```bash
uv sync --extra dashboard                            # fastapi + uvicorn
cd dashboard/frontend && npm install && npm run build && cd ../..
python dashboard/backend/app.py                      # http://localhost:8000
```

Two views, both live-polling (5 s, pausable) with light/dark themes:
- **Solo run** — the **optimization curve** front and centre (best-so-far, with
  each experiment's own score coloured kept vs reverted against the naive
  baseline), KPIs, spend-per-experiment, the best run's learning curve, and the
  full experiment log. Session picker.
- **Population** — per-agent trajectories + the **best-of-N** curve, and a ranked
  leaderboard. Run picker.

It reads the **durable archive** (`history/`), not `logs.md`, so it reflects real
history even after `model_dir` is reset. For hot-reload dev, run the backend and
`npm run dev` (:5173, proxies `/api`) side by side.

## Population runs (N agents in parallel)

Run several agents on the same task at once, each in its own **git worktree** and
OS process, then keep the best (`parallel.py`):

```bash
python parallel.py --agents 4 --iterations 6                 # from model_dir HEAD
python parallel.py --agents 8 --iterations 10 --base <sha>   # clean population from a baseline
```

Each agent explores with a different `REASONING_EFFORT` + focus hint. A live
leaderboard streams while they run; on finish you get `parallel_runs/<name>/`
with per-agent archives, `leaderboard.json`, and merged `all_experiments.jsonl`.
View it:

```bash
python dashboard/backend/app.py    # then open the Population tab at :8000
```

Worktrees are removed afterward but each agent's result is kept as a git branch
(`par/<name>/agent-NN`). The binding constraint at high N is the LLM rate limit,
not CPU (training is ~4 s).

## Replay — measuring the noise floor (`replay.py`)

A measurement instrument, not an agent: **no LLM calls, $0**. It re-runs each
commit of a trajectory under R fresh seeds, with the **harness pinned** to your
working tree (`run.py`/`data.py` copied into throwaway worktrees) and **only the
agent-editable seams swapped per commit** — so every step is scored by identical
evaluation code, and the only thing that varies is the agent's edit.

```bash
python replay.py                         # whole trajectory × 5 seeds
python replay.py --seeds 10 --jobs 10    # 11 commits × 10 seeds ≈ 6 min on 22 cores
python replay.py --commits HEAD -s 20    # noise floor of one pipeline
```

It prints three things:

- **Noise floor** — within-commit σ of the target metric. On mammography: **0.015
  at plateau, 0.219 at the baseline.**
- **Self-report** — target vs held-out metric per step, and whether the gap
  *drifts* as the agent climbs. Only drift is attributable to the agent; a nonzero
  baseline gap just means val is easier than test.
- **Decision audit** — replays the loop's own keep/revert rule at n seeds instead
  of 1, labelling each kept experiment REAL / NOISE / REGRESSION. On the current
  trajectory: **3 of 10 kept experiments are real.**

Why it matters: the loop commits when `improve > 0` (`STATISTICAL_DELTA` only
*labels* the outcome, it does not gate the commit), judged from a **single seed**.
With σ ≈ 0.015, an edit that changes nothing is committed ~50% of the time and the
incumbent only ratchets *up* on val. Read `docs/LAB_NOTEBOOK.md` (2026-07-15)
before quoting any score from this repo.

Task-agnostic by construction — it assumes only "a git repo + editable seams + a
run command that prints `RUN_ID` and writes `runs/<id>/metrics.json`", so a second
task needs no changes to it. Output: `replays/<ts>/runs.jsonl` + `summary.json`.

## How costs are kept low

| Lever | What it does |
|-------|--------------|
| **Surgical AST edits** | The LLM emits `replace_function_body` / `replace_global` / … instead of rewriting whole files. Output tokens dominate cost; a 1-line body swap costs ~1 line, not the file. See `agent/editor.py`. |
| **Fresh bounded context** | Context is rebuilt each turn (system + wiki + current code + recent log) rather than accumulated, so per-call tokens are `O(codebase)`, not `O(iterations × codebase)`. Durable memory lives in `logs.md`. |
| **Prompt caching** | The stable system prompt is marked `cache_control` on Anthropic (toggle with `PROMPT_CACHE=0`). |
| **Reasoning-effort control** | `REASONING_EFFORT` (default `medium`) tunes thinking-token spend. |
| **Cheap task + CPU** | ~4s/run, small model, fixed epoch cap — the *compute* side is cheap too. |
| **Dry-run validation** | Edits are validated (atomicity + `ast.parse`) in memory before any training runs, so a malformed plan never burns a training run. |

Per-call cost is also fetched from OpenRouter's generation API (`agent/llm.py`).

## Swapping in ClearML later (not integrated now, but wired for it)

Experiment tracking is behind the `Tracker` protocol (`agent/tracker.py`). The
default `LocalTracker` reads `runs/<id>/metrics.json` that `run.py` writes — zero
dependencies, zero cost. To move to ClearML with **no orchestrator changes**:

1. `uv sync --extra clearml`
2. In `model_dir/run.py`, enable the ClearML hook (create a `Task`, report
   scalars, and `print(f"CLEARML_TASK_ID={task.id}")`).
3. `export TRACKER=clearml`

The orchestrator only ever talks to `Tracker.parse_handle()` / `Tracker.fetch()`,
so nothing else changes. `ClearMLTracker` is already stubbed.

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `MODEL_DIR` | `./model_dir` | mutable model project |
| `MAIN_MODEL` | `anthropic/claude-sonnet-4.6` | LLM (OpenRouter id) |
| `TRACKER` | `local` | `local` or `clearml` |
| `TARGET_METRIC` / `TARGET_DIRECTION` | `val/auprc` / `max` | objective |
| `STATISTICAL_DELTA` | `0.03` | ±3% = "statistically" better/lower |
| `EXEC_COMMAND` | `<venv>/python run.py` | training command |
| `RUN_TIMEOUT_SEC` | `900` | per-run wall-clock cap |
| `REASONING_EFFORT` | `medium` | LLM thinking budget |
| `PROMPT_CACHE` | `1` | Anthropic prompt caching on/off |
| `GIT_PUSH` | `1` | push kept experiments (auto-skips if no `origin`) |

## Status classes (logged to `logs.md`)

`statistically better` (≥+3%, committed) · `better` (<+3%, committed) ·
`lower` (<−3%… i.e. within −3%, reverted) · `statistically lower` (≥−3%,
reverted) · `crushed` (crashed twice in a row → reverted, context reset).

## Tests

```bash
python tests/test_editor.py   # 17 AST-editor unit tests
python tests/test_loop.py     # end-to-end loop with a MOCK LLM (no API cost)
```
