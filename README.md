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
agent/            orchestrator (no ML deps)
  orchestrator.py   the loop (validate → edit → train → track → git → log)
  editor.py         surgical AST-guided source editor (the token-cost lever)
  schemas.py        ExperimentPlan / FileEdits / EditOp (structured LLM output)
  tracker.py        pluggable metrics backend: LocalTracker | ClearMLTracker
  llm.py            OpenRouter wrapper (prompt caching, cost tracking)
  git_manager.py    commit-on-success / hard-reset-on-failure
  logs_manager.py   logs.md durable memory
  config.py         env-driven knobs
model_dir/        the research TARGET (its own git repo + venv)
  data.py, run.py   FIXED harness — the agent must not edit these
  model.py loss.py optimizer.py transforms.py config.yaml   AGENT-EDITABLE seams
  wiki.md           problem briefing for the LLM
main.py           CLI entry point
```

## The task: rare-event medical detection

OpenML **mammography** — flag breast-cancer microcalcifications from 6 numeric
features. **~2.3% positive**, so it's a genuine imbalanced-detection problem with
no easy solution (plain accuracy is useless; predicting all-negative scores
97.7%). We optimize **AUPRC** (average precision) on a fixed, seeded split.

Baseline (tiny MLP + SGD + unweighted BCE) reaches AUPRC ≈ 0.32 and is clearly
under-trained — lots of headroom. In smoke tests the agent's levers already move
it to ~0.74 (Adam) in one edit. Strong pipelines reach ~0.6–0.75.

A full training run is **~4 seconds on CPU** — cheap enough to loop many times.

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
`loss.py pos_weight`, lifting val/AUPRC 0.32 → 0.69 for ~$0.0025.

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
are always visible. Cost is read from OpenRouter's usage/generation API; crash
retries roll their cost into the experiment they belong to. The final cumulative
spend is also logged at the end of the run.

## Dashboard

`dashboard.py` turns `logs.md` + `runs/*/metrics.json` into a single
self-contained HTML telemetry page (inline SVG charts, no external assets —
works offline and as a shareable artifact):

```bash
python dashboard.py           # -> ./dashboard.html  (project root)
python dashboard.py --open    # ...and open it in a browser
```

It centres on the **optimization curve**: best-so-far target metric across
experiments, with each experiment's own score coloured by kept (committed) vs
reverted, against the naive-baseline reference. Around it: KPI tiles (best
metric, kept/reverted, total spend, success rate), per-experiment cost bars, the
best run's learning curve, and the full status-coloured experiment log. Dark and
light themes; regenerate it any time (or after each run) to refresh.

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
python tests/test_editor.py   # 15 AST-editor unit tests
python tests/test_loop.py     # end-to-end loop with a MOCK LLM (no API cost)
```
