# Lab notebook — autoresearch agent

Running source book for a future blog/paper. **Append a dated entry after each
significant step** (design decision, run, result, bug, fix). Keep the *headline
results* table current. Raw per-experiment data lives in `history/` (durable,
append-only — see §Data).

---

## Headline results (keep current)

**All numbers are 10-seed replays (`replay.py`), not single runs.** Single-seed
scores on this task are not reportable — see the 2026-07-15 entry.

| Task | Model | Baseline (10 seeds) | Best (agent, 10 seeds) | Cost | Notes |
|------|-------|--------------|--------------|------|-------|
| Mammography rare-event (AUPRC, ~2.3% pos) | gpt-5-mini | val **0.363 ± 0.219** · test **0.249 ± 0.159** | val **0.774 ± 0.011** · test **0.650 ± 0.023** (peak, step 6) | ~$0.10 | 9 kept commits, of which **3 are real improvements** and 6 are noise |

**One-liner for the paper:** a cheap (~$0.02–0.10/run) LLM agent, editing code via
surgical AST ops on a fixed harness, lifts **held-out** AUPRC on an imbalanced
medical-detection task from 0.25 to 0.65 — but reaches that ceiling in **3 real
experiments**, and spends the remaining ~40% of its budget committing noise,
because its keep/revert rule fires at `Δ > 0` on a single-seed estimate whose
σ ≈ 0.015.

**Do not quote:** "0.32 baseline", "0.32 → 0.789", or the 3-agent "loss lever
dominates" result. All three are single-seed artifacts; the 2026-07-15 entry
records why.

**Also do not quote the agent's gains as evidence the LLM works.** The control
(2026-07-15, `sweep.py --vary PROPOSER=llm,random`) found **no detectable
advantage over random search** on the same action space: llm 0.7643 ± 0.0281 vs
random 0.7571 ± 0.0412 (n=5, Welch t = 0.33), at $0.19 vs $0.00. Until that is
overturned on a larger action space, the defensible claim is about the **harness**
(surgical edits + keep/revert ratchet + cheap eval loop), not about the model.

---

## Data (where the evidence lives)

- **`history/experiments.jsonl`** — durable, append-only, one JSON line per
  experiment (kept, reverted, or crashed) with: session, iteration, target lever,
  status, score, best-before, the exact **edits**, full **metrics + training
  series**, cost, tokens, and any **traceback**. Never deleted by resets/tests.
  Load: `pandas.read_json("history/experiments.jsonl", lines=True)`.
- **`history/sessions.jsonl`** — one line per run (session id, model, task, start).
- `model_dir/logs.md` — the agent's *working* memory (can be reset; not the source
  of truth). `model_dir` git log — one commit per kept experiment.
- **Dashboard** — `python dashboard/backend/app.py` → http://localhost:8000
  (FastAPI + React; Solo + Population views, reads `history/`).

---

## Log

### 2026-07-14 — Scaffold → working agent
- Reviewed the initial scaffold: canonical `agent/` package was broken (missing
  `__init__`), hard-wired to ClearML, and had **no `model_dir` to train on**.
- Built: **surgical AST editor** (`agent/editor.py`, 15 unit tests) so the LLM
  emits `replace_function_body` etc. instead of whole-file rewrites (main
  token-cost lever); **pluggable tracker** (local now, ClearML later); a
  `model_dir` training target on OpenML **mammography** (rare-event medical,
  2.3% positive, AUPRC) with 4 agent-editable levers + fixed harness.
- Baseline (tiny MLP + SGD + unweighted BCE): **val/AUPRC ≈ 0.32**, still rising
  at last epoch → deliberately under-trained, leaves headroom.

### 2026-07-14 — First real runs (OpenAI gpt-5-mini)
- Made `llm.py` provider-aware (OpenAI direct + OpenRouter); OpenAI cost computed
  from tokens with **cached-input discount** (gpt-5-mini cached = 10× cheaper).
- **7-experiment demo run:** 0.32 → **0.747**, $0.024, 5 kept / 2 reverted.
  Winning levers: class-weighted → focal loss, standardization, AdamW+cosine,
  quadratic features. Reverted: oversampling, an over-wide residual net.
- Built `dashboard.py` (self-contained telemetry HTML; optimization curve centre).

### 2026-07-14 — User's 12-experiment run + two system bugs found
- User ran `main.py -n 20`; it **stopped at 12**. Root cause: only the LLM call
  was inside try/except — an **unhandled exception in a later step (likely the
  editor on a malformed edit) aborted the whole run**. Best reached ~**0.789**,
  ~$0.10; 7 kept (survive as git commits).
- **Fix 1 (resilience):** `validate_plan` now catches *any* editor exception
  (not just `EditError`); the **entire iteration body is wrapped** so a bad turn
  reverts and continues — `max_iterations` is always honoured.
- **Incident (data loss):** while verifying the fix I ran `tests/test_loop.py`,
  which reset the *live* `model_dir` and **wiped the user's `logs.md` + `runs/`**
  (both gitignored → unrecoverable). Kept experiments survived in git; logs.md
  partially reconstructed from commits.
- **Fix 2 (persistence — the real lesson):** added `agent/archive.py` — a
  durable, append-only `history/` archive **outside `model_dir`** that nothing in
  the loop deletes. Every experiment is recorded with full provenance. Rewrote
  `tests/test_loop.py` to run entirely in a **tempdir** with safety assertions so
  it can never touch real data or `history/` again.
- **Takeaway for the paper/system:** short-term agent memory (logs.md) must be
  separate from the immutable experiment record (history/). Reverting the *code*
  must never revert the *evidence*.

### 2026-07-14 — Persistence catches a real editor bug (first payoff)
- The new `history/` archive immediately earned its keep. A real run showed the
  agent proposing `transforms.py replace_function_body` and getting **rejected 6×
  in a row** — visible only because the archive records rejected attempts with
  their error + exact edit.
- **Root cause (editor bug, not the LLM):** the LLM emitted a *ragged first line*
  — first statement flush-left, remaining lines indented by 4. `textwrap.dedent`
  can't normalize that (common prefix = 0), so re-indentation produced
  "unexpected indent" every time → a stuck loop wasting ~$0.03.
- **Fix:** `_normalize_body` in `agent/editor.py` detects the ragged pattern
  (first non-blank line less-indented than the rest, not a block opener), lifts
  the first line, and validates via `ast.parse`; falls back safely otherwise.
  Verified against the exact archived edit + 2 new regression tests (17 total).
- **Lesson:** persistent, append-only run history isn't just for papers — it's
  the debugging substrate. Rejected/crashed attempts are the most informative
  records; never discard them.

### 2026-07-15 — Population search: N agents in parallel worktrees (RESEARCH.md §3)
- **Built `parallel.py`** — runs N agents on the same task simultaneously, each
  isolated, then keeps the best. Design decisions:
  - **Isolation via git worktrees:** each agent = a worktree of `model_dir` on
    its own branch. Shared object store (cheap), separate working tree + commits
    (no collisions). Verified worktree add/remove leaves `model_dir` HEAD intact.
  - **Process-per-agent:** config is process-global (paths derive from `MODEL_DIR`
    env), so each agent is a separate `main.py` process with its own `MODEL_DIR`
    + `AUTORESEARCH_HISTORY`. Clean boundary, no config refactor.
  - **Per-agent archives:** each writes its own `experiments.jsonl` (a shared
    file would corrupt under concurrent >4 KB appends — POSIX only guarantees
    atomicity below PIPE_BUF).
  - **Diversity:** per-agent `REASONING_EFFORT` + an `AGENT_HINT` nudge injected
    into the prompt ("prioritise the loss", "…architecture", …). Session ids got
    a random suffix to avoid same-second collisions.
- **Tracking:** live leaderboard polls each agent's archive; on finish writes
  `leaderboard.json` + merged `all_experiments.jsonl`. New dashboard mode
  `dashboard.py --parallel <run>` overlays per-agent best-so-far curves + the
  **best-of-N** population curve + a ranked leaderboard.
- **First result (3 agents × 2 iters from baseline, $0.019):**

  | agent | focus (hint) | effort | best val/AUPRC |
  |-------|--------------|--------|----------------|
  | agent_00 | loss / imbalance | low | **0.693** |
  | agent_02 | optimiser / LR | low | 0.679 |
  | agent_01 | architecture only | medium | 0.407 |

  **Finding:** on this imbalanced task the **loss lever dominates** — the agent
  steered to architecture-only (a1) barely moved off the baseline, while the
  loss- and optimiser-focused agents nearly doubled AUPRC in 2 experiments. The
  population makes lever-importance legible in a way a single trajectory can't.
- **Caveats / next:** the real test is **best-of-N vs a single agent at equal
  total budget** (does breadth beat depth?), plus a "breed from top-2" second
  round. Also: 2 iters is too few to conclude — scale to N=8 × 10 iters. LLM rate
  limit (not CPU) is the binding constraint at higher N.

### 2026-07-15 — Repo restructure + real dashboard (FastAPI + React)
- The single-file static HTML generator had outgrown itself (data logic, styling
  and JS all tangled in one Python file). Replaced with a proper web app and gave
  the repo a real architecture.
- **Layout now:** `agent/` (library) · `dashboard/{backend,frontend}` ·
  `history/` (durable data) · `model_dir/` (working dir) · `docs/` · `tests/` ·
  `main.py` / `parallel.py` (entry points). Deleted the dead `a/` prototype and
  the empty `context/` leftovers.
- **The key boundary:** `agent/` never imports the dashboard; the dashboard never
  imports the loop. They meet only at `history/` (append-only JSONL). Extracted
  all run-data reading into **`agent/analytics.py`** — one source of truth for the
  API (and anything else). The backend is pure routing.
- **Backend** (`dashboard/backend/app.py`): FastAPI serving `/api/{config,
  sessions,solo,parallel,parallel/{name}}`, plus the built SPA at `/` in prod
  (single server, no node needed after one build). CORS for the Vite dev server.
- **Frontend** (`dashboard/frontend`): React + Vite + TS. Two views (Solo,
  Population), live polling every 5 s (pausable), light/dark, session/run pickers.
  **No chart library** — hand-rolled SVG components → 160 kB JS / **51 kB gzipped**.
  Keeps the deps small and the aesthetic consistent with the old telemetry look.
- **Verified:** `tsc -b && vite build` clean; backend serves API + SPA together
  with real data (12 experiments, best 0.7938; the 3-agent demo population).
- **Learning:** the static exporter was the right call at v0 (zero deps, shareable
  artifact) and the wrong one by v1 — once you want session pickers, live updates
  and multiple views, "regenerate a file" stops being a UI. The rewrite was cheap
  precisely because the data layer was already clean (JSONL archive), which is the
  real lesson: **stable data contract first, presentation is then disposable.**

### 2026-07-15 — `replay.py`: the noise floor, and what it invalidates
Built **`replay.py`** — a measurement instrument, not an agent: **no LLM calls**,
$0 per run. It re-runs each commit of a trajectory under R fresh seeds with the
**harness pinned** to the working tree (run.py/data.py copied into throwaway
worktrees) and **only the agent-editable seams swapped per commit**, so every step
is scored by byte-identical evaluation code. Two harness changes made it possible:
`SEED` env override (the seed lives in `config.yaml`, an agent-editable seam —
also a latent reward-hacking surface) and `*_final` metrics alongside the
best-epoch ones, to separate epoch-selection from split offset.

First full replay: **11 commits × 10 seeds = 110 runs, 6 minutes, $0, 0 failures.**

**1. The baseline is not a number, it's a distribution.** The identical baseline
commit over 10 seeds: `0.046 0.116 0.208 0.235 0.324 0.329 0.518 0.574 0.608
0.676` → **0.363 ± 0.219**. The "0.32" quoted everywhere was one draw. Past the
baseline the pipeline stabilises: σ ≈ 0.015 (steps 2–10), σ ≈ 0.012 at plateau.

**2. The val→test gap is NOT the agent overfitting.** The hypothesis going in was
that hill-climbing on val inflates the self-report. Wrong: the **baseline already
has a +0.115 gap**, and drift across the whole trajectory is **+0.009 ± 0.086**
(flat). The ~0.12 offset is a property of this split — val is simply easier than
test. The agent's gains genuinely transfer: **test 0.249 → 0.650**. Only *drift*
would have been attributable to the agent, and there is none.

**3. The real problem: 7 of 10 kept experiments are noise.** `_classify` labels at
±3% but `_SUCCESS = {statistically better, better}` — so the **commit boundary is
`improve > 0`**, on a single seed. `STATISTICAL_DELTA` never gates anything. With
σ ≈ 0.015 a truly neutral edit is committed ~50% of the time and the incumbent
only ratchets *up* on val. Replaying the agent's own comparison at n=10:

  | step | Δ vs incumbent | verdict | edit |
  |---:|---:|:--|:--|
  | 2 | +0.328 ± 0.069 | **REAL** | class-weighted BCE |
  | 4 | +0.042 ± 0.009 | **REAL** | deeper/wider MLP |
  | 6 | +0.024 ± 0.006 | **REAL** | epochs 120, batch 64 |
  | 3, 5 | +0.000, +0.017 | noise | standardisation; AdamW+cosine |
  | 7, 8, 9, 10 | −0.008, −0.005, −0.002, −0.000 | noise | four `loss.py` tweaks |

  **All real progress is done by step 6.** Steps 7–10 — the last four experiments,
  every one a loss tweak, each paying an LLM call — have uniformly *negative* point
  estimates. On held-out data the agent **walked away from its best pipeline**:
  test peak **0.6597 @ step 6** vs final **0.6498 @ step 10**.

**4. This falsifies the 2026-07-15 population finding above.** "Loss lever
dominates" rested on agent_01 (architecture-only) scoring 0.407 vs 0.693/0.679.
Those agents started **from the baseline**, where σ = 0.219 and the seed range is
0.046–0.676 — 0.407 sits inside the baseline's own noise. And 0.693 vs 0.679 is
noise outright. **Retracted.** Worse, best-of-N *selects the max of N noisy draws*,
so `parallel.py`'s leaderboard is biased upward by construction: population winners
must be re-measured under fresh seeds before they mean anything.

**Takeaway (and the paper's likely contribution).** Training costs 4 s; an LLM call
costs ~$0.008. Compute is ~free relative to the model, yet the loop spends its
whole evaluation budget on one seed and then makes an irreversible commit decision
from it. *In cheap-task autoresearch the binding constraint is evaluation variance,
not model quality* — so spend compute on variance reduction, not more LLM calls.
Seed-averaging R=5 costs 20 s and cuts SE 0.015 → 0.0067, enough to make steps
7–10 visibly not-improvements. Directly testable at fixed LLM budget (RESEARCH.md
§4 "Reliability").

**Lesson, generalised:** we ran ~10 agent sessions and wrote three findings into
this notebook before ever measuring σ. Two of the three did not survive contact
with it. The noise floor is not an optional refinement — it is the unit every
other claim is denominated in, and it cost $0 and 6 minutes to obtain.

---

### 2026-07-15 — Experimental apparatus: the system can now be studied (RESEARCH.md §6)
- Two sessions converged on the same wall from opposite sides: `replay.py` showed
  the *scores* need error bars; this entry is about the *system* needing them too.
  Every §4 question is "does variant A beat variant B?", and the repo could not
  answer one of them — not for lack of ideas, but of apparatus.
- **What was missing (and is now in):**
  - **Run-config record.** The archive stored what the agent *did*, never what it
    *was*. Added `config.snapshot()` (17 independent variables: proposer, model,
    effort, commit threshold, groups, …) → written into the session header, and
    every record tagged `variant` + `trial`. Sessions are now comparable at all.
  - **Repeat trials.** Added `VARIANT` / `TRIAL` / `AGENT_SEED`. An outcome is a
    distribution, never a point — the same lesson `replay.py` taught for scores,
    applied to the agent's own result.
  - **`sweep.py`** — grid runner: `config × trials` at equal budget, reusing
    `parallel.py`'s worktree isolation. `--vary KEY=v1,v2` over any env knob.
    Reports **mean ± σ over trials** per variant, never a single run.
  - **Cross-run analytics** (`list_sweeps`/`collect_sweep`/`compare_variants`) +
    `/api/sweeps` — so variants are comparable in the dashboard, not just sessions.
  - **`PROPOSER` knob reserved** so the non-LLM control arm (§4 "Controls") is a
    config value, not a fork of the codebase.
- **Smoke (n=1, NOT a result):** `--vary REASONING_EFFORT=low,medium -n 1` ran two
  isolated arms and aggregated cleanly (low 0.7935/$0.0039, medium 0.7588/$0.0055).
  Quoting that as a finding would repeat exactly the error the 2026-07-15 replay
  entry documents — it is an apparatus test, and n=1 means nothing.
- **Learning:** the ordering was wrong for a while. We built the agent, then a
  population, then a dashboard — all before anything could measure whether a
  change to the agent *helped*. Instrument (`replay.py`) and apparatus (`sweep.py`)
  should have preceded the population search, which is currently the least
  interpretable thing in the repo.
- **Biggest remaining debt:** the task is hardcoded into the system
  (`TARGET_GROUPS`, `ALLOWED_FILES`, the lever list inside the system prompt).
  Until a task is *data*, every agent-level claim stays confounded with mammography.

---

### 2026-07-15 — THE CONTROL: the LLM does not beat random search (n=5, 1 task)
- Built `agent/proposers.py`: the loop is held fixed and only the *source of
  proposals* varies (`PROPOSER=llm|random`), so the control is a config value, not
  a fork. `RandomProposer` draws uniformly from a **21-edit menu** spanning the
  same four levers, costs $0, and is reproducible from `AGENT_SEED`. All 21 edits
  were dry-run against the baseline before spending anything (21/21 apply+parse).
- **First real sweep** (`sweep.py --vary PROPOSER=llm,random --trials 5 -n 8
  --base 8e1f1e5`), equal budget of 8 experiments per run:

  | arm | best val/AUPRC (mean ± sd, n=5) | best single trial | cost |
  |-----|-------------------------------|-------------------|------|
  | `llm` (gpt-5-mini) | **0.7643 ± 0.0281** | 0.7833 | **$0.193** |
  | `random` (curated menu) | **0.7571 ± 0.0412** | **0.8159** | **$0.000** |

  diff **+0.0073**, SE 0.0223 → **Welch t = 0.33**. The LLM's advantage is **0.33
  SE — indistinguishable from zero**, and random's *best* trial beat the LLM's
  best trial. At this budget, on this task, **we cannot detect that the LLM's
  choices add anything over uniform sampling of the same action space** — and the
  LLM costs $0.19 to random's $0.00.

- **What this does NOT say.** Three confounds, stated plainly because they bound
  the claim:
  1. **The menu is curated by me.** It contains mostly *good* options (pos_weight,
     focal, Adam, AdamW+cosine, standardisation, quadratic features). So "random"
     is random search over an **expert-designed action space**, not a dumb
     baseline. The honest reading: *the LLM's value is not in choosing among these
     levers* — the knowledge is already in the menu (and, for the LLM arm, in the
     4 levers + wiki we hand it).
  2. **The action space is tiny** — 4 levers, one dominant (loss). RESEARCH.md §4
     predicted this exact outcome ("with only 4 levers and one dominant, random may
     do embarrassingly well"). A bigger space is where directed search should pay.
  3. **These are single-seed val maxima**, so both arms are inflated by the same
     selection-on-noise effect `replay.py` measured (σ ≈ 0.015). The comparison is
     fair (both arms equally inflated) but the *levels* are not trustworthy.
- **Why it matters anyway:** for the first time a claim about the system is
  *falsifiable*, and the first thing it did was falsify the implicit one. Every
  "the agent works" statement in this repo was, until today, confounded with the
  harness: fixed epochs + keep/revert ratchet + a 4-lever space. That scaffolding
  — not the model — appears to be doing most of the work here.
- **Next, in order:** (a) replay both arms' winners under R seeds — if the ranking
  is noise, even this comparison is a mirage; (b) *de-curate* the menu (add junk
  options, wider hyperparameter ranges) — if the LLM wins there, its value is
  *filtering*, which is a real and testable claim; (c) more trials + a task with a
  larger action space.

---

## Open threads
See `RESEARCH.md` for the full backlog. Near-term:
- **Replay the control's winners under seeds** — the control compares single-seed
  val maxima; σ ≈ 0.015 may swamp the +0.007 gap entirely.
- **De-curated / wider menu.** The current control tests "choice within an expert
  menu". The sharper question is whether the LLM *filters* a noisy action space.
- **Fix the loop's decision rule** (from `replay.py`): R-seed averaged evaluation
  + a real commit gate (Δ > k·SE, not Δ > 0). Then R=1 vs R=5 at equal LLM budget.
- **Stopping**: the agent plateaued at step 6 and paid for 4 more experiments. Can
  a variance-aware loop detect the plateau and halt? ($ saved at equal held-out.)
- **Re-measure the population run** under fresh seeds; quantify best-of-N's
  selection bias before scaling to N=8.
- §3 equal-budget best-of-N vs single — now interpretable, since σ is known.
- §5 benchmark vs Sakana AI-Scientist (v2's tree search ≈ our population search).
- **N=1 task is the paper's biggest exposure** — every agent-level claim is
  currently confounded with mammography. `replay.py` is already task-agnostic.
