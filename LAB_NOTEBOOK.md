# Lab notebook — autoresearch agent

Running source book for a future blog/paper. **Append a dated entry after each
significant step** (design decision, run, result, bug, fix). Keep the *headline
results* table current. Raw per-experiment data lives in `history/` (durable,
append-only — see §Data).

---

## Headline results (keep current)

| Task | Model | Baseline | Best (agent) | Cost | Notes |
|------|-------|----------|--------------|------|-------|
| Mammography rare-event (AUPRC, ~2.3% pos) | gpt-5-mini | 0.32 (naive prevalence 0.023) | **0.747** (7-exp demo) / **~0.789** (12-exp run) | $0.024 / ~$0.10 | surgical AST edits; 5–7 kept, 2–5 reverted |

**One-liner for the paper:** a cheap (~$0.02–0.10/run) LLM agent, editing code via
surgical AST ops on a fixed harness, autonomously lifts val/AUPRC on an imbalanced
medical-detection task from a 0.32 baseline to ~0.75–0.79 in ~10 experiments,
keeping good changes as git commits and reverting regressions.

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
- `dashboard.html` — visual snapshot (regenerate with `python dashboard.py`).

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

---

## Open threads
See `RESEARCH.md` for the full backlog. Near-term:
- §3 done (v1) — now run the equal-budget best-of-N vs single comparison.
- §3 extension: breed a 2nd generation from the top-2 pipelines.
- §5 benchmark vs Sakana AI-Scientist (v2's tree search ≈ our population search).
- Point per-run dashboard at a chosen session; add best-of-N over wall-clock.
