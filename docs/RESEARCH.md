# Autoresearch — research notes

Running log of experiment ideas, candidate datasets, and open questions. Treat
this as the project's research backlog.

---

## 1. Candidate datasets (Kaggle)

The current harness is **binary imbalanced classification, metric = AUPRC**.
Everything below either matches that directly or needs a small generalization
(metric/direction are already env-configurable; multiclass needs a head + metric
change). All are lightweight tabular → fast CPU loops, same as mammography.

### Recent — Kaggle Playground Series, Season 6 (2026, monthly)

Synthetic tabular data *generated from a real dataset*, so features are named and
realistic but the test labels stay private. CSV `train.csv` + `test.csv`
downloadable; a few-week window each.

| Comp | Task | Fit | Notes |
|------|------|-----|-------|
| **S6E2 – Predicting Heart Disease** | binary classification | ★★★ strong | medical theme like mammography; likely mild imbalance |
| **S6E3 – Predict Customer Churn** | binary classification | ★★★ strong | churn = minority class → naturally imbalanced |
| **S6E7** (Jul 2026) | classification, **balanced accuracy** | ★★ | needs metric swap to balanced-acc |
| **S6E6 – Predicting Stellar Class** | **multiclass**, balanced accuracy | ★★ | needs multiclass head + macro metric |
| S6E1 / E4 / E5 | student scores / irrigation / F1 pit stops | ★ | regression or niche; more generalization |

### Evergreen — strong imbalanced / medical fits (open, always submittable)

| Dataset | Task | Why it fits |
|---------|------|-------------|
| **ICR – Identifying Age-Related Conditions** (2023) | binary, anonymized health, **imbalanced** | closest analog to our task; metric = balanced log loss. NB: was a *code* competition — see §2 |
| **Playground S3/S4/S5** back-catalog | binary/multiclass/regression | dozens of finished tabular comps, all downloadable; good for batch benchmarking |
| Credit-card fraud, Give Me Some Credit | binary, extreme imbalance | classic rare-event, maps to our loss/resampling levers |

**Recommendation for the first cross-dataset run:** S6E3 (churn) or S6E2 (heart
disease) — closest to the current binary/imbalanced setup, minimal code change.

---

## 2. Do they publish the closed test set after a competition?

**Short answer: usually no — the private test *labels* are not released.** But you
can still get a real held-out score two ways:

1. **Late submission (the main mechanism).** For most competitions the scoring
   engine stays open after close: you upload predictions and get a **private
   leaderboard score** against the hidden test — it just doesn't count for
   prizes/ranking. This is how you'd measure "did the agent's idea actually
   generalize" without ever seeing the labels.
   - Caveat: **code competitions** (submit a notebook, e.g. ICR) may *not* accept
     late submissions the same way — verify per competition before relying on it.
2. **Reconstruction.** Playground sets are synthesized from a known public
   dataset; the original (with labels) is often findable, giving an approximate
   ground truth. Not the official test, but useful.
3. **Occasional full release.** A minority of comps publish `test_labels` after
   close (e.g. Jigsaw Toxic Comment). The exception, not the rule.

**Implication for this project (important):** the agent's *optimization signal*
should be a **local cross-validation / held-out split it controls** (like our
fixed val split now) — never the Kaggle test. The hidden test is only a *final,
occasional* check via late submission. This also guards against the agent
overfitting the leaderboard.

**Kaggle wiring (future, small):** `kaggle competitions download -c <slug>` to
fetch data; a `KaggleTracker` (mirrors our `Tracker` protocol) could optionally
`kaggle competitions submit` and read back the LB score. Keep it behind the same
pluggable interface as ClearML.

---

## 3. Experiment: N parallel agents per task (isolated worktrees)

> **Status: implemented (v1) in `parallel.py`.** N agents, git worktrees,
> process-per-agent, per-agent durable archives, live leaderboard, and a
> a Population view in the dashboard app (per-agent curves + best-of-N).
>
> ⚠️ **The first result from it is RETRACTED.** The 3×2 run was read as "the loss
> lever dominates" (agent_01, architecture-only, scored 0.407 vs 0.693/0.679).
> `replay.py` later measured the baseline at **0.363 ± 0.219** — those agents
> started there, so 0.407 is inside the baseline's own noise, and 0.693 vs 0.679
> is noise outright. See `LAB_NOTEBOOK.md` 2026-07-15.
>
> **Methodological consequence for this whole section:** best-of-N **selects the
> max of N noisy draws**, so the leaderboard's winning score is biased upward by
> construction, and the bias *grows with N* — which is exactly the direction that
> would fake a "breadth beats depth" result. Any best-of-N comparison must
> re-measure the winner under fresh seeds (`replay.py --commits <sha>`) and report
> *that*, never the selection-time score. Still TODO: the equal-budget best-of-N
> vs single comparison, and breeding a 2nd generation from the top-2.

**Idea:** spawn ~10 agents on the *same* task at once, each in its own git
worktree, and compare. Turns the single greedy hill-climb into a **population
search**.

**Design**
- Each agent = independent `run_loop` in its own worktree + its own `logs.md`
  (no shared state → no write races). Same baseline, same fixed val split.
- Vary the seed of exploration: different `REASONING_EFFORT`, temperature/model,
  or a different starting hint per agent, so they don't all propose the same edit.
- After K iterations, collect each population member's best pipeline; pick the
  global best, or **breed**: seed a second round from the top-2 pipelines.

**What we measure**
- Best-of-N vs single-agent best at equal *total* token budget (is parallel
  breadth worth more than depth?).
- Diversity of proposed ideas (do agents converge or explore different levers?).
- Variance across runs (how much is the outcome luck vs. signal?).
- Cost/perf Pareto: score vs total $ for N ∈ {1, 3, 10}.

**Hypotheses**
- H1: best-of-10 beats 1×(10 iters) at the same budget on hard tasks (rugged
  landscape → breadth wins).
- H2: returns diminish fast (best-of-3 ≈ best-of-10) on easy tasks.
- H3: breeding from top-2 beats flat best-of-N.

**Infra notes:** the harness already has worktree isolation available; each agent
needs a distinct `MODEL_DIR` worktree + `RUNS_DIR`. Training is CPU-cheap (~4s),
so 10× parallel is feasible on this 22-core box. Watch the LLM rate limit, not CPU.

---

## 4. Broader research questions

Grouped by what they probe. Each is a runnable experiment with this harness —
**once §6 (experimental apparatus) exists**; today the system records what it did
but not what it *was*, so no A/B below is actually decidable yet.

**Evaluation variance & the decision rule** ← *highest priority; σ measured 2026-07-15*

`replay.py` established the numbers these questions turn on: σ ≈ 0.015 at plateau
(σ = 0.219 at the baseline), the commit boundary is `improve > 0` on a **single
seed**, and 7 of 10 kept experiments were noise. The governing asymmetry: a
training run costs **4 s of CPU**, an LLM call costs **~$0.008** — compute is
essentially free relative to the model, yet 100% of the evaluation budget goes to
one seed and an irreversible commit decision is made from it.

- **R-seed averaged evaluation.** R=1 vs R=5 at *equal LLM budget* (same number of
  experiments) → held-out score. R=5 costs +16 s/experiment and cuts SE 0.015 →
  0.0067. H1: R=5 wins, because the ratchet stops locking in noise.
- **A real commit gate.** Commit iff Δ > k·SE (k ≈ 2) vs the current Δ > 0. H2:
  the zero-margin ratchet is *why* steps 7–10 happened; a gate makes them reverts.
  Note this also converts `STATISTICAL_DELTA` from a cosmetic label into a knob.
- **Knowing when to stop.** The agent plateaued at step 6 and paid for 4 more
  experiments (~40% of spend, ≈0 gain, slightly *worse* held-out). Can a
  variance-aware loop detect its own plateau and halt? Measure: $ saved at equal
  held-out score. This is a capability claim, not a hyperparameter.
- **Does the LLM know it's noise?** The archive stores `reasoning` next to
  outcome. Do the agent's stated hypotheses predict REAL vs noise better than
  chance? If not, that bounds what better prompting can ever buy.
- **Cost of the noise floor itself.** σ scales with val-set positives (~51 here).
  Bigger val / k-fold CV as the agent's signal vs seed-averaging: which buys more
  decision quality per CPU-second?

**Controls (without these, "the agent works" is unfalsifiable)**
- **Random-edit arm.** Pick a random lever, apply a random edit from a fixed menu,
  same budget. With only 4 levers and one dominant (loss), random may do
  embarrassingly well — and 3 real improvements in 10 tries is a low bar to beat.
- **Ceiling reference.** A tuned `HistGradientBoostingClassifier` on the same
  split. We have a floor (0.363 val / 0.249 test) but **no ceiling**, so 0.774 /
  0.650 has no scale. Needed to normalise across tasks:
  `headroom captured = (agent_test − baseline_test) / (ceiling_test − baseline_test)`
  — held-out, task-agnostic, and divisible by $ for the cost claim.

**Search strategy**
- Greedy hill-climb (now) vs population (§3) vs MCTS/beam over the edit tree?
- Does letting the agent revisit reverted ideas later (with new context) help?
- Best-of-N *edits per turn* (propose 3, dry-run all, keep the best) vs 1.

**Context & memory**
- How much history does the agent actually need? Ablate `logs.md` window size.
- Does showing *failed* traceback patterns reduce repeat crashes over time?
- Fresh-context-per-turn (now) vs accumulated conversation — score & cost.

**Model & cost**
- Model tier vs outcome: gpt-5-mini vs gpt-5 vs a cheap open model — score per $.
- Reasoning-effort sweep (minimal→high): where's the cost/quality knee?
- Can a cheap model *generate* edits and an expensive one only *judge* them?

**Edit mechanism**
- Surgical AST edits (now) vs whole-file rewrites: token cost & success rate.
- Does constraining to smaller edits improve or hurt exploration?

**Task generalization**
- One agent, many datasets (§1): does it find dataset-specific tricks or apply a
  generic recipe (scale → weighted loss → Adam → capacity)?
- Transfer: seed a new task with `logs.md` distilled from a prior task.
- Regression / multiclass / true rare-disease sets — does the loop hold up?

**Reliability & honesty**
- ~~Overfitting the val split~~ **— answered (2026-07-15), and the answer was no.**
  The val→test gap is ~0.12 but it is a **fixed offset of this split**, present at
  the baseline: drift over the trajectory = +0.009 ± 0.086. The agent's gains
  transfer (test 0.249 → 0.650). Still open on *other* tasks, and still open for
  best-of-N, where selection pressure is much stronger than in a single chain.
- **How often does "better" not generalize? — 7 of 10 kept experiments.** Not via
  val-overfitting, though: via *noise*. See the evaluation-variance group above.
- ~~Reproducibility: variance across seeds~~ **— measured.** σ ≈ 0.015 at plateau;
  σ = 0.219 at the baseline (range 0.046–0.676 over 10 seeds). Any claim on this
  task must carry error bars this wide. `replay.py` is the instrument.
- Reward hacking: does the agent ever game the metric (e.g. degenerate configs)?
  **A live surface exists:** `seed` lives in `config.yaml`, an agent-editable seam
  — tuning it is pure metric-hacking with zero generalization. Never observed yet
  (checked the archive + git history); `SEED` env now overrides it during replay,
  but the agent can still edit it during a normal run. Worth leaving in place and
  watching, since "does it find the hack?" is itself a finding.

**Autonomy**
- Let the agent also edit its own *search policy* / prompt (meta-level)?
- Auto-generate the starting baseline files from just a task + dataset (the
  "generates its own starter code" item in the original design notes).

---

## 5. Benchmark vs. other AI research agents

Compare our cheap, narrow agent against full autonomous-research systems —
primarily **Sakana AI's AI-Scientist (v1 & v2)**.

**What they are**
- **AI-Scientist v1** — end-to-end pipeline: idea → lit-search/novelty → code +
  experiments → figures → **full LaTeX paper** → LLM peer review. *Template-driven*
  (NanoGPT, 2D diffusion, grokking). ~**$15/paper**, best with Claude 3.5 Sonnet.
- **AI-Scientist v2** — template-free, **agentic best-first tree search** with an
  experiment-manager agent + parallel workers; more open-ended, lower success rate.
  ~**$15–20 + ~$5 writeup** per run. Produced the first fully-AI paper accepted at
  a workshop (peer-reviewed).

**Why it's not apples-to-apples (and how to make it fair)**
Ours optimizes *one metric on a fixed task via surgical edits* at ~**$0.02/run** —
it doesn't do ideation, lit-review, or paper-writing. So compare only the shared
sub-task: **the experimentation / metric-optimization loop.**
- Strip their pipeline to the experiment stage; give both the *same* task,
  baseline code, metric, and **compute/token budget**.
- Run on a task both can handle (a tabular set from §1, or port one of their
  templates into our harness).

**Metrics:** best metric reached · **score per $** (ours is 100–1000× cheaper, so
cost-normalize) · tokens · wall-clock · # experiments · success rate ·
generalization gap (val vs held-out, §2).

**Hypotheses**
- H1: on narrow metric-optimization, our cheap loop reaches a comparable metric at
  a tiny fraction of the cost (surgical edits + fixed harness beat full-pipeline
  overhead).
- H2: their systems win on *open-ended* tasks / breadth of ideas, not on cost.
- H3: v2's tree search ≈ our §3 population search — **their v2 is essentially the
  parallel-worker version of §3**, so §3 and §5 inform each other.

**Caveats:** their templates assume GPU ML-research setups; adapting to a cheap
tabular task needs a shim. Both are open-source but run LLM-generated code — use
their containerization. Check licenses before reusing code.

## 6. Experimental apparatus — what must exist before §4 is runnable

*Object of study = the agent. The dataset is only the instrument.* Every question
in §4 has the form **"does system variant A beat variant B?"** — and the system
currently cannot answer *any* of them, not for lack of ideas but for lack of
apparatus. The gaps, in dependency order:

1. **Run-config record.** The archive records what the agent *did*, never what the
   agent *was* (model, effort, edit mode, context window, commit gate, R, search
   policy, seed). Two sessions are therefore **not comparable**, so no §4 question
   is decidable today. → snapshot the resolved config into the session header;
   tag every record with a `variant` label.
2. **Repeat trials.** One run per config. LLM sampling is stochastic, so a single
   trajectory is an anecdote — the same σ lesson §4 applies to *scores* applies to
   the agent's own *outcome*. → N trials per variant; report distributions, never
   points.
3. **Baseline controls.** No non-LLM arm and no ceiling (both named in §4). Both
   need the same loop with a **swappable proposer**. → a `Proposer` interface
   (`llm` | `random` | `scripted`), so the random-edit arm is a config value rather
   than a fork of the codebase.
4. **Sweep harness.** `parallel.py` runs N agents at **one** config. → a grid
   runner: `config × trials` at equal budget, reusing the existing worktree
   isolation. (`parallel.py` is ~80% of this already.)
5. **Cross-run analytics.** `agent/analytics.py` is per-session. → aggregate and
   compare *variants* (mean ± σ of held-out score, $/experiment, headroom
   captured).

Structural blockers behind those:

6. **The task is hardcoded into the system** — `TARGET_GROUPS`, `ALLOWED_FILES`,
   the lever list baked into the system prompt, the wiki path. §4's "task
   generalization", §1's Kaggle sets, and §5's benchmark **cannot start** until a
   task is *data* (a task spec: working dir, levers, metric+direction, contracts,
   briefing) rather than code. Biggest architectural debt in the repo.
7. **Integrity guard.** Nothing verifies the fixed harness is untouched. §4 already
   names a live hacking surface (`seed` inside the agent-editable `config.yaml`).
   → hash-guard `run.py`/`data.py` each iteration + assert held-out signal never
   enters the agent's context. Makes "did it cheat?" *checkable* rather than
   assumed.

**Order:** (1)+(2) unlock everything · (3) makes the claims falsifiable · (4)+(5)
make sweeps cheap · (6) unlocks §1/§4-generalization/§5 · (7) protects the record.

## 7. Logistics / caveats
- Local CV is the optimization signal; Kaggle test is a rare final check (§2).
- Verify late-submission availability per competition (code comps differ).
- Kaggle API needs `~/.kaggle/kaggle.json` creds; respect each comp's rules on
  automated/agent use before publishing anything.

---

### Sources
- [Kaggle Playground Series](https://www.kaggle.com/competitions/playground-series) · [S6E4 example](https://www.kaggle.com/competitions/playground-series-s6e4)
- [About the Tabular Playground Series (synthetic data, hidden labels)](https://www.kaggle.com/competitions/playground-series-s5e11/overview/about-the-tabular-playground-series)
- [Kaggle competitions docs (public/private LB)](https://www.kaggle.com/docs/competitions)
- Late-submission discussions: [scoring engine stays open](https://www.kaggle.com/discussions/general/30899) · [late submission ranking](https://www.kaggle.com/general/47251)
- [Data sets from closed competitions (labels generally not released)](https://www.kaggle.com/general/4350)
- [Sakana AI-Scientist (v1)](https://github.com/SakanaAI/AI-Scientist) · [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)
