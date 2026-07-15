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
> `dashboard.py --parallel` view (per-agent curves + best-of-N). First run
> (3×2 from baseline) already showed the loss lever dominating on this
> imbalanced task — see `LAB_NOTEBOOK.md`. Still TODO: the equal-budget
> best-of-N vs single comparison, and breeding a 2nd generation from the top-2.

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

Grouped by what they probe. Each is a runnable experiment with this harness.

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
- Overfitting the val split: gap between agent's val score and a held-out test
  (or Kaggle late-submission) score. How often does "better" not generalize?
- Reward hacking: does the agent ever game the metric (e.g. degenerate configs)?
- Reproducibility: variance of the same task across seeds.

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

## 6. Logistics / caveats
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
