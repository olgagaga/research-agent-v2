import type { SoloData, ParallelData } from "./types";
import { OptimizationChart, MiniLine, CostBars, MultiAgentChart } from "./charts";

const fmt = (v: number | null | undefined, d = 4) => (v == null ? "—" : (+v).toFixed(d));

const STATUS_CLS: Record<string, string> = {
  "statistically better": "good", better: "good", lower: "warn",
  "statistically lower": "warn", crushed: "crit", crashed: "crit",
  rejected: "warn", no_run_id: "warn", internal_error: "crit",
};

function Kpi({ label, val, note }: { label: string; val: string | number; note: string }) {
  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className="val mono">{val}</div>
      <div className="note">{note}</div>
    </div>
  );
}

/* ============================ SOLO ============================ */
export function SoloView({ data }: { data: SoloData }) {
  const s = data.summary, m = data.meta;
  const delta = s.best != null && s.first_score != null ? s.best - s.first_score : null;
  const pct = delta != null && s.first_score ? (delta / s.first_score) * 100 : null;
  return (
    <>
      <section className="kpis">
        <Kpi label={`Best ${m.metric}`} val={fmt(s.best)} note="higher is better" />
        <Kpi label="Experiments" val={s.count} note={s.attempts > s.count ? `${s.attempts} attempts (incl. rejected/crashed)` : "trainings completed"} />
        <Kpi label="Kept / reverted" val={`${s.kept} / ${s.reverted}`} note="committed vs discarded" />
        <Kpi label="Total spend" val={`$${fmt(s.total_cost, 4)}`} note="LLM cost (all calls)" />
        <Kpi label="Success rate" val={s.count ? `${Math.round((s.kept / s.count) * 100)}%` : "—"} note="kept of completed" />
      </section>

      <section className="card hero">
        <h2>Optimization progress</h2>
        <p className="cardsub">Best <span className="mono">{m.metric}</span> after each experiment · points are individual scores (kept / reverted)</p>
        <div className="hero-figure mono">{fmt(s.best)}</div>
        {delta != null && (
          <div className="delta mono">
            <span style={{ color: "var(--good)" }}>▲ {fmt(delta)}</span>{" "}
            <span style={{ color: "var(--muted)" }}>vs first ({fmt(s.first_score)}) · +{pct!.toFixed(0)}%</span>
          </div>
        )}
        <OptimizationChart data={{ experiments: data.experiments, baseline: data.baseline, metric: m.metric }} />
        <div className="legend">
          <span className="k"><span className="swatch" style={{ background: "var(--accent)" }} />Best so far</span>
          <span className="k"><span className="dot" style={{ background: "var(--good)" }} />Kept</span>
          <span className="k"><span className="dot" style={{ background: "var(--crit)" }} />Reverted</span>
          <span className="k"><span className="swatch" style={{ borderTop: "2px dashed var(--faint)" }} />Naive baseline</span>
        </div>
      </section>

      <section className="row2">
        <div className="card">
          <h2>Spend per experiment</h2>
          <p className="cardsub">LLM cost each turn · cumulative <span className="mono">${fmt(s.total_cost, 4)}</span></p>
          <CostBars experiments={data.experiments} />
        </div>
        <div className="card">
          <h2>Best run · learning curve</h2>
          <p className="cardsub"><span className="mono">{m.metric}</span> across epochs (run <span className="mono">{data.best_run_id || "—"}</span>)</p>
          <MiniLine series={data.best_series} />
        </div>
      </section>

      <section className="card">
        <h2>Experiment log</h2>
        <p className="cardsub">Every experiment in order · source: {m.source}{m.session !== "—" ? ` · session ${m.session}` : ""}</p>
        <div className="tablewrap">
          <table>
            <thead><tr><th>#</th><th>Lever</th><th>Experiment</th><th className="num">Score</th><th>Outcome</th><th className="num">Tokens</th><th className="num">Cost</th></tr></thead>
            <tbody>
              {data.experiments.map((e) => (
                <tr key={e.n}>
                  <td className="mono">{e.n}</td>
                  <td><span className="chip mono">{e.target}</span></td>
                  <td className="desc">{e.description}</td>
                  <td className="num mono">{fmt(e.score)}</td>
                  <td><span className={`pill ${STATUS_CLS[e.status] || "warn"}`}>{e.status}</span></td>
                  <td className="num mono" style={{ color: "var(--muted)" }}>{e.tokens || "—"}</td>
                  <td className="num mono">{e.cost != null ? `$${fmt(e.cost, 4)}` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

/* ============================ POPULATION ============================ */
export function PopulationView({ data }: { data: ParallelData }) {
  const m = data.meta;
  const ranked = [...data.agents].sort((a, b) => (b.best ?? -Infinity) - (a.best ?? -Infinity));
  const winner = ranked[0];
  const done = data.agents.reduce((s, a) => s + a.done, 0);
  const kept = data.agents.reduce((s, a) => s + a.kept, 0);
  return (
    <>
      <section className="kpis">
        <Kpi label={`Best ${m.metric}`} val={fmt(m.best)} note="across all agents" />
        <Kpi label="Agents" val={m.n_agents} note="parallel worktrees" />
        <Kpi label="Experiments" val={done} note={`${kept} kept total`} />
        <Kpi label="Total spend" val={`$${fmt(m.total_cost, 4)}`} note="all agents" />
      </section>

      <section className="card hero">
        <h2>Optimization progress — per agent + best-of-N</h2>
        <p className="cardsub">Each line = one agent's best <span className="mono">{m.metric}</span>; the bold line is the population best</p>
        <div className="hero-figure mono">{fmt(m.best)}</div>
        {winner && winner.best != null && (
          <div className="delta mono"><span className="winner">▲ {winner.label}</span> <span style={{ color: "var(--muted)" }}>leads at {fmt(winner.best)}</span></div>
        )}
        <MultiAgentChart agents={data.agents} population={data.population} />
        <div className="legend">
          {data.agents.map((a) => (
            <span className="k" key={a.label}><span className="swatch-sq" style={{ background: a.color }} />{a.label} ({fmt(a.best)})</span>
          ))}
          <span className="k"><span className="swatch" style={{ width: 16, height: 3, background: "var(--accent)" }} />best-of-N</span>
        </div>
      </section>

      <section className="card">
        <h2>Leaderboard</h2>
        <p className="cardsub">Ranked by best metric · run <span className="mono">{m.run}</span></p>
        <div className="tablewrap">
          <table>
            <thead><tr><th>Agent</th><th>Effort</th><th>Focus (hint)</th><th className="num">Best</th><th className="num">Done</th><th className="num">Kept</th><th className="num">Attempts</th><th className="num">Cost</th></tr></thead>
            <tbody>
              {ranked.map((a, i) => (
                <tr key={a.label}>
                  <td><div className="agentcell"><span className="swatch-sq" style={{ background: a.color }} /><span className={`mono ${i === 0 ? "winner" : ""}`}>{a.label}</span></div></td>
                  <td><span className="chip mono">{a.effort || "—"}</span></td>
                  <td className="desc">{a.hint || "—"}</td>
                  <td className={`num mono ${i === 0 ? "winner" : ""}`}>{fmt(a.best)}</td>
                  <td className="num mono">{a.done}</td>
                  <td className="num mono">{a.kept}</td>
                  <td className="num mono" style={{ color: "var(--muted)" }}>{a.attempts}</td>
                  <td className="num mono">${fmt(a.cost, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
