import { createContext, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { Experiment, AgentData } from "./types";

/* ---------- theme-aware color reader ---------- */
export function cvar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
const fmt = (v: number | null | undefined, d = 4) => (v == null ? "—" : (+v).toFixed(d));

/* ---------- responsive width hook ---------- */
function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(760);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((e) => setW(e[0].contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/* ---------- shared tooltip ---------- */
const TipCtx = createContext<{ show: (h: string, x: number, y: number) => void; hide: () => void }>({
  show: () => {}, hide: () => {},
});
export function TooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<{ h: string; x: number; y: number } | null>(null);
  const show = (h: string, x: number, y: number) => setTip({ h, x, y });
  const hide = () => setTip(null);
  return (
    <TipCtx.Provider value={{ show, hide }}>
      {children}
      {tip && (
        <div
          className="tip"
          style={{ left: Math.min(tip.x + 14, window.innerWidth - 270), top: Math.max(tip.y - 60, 8) }}
          dangerouslySetInnerHTML={{ __html: tip.h }}
        />
      )}
    </TipCtx.Provider>
  );
}
const useTip = () => useContext(TipCtx);

/* ---------- scales ---------- */
function bounds(vals: number[], pad = 0.08) {
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const r = mx - mn || 1;
  return { mn: mn - r * pad, mx: mx + r * pad };
}

/* =========================================================================
   Solo: optimization curve (best-so-far + per-experiment points)
   ========================================================================= */
export function OptimizationChart({ data }: { data: { experiments: Experiment[]; baseline: number | null; metric: string } }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const tip = useTip();
  const E = data.experiments.filter((e) => e.score != null);
  return (
    <div ref={ref}>
      {E.length === 0 ? (
        <p className="empty">No scored experiments yet.</p>
      ) : (
        (() => {
          const H = 300, m = { t: 14, r: 18, b: 30, l: 46 };
          const iw = W - m.l - m.r, ih = H - m.t - m.b;
          const xs = E.map((e) => e.n);
          const vals = E.flatMap((e) => [e.score!, e.best_so_far!]);
          if (data.baseline != null) vals.push(data.baseline);
          const { mn, mx } = bounds(vals);
          const X = (n: number) => m.l + (xs.length < 2 ? iw / 2 : ((n - xs[0]) / (xs[xs.length - 1] - xs[0])) * iw);
          const Y = (v: number) => m.t + (1 - (v - mn) / (mx - mn || 1)) * ih;
          const acc = cvar("--accent");
          const bestPt = E.reduce((a, b) => (b.best_so_far! >= a.best_so_far! ? b : a), E[0]);
          const line = `M ${X(E[0].n)} ${Y(E[0].best_so_far!)} ` + E.map((e) => `L ${X(e.n)} ${Y(e.best_so_far!)}`).join(" ");
          return (
            <svg viewBox={`0 0 ${W} ${H}`} role="img">
              <g className="axis">
                {[0, 1, 2, 3, 4].map((i) => {
                  const v = mn + ((mx - mn) * i) / 4, y = Y(v);
                  return (
                    <g key={i}>
                      <line x1={m.l} y1={y} x2={W - m.r} y2={y} />
                      <text x={m.l - 8} y={y + 3} textAnchor="end">{v.toFixed(2)}</text>
                    </g>
                  );
                })}
                {E.map((e) => (
                  <text key={e.n} x={X(e.n)} y={H - 10} textAnchor="middle">#{e.n}</text>
                ))}
              </g>
              {data.baseline != null && (
                <line x1={m.l} y1={Y(data.baseline)} x2={W - m.r} y2={Y(data.baseline)}
                  stroke={cvar("--faint")} strokeWidth={1.5} strokeDasharray="5 4" opacity={0.7} />
              )}
              <path d={`${line} L ${X(E[E.length - 1].n)} ${Y(mn)} L ${X(E[0].n)} ${Y(mn)} Z`} fill={cvar("--accent-soft")} />
              <path d={line} fill="none" stroke={acc} strokeWidth={2.5} strokeLinejoin="round" strokeLinecap="round" />
              {E.map((e) => (
                <circle key={e.n} cx={X(e.n)} cy={Y(e.score!)} r={5} fill={e.kept ? cvar("--good") : cvar("--crit")}
                  stroke={cvar("--surface")} strokeWidth={2} style={{ cursor: "pointer" }}
                  onMouseMove={(ev) => tip.show(
                    `<b>#${e.n} · ${e.target}</b><br><span class="k">score</span> ${fmt(e.score)} · ${e.status}<br><span class="k">best</span> ${fmt(e.best_so_far)}`,
                    ev.clientX, ev.clientY)}
                  onMouseLeave={tip.hide} />
              ))}
              <circle cx={X(bestPt.n)} cy={Y(bestPt.best_so_far!)} r={5.5} fill={acc} stroke={cvar("--surface")} strokeWidth={2.5} />
              <text x={X(bestPt.n)} y={Y(bestPt.best_so_far!) - 12} textAnchor="middle" fill={cvar("--ink")} fontSize={12} fontWeight={700} className="mono">
                {fmt(bestPt.best_so_far)}
              </text>
            </svg>
          );
        })()
      )}
    </div>
  );
}

/* ---------- learning curve (area line) ---------- */
export function MiniLine({ series }: { series: [number, number][] }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  return (
    <div ref={ref}>
      {series.length < 2 ? (
        <p className="empty">No learning curve.</p>
      ) : (
        (() => {
          const H = 200, m = { t: 12, r: 14, b: 26, l: 44 };
          const iw = W - m.l - m.r, ih = H - m.t - m.b;
          const xs = series.map((p) => p[0]), ys = series.map((p) => p[1]);
          const xmn = Math.min(...xs), xmx = Math.max(...xs);
          const { mn, mx } = bounds(ys, 0.06);
          const X = (x: number) => m.l + ((x - xmn) / (xmx - xmn || 1)) * iw;
          const Y = (v: number) => m.t + (1 - (v - mn) / (mx - mn || 1)) * ih;
          const acc = cvar("--accent");
          const line = `M ${X(xs[0])} ${Y(ys[0])} ` + series.map((p) => `L ${X(p[0])} ${Y(p[1])}`).join(" ");
          return (
            <svg viewBox={`0 0 ${W} ${H}`}>
              <g className="axis">
                {[0, 1, 2, 3].map((i) => {
                  const v = mn + ((mx - mn) * i) / 3, y = Y(v);
                  return <g key={i}><line x1={m.l} y1={y} x2={W - m.r} y2={y} /><text x={m.l - 8} y={y + 3} textAnchor="end">{v.toFixed(2)}</text></g>;
                })}
                {[xmn, Math.round((xmn + xmx) / 2), xmx].map((x, i) => (
                  <text key={i} x={X(x)} y={H - 9} textAnchor="middle">ep {x}</text>
                ))}
              </g>
              <path d={`${line} L ${X(xmx)} ${Y(mn)} L ${X(xmn)} ${Y(mn)} Z`} fill={cvar("--accent-soft")} />
              <path d={line} fill="none" stroke={acc} strokeWidth={2.5} strokeLinejoin="round" />
            </svg>
          );
        })()
      )}
    </div>
  );
}

/* ---------- cost bars ---------- */
export function CostBars({ experiments }: { experiments: Experiment[] }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const tip = useTip();
  const E = experiments.filter((e) => e.cost != null);
  return (
    <div ref={ref}>
      {E.length === 0 ? <p className="empty">No cost data.</p> : (() => {
        const H = 200, m = { t: 12, r: 14, b: 26, l: 46 };
        const iw = W - m.l - m.r, ih = H - m.t - m.b;
        const mx = Math.max(...E.map((e) => e.cost!)) || 1;
        const bw = Math.min(38, (iw / E.length) * 0.62);
        const X = (i: number) => m.l + (E.length < 2 ? iw / 2 : (i / (E.length - 1)) * (iw - bw) + bw / 2);
        const Y = (v: number) => m.t + (1 - v / mx) * ih;
        return (
          <svg viewBox={`0 0 ${W} ${H}`}>
            <g className="axis">
              {[0, 1, 2, 3].map((i) => {
                const v = (mx * i) / 3, y = Y(v);
                return <g key={i}><line x1={m.l} y1={y} x2={W - m.r} y2={y} /><text x={m.l - 8} y={y + 3} textAnchor="end">${v.toFixed(4)}</text></g>;
              })}
            </g>
            {E.map((e, i) => {
              const x = X(i) - bw / 2, y = Y(e.cost!), h = m.t + ih - y;
              return (
                <g key={e.n}>
                  <rect x={x} y={y} width={bw} height={Math.max(h, 1)} rx={4} fill={cvar("--accent")} opacity={0.85}
                    style={{ cursor: "pointer" }}
                    onMouseMove={(ev) => tip.show(`<b>#${e.n} · ${e.target}</b><br><span class="k">cost</span> $${fmt(e.cost, 4)} · ${e.tokens} tok`, ev.clientX, ev.clientY)}
                    onMouseLeave={tip.hide} />
                  <text x={X(i)} y={H - 9} textAnchor="middle" fill={cvar("--faint")} fontSize={10}>#{e.n}</text>
                </g>
              );
            })}
          </svg>
        );
      })()}
    </div>
  );
}

/* =========================================================================
   Population: per-agent best-so-far + best-of-N
   ========================================================================= */
export function MultiAgentChart({ agents, population }: { agents: AgentData[]; population: [number, number][] }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const tip = useTip();
  const A = agents.filter((a) => a.points.length);
  return (
    <div ref={ref}>
      {A.length === 0 ? <p className="empty">No scored experiments yet.</p> : (() => {
        const H = 320, m = { t: 14, r: 20, b: 30, l: 46 };
        const iw = W - m.l - m.r, ih = H - m.t - m.b;
        const allpts = A.flatMap((a) => a.points).concat(population);
        const xmax = Math.max(...allpts.map((p) => p[0]), 1);
        const { mn, mx } = bounds(allpts.map((p) => p[1]));
        const X = (k: number) => m.l + ((k - 1) / (xmax - 1 || 1)) * iw;
        const Y = (v: number) => m.t + (1 - (v - mn) / (mx - mn || 1)) * ih;
        return (
          <svg viewBox={`0 0 ${W} ${H}`}>
            <g className="axis">
              {[0, 1, 2, 3, 4].map((i) => {
                const v = mn + ((mx - mn) * i) / 4, y = Y(v);
                return <g key={i}><line x1={m.l} y1={y} x2={W - m.r} y2={y} /><text x={m.l - 8} y={y + 3} textAnchor="end">{v.toFixed(2)}</text></g>;
              })}
              {Array.from({ length: xmax }, (_, i) => i + 1).map((k) => (
                <text key={k} x={X(k)} y={H - 10} textAnchor="middle">{k}</text>
              ))}
            </g>
            {A.map((a) => {
              const line = `M ${X(a.points[0][0])} ${Y(a.points[0][1])} ` + a.points.map((p) => `L ${X(p[0])} ${Y(p[1])}`).join(" ");
              const last = a.points[a.points.length - 1];
              return (
                <g key={a.label}>
                  <path d={line} fill="none" stroke={a.color} strokeWidth={1.8} strokeLinejoin="round" opacity={0.9} />
                  {a.points.map((p, i) => (
                    <circle key={i} cx={X(p[0])} cy={Y(p[1])} r={3.2} fill={a.color} style={{ cursor: "pointer" }}
                      onMouseMove={(ev) => tip.show(`<b>${a.label}</b> · exp ${p[0]}<br>best ${fmt(p[1])}`, ev.clientX, ev.clientY)}
                      onMouseLeave={tip.hide} />
                  ))}
                  <text x={X(last[0]) + 6} y={Y(last[1]) + 3} fill={a.color} fontSize={10} fontWeight={700}>{a.label.replace("agent_", "a")}</text>
                </g>
              );
            })}
            {population.length > 0 && (
              <path d={`M ${X(population[0][0])} ${Y(population[0][1])} ` + population.map((p) => `L ${X(p[0])} ${Y(p[1])}`).join(" ")}
                fill="none" stroke={cvar("--accent")} strokeWidth={3.2} strokeLinejoin="round" />
            )}
          </svg>
        );
      })()}
    </div>
  );
}
