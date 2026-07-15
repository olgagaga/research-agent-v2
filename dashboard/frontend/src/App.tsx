import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { SoloData, SessionInfo, ParallelData, ParallelRunInfo } from "./types";
import { SoloView, PopulationView } from "./views";
import { TooltipProvider } from "./charts";

type Tab = "solo" | "population";
const POLL_MS = 5000;

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() =>
    window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  useEffect(() => { document.documentElement.setAttribute("data-theme", theme); }, [theme]);
  return { theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) };
}

export default function App() {
  const { theme, toggle } = useTheme();
  const [tab, setTab] = useState<Tab>("solo");
  const [live, setLive] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [cfg, setCfg] = useState<Record<string, string> | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [session, setSession] = useState<string>("");
  const [solo, setSolo] = useState<SoloData | null>(null);

  const [runs, setRuns] = useState<ParallelRunInfo[]>([]);
  const [run, setRun] = useState<string>("");
  const [pop, setPop] = useState<ParallelData | null>(null);

  const refresh = useCallback(async () => {
    try {
      if (tab === "solo") {
        const [ss, d] = await Promise.all([api.sessions(), api.solo(session || undefined)]);
        setSessions(ss);
        setSolo(d);
      } else {
        const rs = await api.parallelRuns();
        setRuns(rs);
        const name = run || rs[0]?.name;
        if (name) {
          if (!run) setRun(name);
          setPop(await api.parallel(name));
        } else setPop(null);
      }
      setErr(null);
    } catch (e: any) {
      setErr(String(e.message || e));
    }
  }, [tab, session, run]);

  useEffect(() => { api.config().then(setCfg).catch(() => {}); }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!live) return;
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [live, refresh]);

  return (
    <TooltipProvider>
      <div className="wrap">
        <header className="top">
          <div>
            <p className="eyebrow">Autoresearch · Telemetry</p>
            <h1>Self-optimizing ML agent</h1>
            <p className="sub">
              {cfg?.task}<br />
              Objective: maximize <b className="mono">{cfg?.metric}</b> · agent <b className="mono">{cfg?.model}</b>
            </p>
          </div>
          <div className="controls">
            {tab === "solo" && sessions.length > 0 && (
              <select className="btn mono" value={session} onChange={(e) => setSession(e.target.value)}>
                <option value="">latest session</option>
                {sessions.map((s) => (
                  <option key={s.session} value={s.session}>
                    {s.session} · {s.agent} · best {s.best != null ? s.best.toFixed(3) : "—"}
                  </option>
                ))}
              </select>
            )}
            {tab === "population" && runs.length > 0 && (
              <select className="btn mono" value={run} onChange={(e) => { setRun(e.target.value); setPop(null); }}>
                {runs.map((r) => (
                  <option key={r.name} value={r.name}>{r.name} · {r.n_agents ?? "?"} agents</option>
                ))}
              </select>
            )}
            <button className={`btn ${live ? "on" : ""}`} onClick={() => setLive((l) => !l)} title="Auto-refresh every 5s">
              <span className={`dot-live ${live ? "pulse" : ""}`} style={{ background: live ? "var(--good)" : "var(--faint)" }} />
              {live ? "Live" : "Paused"}
            </button>
            <button className="btn" onClick={refresh}>Refresh</button>
            <button className="btn" onClick={toggle}>{theme === "dark" ? "Dark" : "Light"}</button>
          </div>
        </header>

        <nav className="tabs">
          <button className={`tab ${tab === "solo" ? "on" : ""}`} onClick={() => setTab("solo")}>Solo run</button>
          <button className={`tab ${tab === "population" ? "on" : ""}`} onClick={() => setTab("population")}>Population</button>
        </nav>

        {err && <div className="err">API error: {err} — is the backend running on :8000?</div>}

        {tab === "solo" ? (
          solo ? <SoloView data={solo} /> : <p className="empty">Loading…</p>
        ) : pop ? (
          <PopulationView data={pop} />
        ) : (
          <p className="empty">
            {runs.length === 0 ? "No population runs yet — try: python parallel.py --agents 4 --iterations 6" : "Loading…"}
          </p>
        )}

        <footer>
          Data from the durable archive (<span className="mono">history/</span>) · autoresearch agent
        </footer>
      </div>
    </TooltipProvider>
  );
}
