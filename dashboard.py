#!/usr/bin/env python3
"""
Dashboard generator — turns the agent's ``logs.md`` + ``runs/*/metrics.json``
into a single self-contained HTML telemetry page.

    python dashboard.py                 # -> model_dir/dashboard.html (standalone)
    python dashboard.py --open          # ...and open it in a browser
    python dashboard.py --fragment out  # inner-only (for embedding / artifacts)

The page centres on the **optimization curve**: best-so-far target metric across
experiments, with each experiment's own score and whether it was kept or
reverted. No external assets — charts are inline SVG drawn from embedded JSON,
so it works offline and as a shareable artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import config
from agent.logs_manager import read_logs

_SUCCESS = {"statistically better", "better"}
_METRIC = config.TARGET_METRIC


def _tok_str(rec: dict) -> str:
    ti, to, tc = rec.get("tokens_in", 0), rec.get("tokens_out", 0), rec.get("tokens_cached", 0)
    if not (ti or to):
        return "—"
    s = f"{ti}→{to}"
    return s + (f" ({tc} cached)" if tc else "")


def _load_archive(path: Path) -> "dict[str, list]":
    """Group archive JSONL records by session (preserving file order)."""
    sessions: "dict[str, list]" = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        sessions.setdefault(rec.get("session", "?"), []).append(rec)
    return sessions


def collect(session: str | None = None, all_sessions: bool = False) -> dict:
    """Build dashboard data. Prefers the durable archive; falls back to logs.md.

    The archive (history/experiments.jsonl) survives model_dir resets, so the
    dashboard reflects real history even after the working logs.md is wiped.
    """
    archive = config.ARCHIVE_DIR / "experiments.jsonl"
    if archive.exists() and archive.stat().st_size > 0:
        return _collect_from_archive(archive, session, all_sessions)
    return _collect_from_logs()


def _collect_from_archive(path: Path, session: str | None, all_sessions: bool) -> dict:
    sessions = _load_archive(path)
    session_ids = sorted(sessions)  # session ids are sortable timestamps
    if all_sessions:
        records = [r for sid in session_ids for r in sessions[sid]]
        chosen = f"all ({len(session_ids)} sessions)"
    else:
        sid = session if session in sessions else (session_ids[-1] if session_ids else None)
        records = sessions.get(sid, [])
        chosen = sid or "—"

    experiments, best, baseline = [], None, None
    best_series, best_run_id, best_val = [], None, -1.0
    cum = 0.0
    for i, r in enumerate(records, 1):
        score = r.get("score")
        if score is not None:
            best = score if best is None else max(best, score)
        cum += float(r.get("cost") or 0.0)
        metrics = r.get("metrics") or {}
        baseline = baseline if baseline is not None else metrics.get("pos_rate")
        va = metrics.get(_METRIC)
        if va is not None and va > best_val:
            best_val = va
            best_series = (r.get("series") or {}).get(_METRIC, [])
            best_run_id = r.get("run_id")
        experiments.append({
            "n": i,
            "iteration": r.get("iteration"),
            "target": r.get("target") or "—",
            "description": r.get("short_description") or r.get("status", ""),
            "status": r.get("status", ""),
            "kept": bool(r.get("kept")),
            "score": score,
            "best_so_far": best,
            "cost": r.get("cost"),
            "cum_cost": round(cum, 6),
            "tokens": _tok_str(r),
        })

    scored = [e for e in experiments if e["score"] is not None]
    kept = sum(1 for e in scored if e["kept"])
    return {
        "meta": {
            "metric": _METRIC, "direction": config.TARGET_DIRECTION,
            "model": config.LLM_MODEL, "model_dir": config.MODEL_DIR.name,
            "task": "mammography · rare-event medical detection (~2.3% positive)",
            "source": "history/experiments.jsonl (durable archive)",
            "session": chosen, "n_sessions": len(session_ids),
        },
        "experiments": experiments,
        "best_series": best_series, "best_run_id": best_run_id, "baseline": baseline,
        "summary": {
            "best": best, "count": len(scored), "kept": kept,
            "reverted": len(scored) - kept, "attempts": len(experiments),
            "total_cost": round(cum, 6),
            "first_score": scored[0]["score"] if scored else None,
        },
    }


def _collect_from_logs() -> dict:
    """Fallback: read the ephemeral logs.md (used before any archive exists)."""
    entries, _ = read_logs(config.LOGS_FILE)
    experiments, best = [], None
    for e in entries:
        if e.score is not None:
            best = e.score if best is None else max(best, e.score)
        experiments.append({
            "n": e.number, "target": e.target, "description": e.description,
            "status": e.status, "kept": e.status in _SUCCESS, "score": e.score,
            "best_so_far": best, "cost": e.cost, "cum_cost": e.cum_cost, "tokens": e.tokens,
        })
    best_series, baseline, best_run_id, best_val = [], None, None, -1.0
    if config.RUNS_DIR.exists():
        for mp in config.RUNS_DIR.glob("*/metrics.json"):
            try:
                data = json.loads(mp.read_text())
            except Exception:
                continue
            va = data.get("metrics", {}).get(_METRIC, -1)
            baseline = baseline if baseline is not None else data.get("metrics", {}).get("pos_rate")
            if va is not None and va > best_val:
                best_val, best_series, best_run_id = va, data.get("series", {}).get(_METRIC, []), data.get("run_id")
    scored = [e for e in experiments if e["score"] is not None]
    kept = sum(1 for e in experiments if e["kept"])
    total_cost = next((e["cum_cost"] for e in reversed(experiments) if e["cum_cost"] is not None), 0.0)
    return {
        "meta": {"metric": _METRIC, "direction": config.TARGET_DIRECTION,
                 "model": config.LLM_MODEL, "model_dir": config.MODEL_DIR.name,
                 "task": "mammography · rare-event medical detection (~2.3% positive)",
                 "source": "model_dir/logs.md (ephemeral — run the agent to build history/)",
                 "session": "—", "n_sessions": 0},
        "experiments": experiments, "best_series": best_series,
        "best_run_id": best_run_id, "baseline": baseline,
        "summary": {"best": best, "count": len(experiments), "kept": kept,
                    "reverted": len(experiments) - kept, "attempts": len(experiments),
                    "total_cost": total_cost,
                    "first_score": scored[0]["score"] if scored else None},
    }


# --------------------------------------------------------------------------- HTML

_STYLE = """
:root{
  --ground:#f5f7fb; --surface:#ffffff; --surface-2:#eef2f9; --border:#dbe3f0;
  --ink:#16202f; --muted:#55637a; --faint:#8493a8;
  --accent:#0e8fa8; --accent-soft:rgba(14,143,168,.14);
  --good:#12a266; --warn:#b9781a; --crit:#d83a52;
  --grid:rgba(20,40,70,.08);
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#0a1020; --surface:#111a2e; --surface-2:#16223c; --border:#243352;
    --ink:#e8eef7; --muted:#93a4c0; --faint:#5d708e;
    --accent:#35d0e8; --accent-soft:rgba(53,208,232,.16);
    --good:#37d391; --warn:#f2b544; --crit:#ff6b81;
    --grid:rgba(150,180,220,.10);
  }
}
:root[data-theme=light]{
  --ground:#f5f7fb; --surface:#ffffff; --surface-2:#eef2f9; --border:#dbe3f0;
  --ink:#16202f; --muted:#55637a; --faint:#8493a8;
  --accent:#0e8fa8; --accent-soft:rgba(14,143,168,.14);
  --good:#12a266; --warn:#b9781a; --crit:#d83a52; --grid:rgba(20,40,70,.08);
}
:root[data-theme=dark]{
  --ground:#0a1020; --surface:#111a2e; --surface-2:#16223c; --border:#243352;
  --ink:#e8eef7; --muted:#93a4c0; --faint:#5d708e;
  --accent:#35d0e8; --accent-soft:rgba(53,208,232,.16);
  --good:#37d391; --warn:#f2b544; --crit:#ff6b81; --grid:rgba(150,180,220,.10);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;line-height:1.5}
.mono{font-family:"SF Mono","JetBrains Mono","Cascadia Code",ui-monospace,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 60px}

header.top{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;
  flex-wrap:wrap;margin-bottom:26px}
.eyebrow{font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);
  font-weight:700;margin:0 0 6px}
h1{font-size:23px;margin:0;letter-spacing:-.01em;font-weight:650;text-wrap:balance}
.sub{color:var(--muted);font-size:13.5px;margin-top:5px}
.sub b{color:var(--ink);font-weight:600}
.toggle{border:1px solid var(--border);background:var(--surface);color:var(--muted);
  border-radius:9px;padding:8px 12px;font-size:12px;cursor:pointer;display:flex;gap:7px;align-items:center}
.toggle:hover{color:var(--ink);border-color:var(--accent)}
.toggle:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:13px;padding:14px 15px}
.kpi .label{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);font-weight:600}
.kpi .val{font-size:25px;font-weight:640;margin-top:7px;letter-spacing:-.02em}
.kpi .note{font-size:11.5px;color:var(--muted);margin-top:3px}
.kpi .up{color:var(--good)} .kpi .down{color:var(--crit)}

.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:18px 18px 8px}
.card.hero{padding:20px 22px 12px;margin-bottom:16px;position:relative;overflow:hidden}
.card h2{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
  font-weight:650;margin:0 0 2px}
.card .cardsub{font-size:12.5px;color:var(--faint);margin:0 0 8px}
.hero-figure{font-size:44px;font-weight:660;letter-spacing:-.03em;line-height:1;margin:4px 0 2px}
.hero-figure .unit{font-size:16px;color:var(--muted);font-weight:500;margin-left:6px}
.delta{font-size:13px;font-weight:600}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}

svg{display:block;width:100%;height:auto;overflow:visible}
.axis text{fill:var(--faint);font-size:10.5px}
.axis line{stroke:var(--grid)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:8px 2px 4px}
.legend .k{display:inline-flex;align-items:center;gap:6px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.swatch{width:14px;height:3px;border-radius:2px;display:inline-block}

table{width:100%;border-collapse:collapse;font-size:13px}
.tablewrap{overflow-x:auto}
th{text-align:left;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);
  font-weight:600;padding:0 12px 9px;border-bottom:1px solid var(--border);white-space:nowrap}
td{padding:11px 12px;border-bottom:1px solid var(--border);vertical-align:top}
tr:last-child td{border-bottom:none}
td.num{text-align:right;white-space:nowrap}
.chip{font-size:11px;padding:2px 8px;border-radius:6px;background:var(--surface-2);
  border:1px solid var(--border);color:var(--muted);white-space:nowrap}
.pill{font-size:11px;font-weight:600;padding:2px 9px;border-radius:20px;display:inline-flex;
  align-items:center;gap:5px;white-space:nowrap}
.pill::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.pill.good{color:var(--good);background:color-mix(in srgb,var(--good) 14%,transparent)}
.pill.warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
.pill.crit{color:var(--crit);background:color-mix(in srgb,var(--crit) 14%,transparent)}
.desc{color:var(--muted);max-width:430px}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--ground);
  padding:7px 10px;border-radius:8px;font-size:11.5px;opacity:0;transition:opacity .1s;z-index:9;
  box-shadow:0 6px 24px rgba(0,0,0,.28);max-width:240px}
.tip .t-k{color:var(--ground);opacity:.7}
footer{color:var(--faint);font-size:11.5px;margin-top:26px;text-align:center}
@media (max-width:820px){
  .kpis{grid-template-columns:repeat(2,1fr)} .row2{grid-template-columns:1fr}
  .desc{max-width:none}
}
"""

_BODY = """
<div class="wrap">
  <header class="top">
    <div>
      <p class="eyebrow">Autoresearch · Run Telemetry</p>
      <h1>Self-optimizing ML agent</h1>
      <p class="sub"><b id="m-task"></b><br>
        Objective: maximize <b id="m-metric" class="mono"></b> · agent <b id="m-model" class="mono"></b><br>
        <span id="m-source" style="font-size:11.5px;color:var(--faint)"></span></p>
    </div>
    <button class="toggle" id="themeBtn" aria-label="Toggle theme"><span id="themeIcon">◐</span><span id="themeTxt">Theme</span></button>
  </header>

  <section class="kpis" id="kpis"></section>

  <section class="card hero">
    <h2>Optimization progress</h2>
    <p class="cardsub">Best <span class="mono" id="h-metric"></span> reached after each experiment. Points show every experiment's own score — kept or reverted.</p>
    <div class="hero-figure mono"><span id="h-best">—</span><span class="unit" id="h-unit"></span></div>
    <div class="delta mono" id="h-delta"></div>
    <div id="heroChart"></div>
    <div class="legend">
      <span class="k"><span class="swatch" style="background:var(--accent)"></span>Best so far</span>
      <span class="k"><span class="dot" style="background:var(--good)"></span>Kept (committed)</span>
      <span class="k"><span class="dot" style="background:var(--crit)"></span>Reverted</span>
      <span class="k"><span class="swatch" style="background:var(--faint);height:0;border-top:2px dashed var(--faint)"></span>Naive baseline</span>
    </div>
  </section>

  <section class="row2">
    <div class="card">
      <h2>Spend per experiment</h2>
      <p class="cardsub">LLM cost each turn · cumulative <span class="mono" id="c-total"></span></p>
      <div id="costChart"></div>
    </div>
    <div class="card">
      <h2>Best run · learning curve</h2>
      <p class="cardsub"><span class="mono" id="lc-metric"></span> across training epochs (run <span class="mono" id="lc-run"></span>)</p>
      <div id="lcChart"></div>
    </div>
  </section>

  <section class="card" style="padding-bottom:6px">
    <h2>Experiment log</h2>
    <p class="cardsub">Every experiment the agent proposed, in order</p>
    <div class="tablewrap">
      <table><thead><tr>
        <th>#</th><th>Lever</th><th>Experiment</th><th class="num">Score</th>
        <th>Outcome</th><th class="num">Tokens</th><th class="num">Cost</th>
      </tr></thead><tbody id="logBody"></tbody></table>
    </div>
  </section>

  <footer>Generated from <span class="mono">logs.md</span> · autoresearch agent · <span id="gen-count"></span> experiments</footer>
</div>
<div class="tip" id="tip"></div>
"""

_SCRIPT = """
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const cvar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v,d=4)=> v==null? "—" : (+v).toFixed(d);
const NS="http://www.w3.org/2000/svg";
function el(n,a={}){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
const tip=$("#tip");
function showTip(html,x,y){tip.innerHTML=html;tip.style.opacity=1;
  const w=tip.offsetWidth,h=tip.offsetHeight;
  tip.style.left=Math.min(x+14,innerWidth-w-8)+"px";
  tip.style.top=Math.max(y-h-10,8)+"px";}
function hideTip(){tip.style.opacity=0;}

function fillMeta(){
  const m=DATA.meta,s=DATA.summary;
  $("#m-task").textContent=m.task;
  $("#m-metric").textContent=m.metric; $("#h-metric").textContent=m.metric;
  $("#lc-metric").textContent=m.metric; $("#m-model").textContent=m.model;
  $("#lc-run").textContent=DATA.best_run_id||"—";
  const src=$("#m-source");
  if(src){const ses=m.session&&m.session!=="—"?` · session ${m.session}`:"";
    src.textContent=`source: ${m.source||"logs.md"}${ses}`;}
  $("#gen-count").textContent=s.count;
  $("#h-best").textContent=fmt(s.best);
  $("#c-total").textContent="$"+fmt(s.total_cost,4);
  if(s.best!=null && s.first_score!=null){
    const d=s.best-s.first_score, pct=s.first_score? d/s.first_score*100:0;
    $("#h-delta").innerHTML=`<span style="color:var(--good)">▲ ${fmt(d,4)}</span> `+
      `<span style="color:var(--muted)">vs first experiment (${fmt(s.first_score)}) · +${pct.toFixed(0)}%</span>`;
  }
  const attempts=s.attempts!=null?s.attempts:s.count;
  const k=[
    ["Best "+m.metric, fmt(s.best), "higher is better"],
    ["Experiments", s.count, attempts>s.count?`${attempts} attempts (incl. rejected/crashed)`:"trainings completed"],
    ["Kept / reverted", `${s.kept} / ${s.reverted}`, "committed vs discarded"],
    ["Total spend", "$"+fmt(s.total_cost,4), "LLM cost (all calls)"],
    ["Success rate", s.count? Math.round(s.kept/s.count*100)+"%":"—", "kept of completed"],
  ];
  $("#kpis").innerHTML=k.map(([l,v,n])=>
    `<div class="kpi"><div class="label">${l}</div><div class="val mono">${v}</div><div class="note">${n}</div></div>`).join("");
}

function scales(vals,pad){const mn=Math.min(...vals),mx=Math.max(...vals);
  const r=(mx-mn)||1;return {mn:mn-r*pad,mx:mx+r*pad};}

function heroChart(){
  const box=$("#heroChart");box.innerHTML="";
  const E=DATA.experiments.filter(e=>e.score!=null);
  if(!E.length){box.innerHTML='<p class="cardsub">No experiments yet.</p>';return;}
  const W=box.clientWidth||900,H=300,m={t:14,r:16,b:30,l:44};
  const iw=W-m.l-m.r,ih=H-m.t-m.b;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,role:"img"});
  const xs=E.map(e=>e.n);
  const allv=E.map(e=>e.score).concat(E.map(e=>e.best_so_far));
  if(DATA.baseline!=null) allv.push(DATA.baseline);
  const {mn,mx}=scales(allv,.08);
  const X=n=> m.l + (xs.length<2?iw/2:(n-xs[0])/(xs[xs.length-1]-xs[0])*iw);
  const Y=v=> m.t + (1-(v-mn)/((mx-mn)||1))*ih;
  // grid + y labels
  const g=el("g",{class:"axis"});
  for(let i=0;i<=4;i++){const v=mn+(mx-mn)*i/4,y=Y(v);
    g.appendChild(el("line",{x1:m.l,y1:y,x2:W-m.r,y2:y}));
    const t=el("text",{x:m.l-8,y:y+3,"text-anchor":"end"});t.textContent=v.toFixed(2);g.appendChild(t);}
  E.forEach(e=>{const t=el("text",{x:X(e.n),y:H-10,"text-anchor":"middle"});t.textContent="#"+e.n;g.appendChild(t);});
  svg.appendChild(g);
  // baseline
  if(DATA.baseline!=null){const y=Y(DATA.baseline);
    svg.appendChild(el("line",{x1:m.l,y1:y,x2:W-m.r,y2:y,stroke:cvar("--faint"),
      "stroke-width":1.5,"stroke-dasharray":"5 4",opacity:.7}));}
  // best-so-far area + line
  const acc=cvar("--accent");
  let dl=`M ${X(E[0].n)} ${Y(E[0].best_so_far)}`;
  E.forEach(e=>dl+=` L ${X(e.n)} ${Y(e.best_so_far)}`);
  const area=`${dl} L ${X(E[E.length-1].n)} ${Y(mn)} L ${X(E[0].n)} ${Y(mn)} Z`;
  svg.appendChild(el("path",{d:area,fill:cvar("--accent-soft")}));
  svg.appendChild(el("path",{d:dl,fill:"none",stroke:acc,"stroke-width":2.5,
    "stroke-linejoin":"round","stroke-linecap":"round"}));
  // per-experiment score points
  E.forEach(e=>{
    const c=e.kept?cvar("--good"):cvar("--crit");
    const cir=el("circle",{cx:X(e.n),cy:Y(e.score),r:5,fill:c,
      stroke:cvar("--surface"),"stroke-width":2});
    cir.style.cursor="pointer";
    cir.addEventListener("mousemove",ev=>showTip(
      `<b>#${e.n} · ${e.target}</b><br><span class="t-k">score</span> ${fmt(e.score)} `+
      `<span class="t-k">·</span> ${e.status}<br><span class="t-k">best</span> ${fmt(e.best_so_far)}`,
      ev.clientX,ev.clientY));
    cir.addEventListener("mouseleave",hideTip);
    svg.appendChild(cir);
  });
  // emphasized best endpoint
  const bi=E.reduce((a,b)=>b.best_so_far>=a.best_so_far?b:a,E[0]);
  svg.appendChild(el("circle",{cx:X(bi.n),cy:Y(bi.best_so_far),r:5.5,fill:acc,
    stroke:cvar("--surface"),"stroke-width":2.5}));
  const lbl=el("text",{x:X(bi.n),y:Y(bi.best_so_far)-12,"text-anchor":"middle",
    fill:cvar("--ink"),"font-size":12,"font-weight":700,class:"mono"});
  lbl.textContent=fmt(bi.best_so_far);svg.appendChild(lbl);
  box.appendChild(svg);
}

function costChart(){
  const box=$("#costChart");box.innerHTML="";
  const E=DATA.experiments.filter(e=>e.cost!=null);
  if(!E.length){box.innerHTML='<p class="cardsub">No cost data.</p>';return;}
  const W=box.clientWidth||500,H=200,m={t:12,r:14,b:26,l:44};
  const iw=W-m.l-m.r,ih=H-m.t-m.b;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`});
  const mx=Math.max(...E.map(e=>e.cost))||1;
  const bw=Math.min(38,iw/E.length*.62);
  const X=i=> m.l + (E.length<2?iw/2:i/(E.length-1)*(iw-bw)+bw/2);
  const Y=v=> m.t + (1-v/mx)*ih;
  const g=el("g",{class:"axis"});
  for(let i=0;i<=3;i++){const v=mx*i/3,y=Y(v);
    g.appendChild(el("line",{x1:m.l,y1:y,x2:W-m.r,y2:y}));
    const t=el("text",{x:m.l-8,y:y+3,"text-anchor":"end"});t.textContent="$"+v.toFixed(4);g.appendChild(t);}
  svg.appendChild(g);
  E.forEach((e,i)=>{const x=X(i)-bw/2,y=Y(e.cost),h=m.t+ih-y;
    const r=el("rect",{x,y,width:bw,height:Math.max(h,1),rx:4,fill:cvar("--accent"),opacity:.85});
    r.style.cursor="pointer";
    r.addEventListener("mousemove",ev=>showTip(
      `<b>#${e.n} · ${e.target}</b><br><span class="t-k">cost</span> $${fmt(e.cost,4)} `+
      `<span class="t-k">·</span> ${e.tokens} tok`,ev.clientX,ev.clientY));
    r.addEventListener("mouseleave",hideTip);
    svg.appendChild(r);
    const t=el("text",{x:X(i),y:H-9,"text-anchor":"middle",class:"axis"});
    t.setAttribute("fill",cvar("--faint"));t.setAttribute("font-size",10);
    t.textContent="#"+e.n;svg.appendChild(t);
  });
  box.appendChild(svg);
}

function lcChart(){
  const box=$("#lcChart");box.innerHTML="";
  const S=DATA.best_series||[];
  if(S.length<2){box.innerHTML='<p class="cardsub">No learning curve available.</p>';return;}
  const W=box.clientWidth||500,H=200,m={t:12,r:14,b:26,l:44};
  const iw=W-m.l-m.r,ih=H-m.t-m.b;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`});
  const xs=S.map(p=>p[0]),ys=S.map(p=>p[1]);
  const xmn=Math.min(...xs),xmx=Math.max(...xs);
  const {mn,mx}=scales(ys,.06);
  const X=x=> m.l+(x-xmn)/((xmx-xmn)||1)*iw;
  const Y=v=> m.t+(1-(v-mn)/((mx-mn)||1))*ih;
  const g=el("g",{class:"axis"});
  for(let i=0;i<=3;i++){const v=mn+(mx-mn)*i/3,y=Y(v);
    g.appendChild(el("line",{x1:m.l,y1:y,x2:W-m.r,y2:y}));
    const t=el("text",{x:m.l-8,y:y+3,"text-anchor":"end"});t.textContent=v.toFixed(2);g.appendChild(t);}
  [xmn,Math.round((xmn+xmx)/2),xmx].forEach(x=>{const t=el("text",{x:X(x),y:H-9,"text-anchor":"middle"});t.textContent="ep "+x;g.appendChild(t);});
  svg.appendChild(g);
  const acc=cvar("--accent");
  let dl=`M ${X(xs[0])} ${Y(ys[0])}`;
  S.forEach(p=>dl+=` L ${X(p[0])} ${Y(p[1])}`);
  svg.appendChild(el("path",{d:`${dl} L ${X(xmx)} ${Y(mn)} L ${X(xmn)} ${Y(mn)} Z`,fill:cvar("--accent-soft")}));
  svg.appendChild(el("path",{d:dl,fill:"none",stroke:acc,"stroke-width":2.5,"stroke-linejoin":"round"}));
  const last=S[S.length-1];
  svg.appendChild(el("circle",{cx:X(last[0]),cy:Y(last[1]),r:4.5,fill:acc,stroke:cvar("--surface"),"stroke-width":2}));
  box.appendChild(svg);
}

function logTable(){
  const cls={"statistically better":"good","better":"good","lower":"warn",
    "statistically lower":"warn","crushed":"crit","crashed":"crit",
    "rejected":"warn","no_run_id":"warn","internal_error":"crit"};
  $("#logBody").innerHTML=DATA.experiments.map(e=>`
    <tr>
      <td class="mono">${e.n}</td>
      <td><span class="chip mono">${e.target}</span></td>
      <td class="desc">${e.description}</td>
      <td class="num mono">${fmt(e.score)}</td>
      <td><span class="pill ${cls[e.status]||'warn'}">${e.status}</span></td>
      <td class="num mono" style="color:var(--muted)">${e.tokens||"—"}</td>
      <td class="num mono">${e.cost!=null?"$"+fmt(e.cost,4):"—"}</td>
    </tr>`).join("");
}

function renderAll(){fillMeta();$("#h-unit").textContent="";heroChart();costChart();lcChart();logTable();}

// theme toggle
function applyTheme(t){document.documentElement.setAttribute("data-theme",t);
  $("#themeTxt").textContent=t==="dark"?"Dark":"Light";
  requestAnimationFrame(()=>{heroChart();costChart();lcChart();});}
(function(){const mq=matchMedia("(prefers-color-scheme:dark)");
  applyThemeInit(mq.matches?"dark":"light");
  $("#themeBtn").addEventListener("click",()=>{
    const cur=document.documentElement.getAttribute("data-theme");
    applyTheme(cur==="dark"?"light":"dark");});
  function applyThemeInit(t){document.documentElement.setAttribute("data-theme",t);
    $("#themeTxt").textContent=t==="dark"?"Dark":"Light";}
})();
renderAll();
let rt;addEventListener("resize",()=>{clearTimeout(rt);rt=setTimeout(()=>{heroChart();costChart();lcChart();},150);});
"""


def render(data: dict, fragment: bool = False) -> str:
    script = _SCRIPT.replace("__DATA__", json.dumps(data))
    inner = f"<style>{_STYLE}</style>\n{_BODY}\n<script>{script}</script>"
    if fragment:
        return inner
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Autoresearch · Run Telemetry</title></head><body>"
        f"{inner}</body></html>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the autoresearch dashboard")
    # Default outside model_dir so the agent's git add/commit never touches it.
    ap.add_argument("--out", default=str(config.MODEL_DIR.parent / "dashboard.html"))
    ap.add_argument("--fragment", metavar="PATH", default=None,
                    help="also write an inner-only fragment (for embedding)")
    ap.add_argument("--open", action="store_true", help="open the result in a browser")
    ap.add_argument("--session", default=None, help="archive session id (default: latest)")
    ap.add_argument("--all", action="store_true", help="aggregate across all sessions")
    args = ap.parse_args()

    data = collect(session=args.session, all_sessions=args.all)
    Path(args.out).write_text(render(data))
    print(f"Wrote {args.out}  (source: {data['meta']['source']}; "
          f"session {data['meta']['session']}; {data['summary']['count']} experiments, "
          f"best {data['meta']['metric']}={data['summary']['best']})")
    if args.fragment:
        Path(args.fragment).write_text(render(data, fragment=True))
        print(f"Wrote fragment {args.fragment}")
    if args.open:
        webbrowser.open(f"file://{Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
