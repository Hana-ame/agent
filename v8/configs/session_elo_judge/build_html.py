#!/usr/bin/env python3
"""把 ELO 裁判/排行结果生成人工核对 HTML。"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"

def main() -> int:
    judge = json.loads((REPORTS / "session_elo_judge.json").read_text(encoding="utf-8"))
    ranking = json.loads((REPORTS / "session_elo_ranking.json").read_text(encoding="utf-8"))
    payload = json.dumps({"judge": judge, "ranking": ranking}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")

    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DSH Session ELO 人工核对</title>
<style>
  :root{--bg:#f6f8fb;--panel:#fff;--ink:#0f172a;--muted:#64748b;--line:#dbe3ee;
    --blue:#2563eb;--green:#059669;--red:#dc2626;--amber:#d97706;--purple:#7c3aed;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}
  header{padding:26px 32px 18px;background:linear-gradient(135deg,#eef4ff,#f7f4ff 55%,#eefaf5);border-bottom:1px solid var(--line)}
  h1{margin:0 0 5px;font-size:25px;letter-spacing:-.02em}
  .sub{color:var(--muted);margin:0}
  .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
  .chip{background:#fff;border:1px solid var(--line);border-radius:999px;padding:4px 10px;font-size:12px}
  .warn{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;border-radius:10px;padding:8px 12px;margin-top:12px;font-size:12.5px}
  main{max-width:1500px;margin:0 auto;padding:16px 24px 30px}
  .kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:0 0 14px}
  .kpi{background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px 12px}
  .kpi b{display:block;font-size:22px;line-height:1.2}.kpi span{color:var(--muted);font-size:11.5px}
  .panel{background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 8px 24px rgba(15,23,42,.06)}
  .toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:12px 14px;border-bottom:1px solid var(--line)}
  .tabs{display:flex;gap:6px;flex-wrap:wrap;margin-right:auto}
  button{font:inherit;font-size:12px;border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 10px;cursor:pointer}
  button:hover{background:#f8fafc} button.active{background:var(--ink);color:#fff;border-color:var(--ink)}
  button.ok{border-color:#86efac;color:#166534;background:#f0fdf4}
  button.no{border-color:#fca5a5;color:#991b1b;background:#fef2f2}
  button.mid{border-color:#fcd34d;color:#92400e;background:#fffbeb}
  button.on{outline:2px solid currentColor;outline-offset:1px}
  input[type=search],input[type=text]{font:inherit;font-size:12px;border:1px solid var(--line);border-radius:8px;padding:6px 9px;outline:none;background:#fff}
  input[type=search]{min-width:220px}
  table{width:100%;border-collapse:separate;border-spacing:0;font-size:12.5px}
  th{position:sticky;top:0;background:#f8fafc;color:var(--muted);text-align:left;font-weight:600;padding:9px 10px;border-bottom:1px solid var(--line);z-index:2;white-space:nowrap}
  td{padding:9px 10px;border-bottom:1px solid #edf1f7;vertical-align:top}
  tr:hover td{background:#fbfdff}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px}
  .muted{color:var(--muted)} .small{font-size:11.5px}
  .elo{font-weight:700;color:var(--blue)}
  .score{font-variant-numeric:tabular-nums}
  .winner{font-weight:700;border-radius:6px;padding:1px 7px;display:inline-block}
  .winner.A{background:#eff6ff;color:#1d4ed8}.winner.B{background:#f5f3ff;color:#6d28d9}.winner.tie{background:#f1f5f9;color:#475569}
  details{margin-top:4px} summary{cursor:pointer;color:var(--blue);font-size:11.5px}
  pre{white-space:pre-wrap;word-break:break-word;background:#0f172a;color:#e2e8f0;border-radius:8px;padding:9px;font-size:11px;max-height:280px;overflow:auto;margin:6px 0 0}
  .verdict{display:flex;gap:5px;flex-wrap:wrap}.verdict button{padding:3px 7px;font-size:11px}
  .note{margin-top:5px;width:100%;font-size:11px;padding:4px 6px;border:1px solid var(--line);border-radius:6px}
  .empty{padding:24px;text-align:center;color:var(--muted)}
  .bar{height:8px;background:#e2e8f0;border-radius:999px;overflow:hidden;margin-left:8px;min-width:120px;flex:1}
  .bar i{display:block;height:100%;background:var(--green);width:0}
  .foot{padding:10px 14px;border-top:1px solid var(--line);color:var(--muted);font-size:11.5px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .hidden{display:none}
  @media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}header{padding:20px 16px}main{padding:12px}}
</style>
</head>
<body>
<header>
  <h1>DSH Session ELO 人工核对</h1>
  <p class="sub">裁判 DAG 结果 + 全量启发式排行；先核对对局票，再看排名。</p>
  <div class="chips">
    <span class="chip">生成 <b id="gen"></b></span>
    <span class="chip">裁判后端 <b id="backend"></b></span>
    <span class="chip">选手 <b id="players"></b></span>
    <span class="chip">对局 <b id="pairs"></b></span>
    <span class="chip">有效票 <b id="votes"></b></span>
  </div>
  <div class="warn" id="warn"></div>
</header>
<main>
  <div class="kpis">
    <div class="kpi"><b id="k1"></b><span>裁判选手</span></div>
    <div class="kpi"><b id="k2"></b><span>对局</span></div>
    <div class="kpi"><b id="k3"></b><span>已核对</span></div>
    <div class="kpi"><b id="k4"></b><span>同意</span></div>
    <div class="kpi"><b id="k5"></b><span>不同意/存疑</span></div>
  </div>
  <section class="panel">
    <div class="toolbar">
      <div class="tabs">
        <button class="active" data-tab="matches">裁判对局</button>
        <button data-tab="leader">裁判排名</button>
        <button data-tab="full">全量排行</button>
      </div>
      <input id="q" type="search" placeholder="搜索 session / 标题 / 理由…">
      <button id="export">导出核对结果</button>
      <button id="clear">清空本地核对</button>
    </div>
    <div id="matchesView">
      <table>
        <thead><tr><th>#</th><th>对局</th><th>裁判</th><th>人工判定</th></tr></thead>
        <tbody id="matchBody"></tbody>
      </table>
      <div class="empty" id="matchEmpty" hidden>没有匹配的对局</div>
    </div>
    <div id="leaderView" class="hidden">
      <table><thead><tr><th>#</th><th>Elo</th><th>质量分</th><th>Session</th><th>标题</th><th>类型</th><th>战绩</th><th>字数</th></tr></thead>
        <tbody id="leaderBody"></tbody></table>
    </div>
    <div id="fullView" class="hidden">
      <table><thead><tr><th>#</th><th>Elo</th><th>质量分</th><th>Session</th><th>标题</th><th>类型</th><th>工作区</th><th>字数</th></tr></thead>
        <tbody id="fullBody"></tbody></table>
      <div class="empty" id="fullMore" hidden><button id="more">显示更多</button></div>
    </div>
    <div class="foot">
      <span>进度</span><div class="bar"><i id="progress"></i></div><span id="progressText">0%</span>
      <span class="muted">核对状态只存浏览器 localStorage；导出后可提交。</span>
    </div>
  </section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const J = DATA.judge, R = DATA.ranking;
const KEY = 'dsh-elo-human-audit-v1';
let audit = {};
try { audit = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch(e) { audit = {}; }

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const shortId = s => String(s || '').slice(0, 24);
const players = {}; J.items.forEach(x => players[x.session_id] = x);

document.getElementById('gen').textContent = J.generated_at;
document.getElementById('backend').textContent = J.backend || 'local';
document.getElementById('players').textContent = J.players;
document.getElementById('pairs').textContent = J.pairs;
document.getElementById('votes').textContent = J.votes;
document.getElementById('warn').textContent = J.backend === 'local'
  ? '当前是 local 裁判：用质量分模拟投票，只用于核对 DAG/采样/票数展示；不是真实 LLM 语义裁判。'
  : '当前是 http 裁判：逐对判 A/B/tie；请人工复核 winner 与 reason。';
document.getElementById('k1').textContent = J.players;
document.getElementById('k2').textContent = J.pairs;
document.getElementById('k3').textContent = Object.keys(audit).length;

function save(){ localStorage.setItem(KEY, JSON.stringify(audit)); updateKpis(); }
function updateKpis(){
  const vals = Object.values(audit);
  document.getElementById('k3').textContent = vals.length;
  document.getElementById('k4').textContent = vals.filter(v => v.verdict === 'agree').length;
  document.getElementById('k5').textContent = vals.filter(v => v.verdict === 'disagree' || v.verdict === 'unsure').length;
  const pct = J.pairs ? Math.round(vals.length / J.pairs * 100) : 0;
  document.getElementById('progress').style.width = pct + '%';
  document.getElementById('progressText').textContent = pct + '%';
}
function playerCell(p, elo){
  const item = players[p] || {};
  return `<div class="mono">${esc(shortId(p))}</div>
    <div>${esc(item.title || '')}</div>
    <div class="small muted">Elo ${elo ?? item.elo ?? '-'} · 质量分 ${(item.score ?? 0).toFixed(2)} · ${item.kind || ''}</div>
    <details><summary>结论摘录</summary><pre>${esc(item.excerpt || '（无摘录）')}</pre></details>`;
}
function renderMatches(){
  const q = document.getElementById('q').value.trim().toLowerCase();
  const rows = J.details.filter(d => {
    if (!q) return true;
    const s = [d.pair_id,d.winner,d.reason,d.a,d.b].join(' ').toLowerCase();
    return s.includes(q);
  });
  const body = document.getElementById('matchBody');
  body.innerHTML = rows.map((d, i) => {
    const a = players[d.a] || {}, b = players[d.b] || {};
    const st = audit[d.pair_id] || {};
    const pick = (v, label, cls) => `<button class="${cls} ${st.verdict===v?'on':''}" data-pair="${esc(d.pair_id)}" data-verdict="${v}">${label}</button>`;
    return `<tr>
      <td>${i+1}</td>
      <td>
        <div class="small muted">A · ${esc(a.title || '')}</div>
        ${playerCell(d.a, a.elo)}
        <hr style="border:0;border-top:1px dashed var(--line);margin:8px 0">
        <div class="small muted">B · ${esc(b.title || '')}</div>
        ${playerCell(d.b, b.elo)}
      </td>
      <td><span class="winner ${esc(d.winner)}">${esc(d.winner)}</span><div class="small muted" style="margin-top:4px">${esc(d.reason || '')}</div></td>
      <td>
        <div class="verdict">
          ${pick('agree','同意','ok')} ${pick('disagree','不同意','no')} ${pick('unsure','存疑','mid')}
        </div>
        <input class="note" data-note="${esc(d.pair_id)}" placeholder="备注…" value="${esc(st.note || '')}">
      </td>
    </tr>`;
  }).join('');
  document.getElementById('matchEmpty').hidden = rows.length !== 0;
}
function renderLeader(){
  const q = document.getElementById('q').value.trim().toLowerCase();
  const rows = J.items.filter(x => !q || [x.session_id,x.title,x.kind].join(' ').toLowerCase().includes(q));
  document.getElementById('leaderBody').innerHTML = rows.map(x => `<tr>
    <td>${x.rank}</td><td class="elo">${x.elo}</td><td class="score">${(x.score||0).toFixed(2)}</td>
    <td class="mono">${esc(shortId(x.session_id))}</td><td>${esc(x.title)}</td><td>${esc(x.kind)}</td>
    <td>${x.wins}W/${x.losses}L/${x.ties}T</td><td>${x.chars}</td></tr>`).join('');
}
let shown = 200;
function renderFull(){
  const q = document.getElementById('q').value.trim().toLowerCase();
  const rows = R.items.filter(x => !q || [x.session_id,x.title,x.kind,x.cwd].join(' ').toLowerCase().includes(q));
  document.getElementById('fullBody').innerHTML = rows.slice(0, shown).map(x => `<tr>
    <td>${x.rank}</td><td class="elo">${x.elo}</td><td class="score">${x.score.toFixed(2)}</td>
    <td class="mono">${esc(shortId(x.session_id))}</td><td>${esc(x.title)}</td><td>${esc(x.kind)}</td>
    <td class="small">${esc(x.cwd)}</td><td>${x.chars}</td></tr>`).join('');
  document.getElementById('fullMore').hidden = rows.length <= shown;
}
document.querySelectorAll('[data-tab]').forEach(b => b.onclick = () => {
  document.querySelectorAll('[data-tab]').forEach(x => x.classList.toggle('active', x === b));
  const t = b.dataset.tab;
  document.getElementById('matchesView').classList.toggle('hidden', t !== 'matches');
  document.getElementById('leaderView').classList.toggle('hidden', t !== 'leader');
  document.getElementById('fullView').classList.toggle('hidden', t !== 'full');
  document.getElementById('q').placeholder = t === 'full' ? '搜索 1920 条 session / 标题 / 工作区…' : '搜索 session / 标题 / 理由…';
  t === 'matches' ? renderMatches() : (t === 'leader' ? renderLeader() : renderFull());
});
document.getElementById('q').oninput = () => {
  const active = document.querySelector('[data-tab].active').dataset.tab;
  active === 'matches' ? renderMatches() : (active === 'leader' ? renderLeader() : renderFull());
};
document.getElementById('matchBody').addEventListener('click', e => {
  const b = e.target.closest('[data-pair]');
  if (!b) return;
  const id = b.dataset.pair; const v = b.dataset.verdict;
  audit[id] = audit[id] || {};
  audit[id].verdict = audit[id].verdict === v ? null : v;
  save(); renderMatches();
});
document.getElementById('matchBody').addEventListener('input', e => {
  const n = e.target.closest('[data-note]'); if (!n) return;
  const id = n.dataset.note; audit[id] = audit[id] || {}; audit[id].note = n.value;
  localStorage.setItem(KEY, JSON.stringify(audit));
});
document.getElementById('more').onclick = () => { shown += 200; renderFull(); };
document.getElementById('clear').onclick = () => {
  if (confirm('清空本地核对状态？')) { audit = {}; localStorage.removeItem(KEY); renderMatches(); updateKpis(); }
};
document.getElementById('export').onclick = () => {
  const out = { exported_at: new Date().toISOString(), source: J.generated_at, audits: J.details.map(d => ({
    pair_id: d.pair_id, a: d.a, b: d.b, judge_winner: d.winner, reason: d.reason,
    human_verdict: (audit[d.pair_id] || {}).verdict || null,
    human_note: (audit[d.pair_id] || {}).note || ''
  }))};
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'elo_human_audit.json'; a.click(); URL.revokeObjectURL(a.href);
};
renderMatches(); renderLeader(); renderFull(); updateKpis();
</script>
</body>
</html>'''
    out = REPORTS / "session_elo_judge.html"
    out.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    print(out, out.stat().st_size)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
