'use strict';

/* =========================================================
   TaskPad 前端 — ComfyUI 风格节点图（数据流版）
   多输入/输出端口、端口拖拽连线、成果捕获与卡片、运行图队列、定时、Trace
   ========================================================= */

const $ = (sel) => document.querySelector(sel);
const board = $('#board');
const world = $('#world');
const wiresSvg = $('#wires');
const panel = $('#panel');
const panelBody = $('#panel-body');

const NODE_W = 300;
const HEADER_H = 38;
const PORTS_Y0 = 51.5;        // 第一个端口圆心（相对节点顶部，世界坐标）
const PORT_STEP = 22;

const PALETTE = ['#3b82f6', '#22c55e', '#eab308', '#a855f7', '#06b6d4', '#f43f5e', '#f97316', '#8b5cf6'];

const state = {
  tasks: [],
  edges: [],
  graphs: [],
  schedules: [],
  runs: [],
  engines: [],
  templates: [],
  workers: [],
  activeGraphId: null,
  queue: { running: false, remaining: 0, currentTaskId: null }
};
const view = { x: 30, y: 30, scale: 1 };
const sel = { nodeId: null, edgeId: null };

const formData = { name: '', engine: 'v8', color: PALETTE[0], description: '', command: '', scope: '', spec: '', group: '' };
const portData = { inputs: [], outputs: [] };   // 端口编辑缓冲
const expandedPorts = new Set();                // 展开的端口（规范/测试编辑）
const schedData = { type: 'interval', minutes: 5, time: '08:00', cron: '0 * * * *', enabled: true };
let selectedTaskId = null;
let openRunId = null;
let fitDone = false;

const wireDrag = { active: false, fromTaskId: null, fromPort: null, fromSide: null, x: 0, y: 0, target: null };
let nodeDrag = null;

/* ---------- 工具 ---------- */
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}
async function api(path, opts) {
  const res = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, opts));
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
  return data;
}
function toast(msg, kind) {
  const t = document.createElement('div');
  t.className = 'toast' + (kind ? ' ' + kind : '');
  t.textContent = msg;
  $('#toasts').appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .3s'; }, 2600);
  setTimeout(() => t.remove(), 3000);
}
function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
function fmtNext(ts) {
  if (ts == null) return '—';
  return fmtTime(new Date(ts).toISOString());
}
function fmtDur(ms) {
  if (ms == null) return '';
  if (ms < 1000) return ms + 'ms';
  if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
  return (ms / 60000).toFixed(1) + 'min';
}
function fmtSize(n) {
  if (n == null) return '';
  if (n < 1024) return n + 'B';
  if (n < 1048576) return (n / 1024).toFixed(1) + 'KB';
  return (n / 1048576).toFixed(1) + 'MB';
}
function statusText(s) {
  return { idle: '待机', pending: '待认领', running: '运行中', success: '成功', error: '失败', skipped: '跳过', killed: '已终止', aborted: '已中止', queue: '排队中', stale: '失联' }[s] || s;
}
function engineOf(id) {
  return state.engines.find((e) => e.id === id) || { id, label: id, color: '#64748b' };
}
function taskOf(id) { return state.tasks.find((t) => t.id === id); }
function edgeOf(id) { return state.edges.find((e) => e.id === id); }
function graphById(id) { return (state.graphs || []).find((g) => g.id === id); }
function rootGraphId() { return (state.graphs && state.graphs[0] && state.graphs[0].id) || null; }
function currentGraph() { return graphById(state.activeGraphId) || (state.graphs && state.graphs[0]) || null; }
function currentTasks() {
  const gid = state.activeGraphId || rootGraphId();
  return state.tasks.filter((t) => (t.graphId || rootGraphId()) === gid);
}
function currentEdges() {
  const gid = state.activeGraphId || rootGraphId();
  return state.edges.filter((e) => {
    const f = taskOf(e.from), t2 = taskOf(e.to);
    if (!f || !t2) return false;
    return (f.graphId || rootGraphId()) === gid && (t2.graphId || rootGraphId()) === gid;
  });
}
function graphPathList() {
  const out = [];
  let g = currentGraph();
  const seen = new Set();
  while (g && !seen.has(g.id)) {
    seen.add(g.id);
    out.unshift(g);
    if (!g.parentNodeId) break;
    const owner = taskOf(g.parentNodeId);
    g = owner ? graphById(owner.graphId) : null;
  }
  return out;
}
function renderGraphBar() {
  const pathEl = $('#graph-path');
  if (!pathEl) return;
  const path = graphPathList();

  // 面包屑：主图 › 模块内部流程 › … 当前不可点，其他点一下跳过去
  pathEl.innerHTML = path.map((g, i) => {
    const last = i === path.length - 1;
    const isRoot = !g.parentNodeId;
    const owner = g.parentNodeId ? taskOf(g.parentNodeId) : null;
    const title = owner
      ? '内部流程图：' + g.name + '（所属模块：' + owner.name + '）'
      : '顶层图：' + g.name;
    const cls = 'gp-item' + (last ? ' current' : '') + (isRoot ? ' root' : '');
    const goto = last ? '' : ` data-goto-graph="${esc(g.id)}"`;
    const icon = isRoot ? '<span class="gp-ico">⌂</span>' : '<span class="gp-ico">◫</span>';
    return `<span class="${cls}"${goto} title="${esc(title)}">${icon}<span class="gp-txt">${esc(g.name)}</span></span>`
      + (last ? '' : '<span class="gp-sep">›</span>');
  }).join('') || '<span class="gp-item current"><span class="gp-txt">（无图）</span></span>';

  // 层级深/名字长时滚到当前项，别让当前图被挤出视野
  const cur = pathEl.querySelector('.gp-item.current');
  if (cur && cur.scrollIntoView) {
    try { cur.scrollIntoView({ block: 'nearest', inline: 'nearest' }); } catch (e) { /* ignore */ }
  }

  const back = $('#btn-graph-back');
  if (back) back.disabled = path.length <= 1;
  const del = $('#btn-graph-del');
  if (del) del.disabled = path.length <= 1;

  const sel = $('#graph-select');
  if (sel) {
    const roots = (state.graphs || []).filter((g) => !g.parentNodeId);
    sel.innerHTML = roots.map((g) => `<option value="${esc(g.id)}">${esc(g.name)}</option>`).join('');
    // 在子图里时，下拉框跟随该子图所属的顶层图
    const rootOf = path.length ? path[0].id : state.activeGraphId;
    sel.value = rootOf;
  }
}
$('#graph-select').addEventListener('change', (e) => enterGraph(e.target.value));
function enterGraph(id) {
  if (!graphById(id)) return;
  state.activeGraphId = id;
  sel.nodeId = null; sel.edgeId = null;
  renderGraphBar();
  renderNodes();
  fitView();
}
function goBackGraph() {
  const path = graphPathList();
  if (path.length <= 1) return;
  enterGraph(path[path.length - 2].id);
}
function portsIn(t) { return (t.inputs || []).length ? t.inputs : [{ id: 'in', name: 'dep' }]; }
function portsOut(t) { return (t.outputs || []).length ? t.outputs : [{ id: 'out', name: 'out', capture: 'stdout' }]; }

/* ---------- 视口 ---------- */
const GRID = 36;
function applyView() {
  world.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
  board.style.backgroundImage = 'radial-gradient(circle, #262c38 1.1px, transparent 1.5px)';
  board.style.backgroundSize = `${GRID * view.scale}px ${GRID * view.scale}px`;
  board.style.backgroundPosition = `${view.x}px ${view.y}px`;
  $('#zoom-level').textContent = Math.round(view.scale * 100) + '%';
  layoutWires();
  if (wireDrag.active) updateGhostWire();
}
function screenToWorld(cx, cy) {
  return [(cx - view.x) / view.scale, (cy - view.y) / view.scale];
}
function zoomBy(factor, cx, cy) {
  const [wx, wy] = screenToWorld(cx, cy);
  view.scale = Math.min(3.5, Math.max(0.15, view.scale * factor));
  view.x = cx - wx * view.scale;
  view.y = cy - wy * view.scale;
  applyView();
}
function fitView() {
  const list = currentTasks();
  if (!list.length) { view.x = 30; view.y = 30; view.scale = 1; applyView(); return; }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  list.forEach((t) => {
    minX = Math.min(minX, t.x); minY = Math.min(minY, t.y);
    maxX = Math.max(maxX, t.x + NODE_W); maxY = Math.max(maxY, t.y + nodeHeight(t));
  });
  const vw = board.clientWidth, vh = board.clientHeight;
  const bw = Math.max(maxX - minX, 400), bh = Math.max(maxY - minY, 300);
  const scale = Math.min(1.1, Math.max(0.15, Math.min((vw - 120) / bw, (vh - 120) / bh)));
  view.scale = scale;
  view.x = (vw - bw * scale) / 2 - minX * scale;
  view.y = (vh - bh * scale) / 2 - minY * scale;
  applyView();
}

/* ---------- 节点几何（模块，无 in/out 端口） ---------- */
const BODY_H = 76;
function nodeHeight(t) { return HEADER_H + BODY_H + 30; }
function edgeListFor(taskId, dir) {
  return state.edges.filter((e) => (dir === 'out' ? e.from === taskId : e.to === taskId));
}
function ifaceAnchor(task, dir, index, total) {
  const sideX = dir === 'out' ? task.x + NODE_W : task.x;
  const baseY = task.y + HEADER_H + 18;
  const step = 13;
  const y = baseY + (index - (total - 1) / 2) * step;
  return { x: sideX * view.scale + view.x, y: y * view.scale + view.y };
}
function wirePath(x1, y1, x2, y2) {
  const dx = Math.max(36, Math.abs(x2 - x1) * 0.5);
  return `M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${(x1 + dx).toFixed(1)} ${y1.toFixed(1)}, ${(x2 - dx).toFixed(1)} ${y2.toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}`;
}
function nodeStatusOf(t) {
  if (state.queue.currentTaskId === t.id) return 'queue';
  const run = state.runs.find((r) => r.taskId === t.id);
  if (!run) return 'idle';
  if (run.status === 'running') return 'running';
  return run.status;
}
function layoutWires() {
  const byId = new Map(state.tasks.map((t) => [t.id, t]));
  const edges = currentEdges();
  const outTotal = new Map(), inTotal = new Map();
  for (const e of edges) {
    outTotal.set(e.from, (outTotal.get(e.from) || 0) + 1);
    inTotal.set(e.to, (inTotal.get(e.to) || 0) + 1);
  }
  const outIdx = new Map(), inIdx = new Map();
  let html = '';
  for (const e of edges) {
    const from = byId.get(e.from), to = byId.get(e.to);
    if (!from || !to) continue;
    const oi = outIdx.get(e.from) || 0; outIdx.set(e.from, oi + 1);
    const ii = inIdx.get(e.to) || 0; inIdx.set(e.to, ii + 1);
    const a = ifaceAnchor(from, 'out', oi, outTotal.get(e.from));
    const b = ifaceAnchor(to, 'in', ii, inTotal.get(e.to));
    const fromStatus = nodeStatusOf(from);
    const kind = e.kind || 'code';
    const KIND_COLOR = { code: '#4f8cff', data: '#22c55e', net: '#f59e0b' };
    const color = fromStatus === 'running' ? '#f59e0b' : (KIND_COLOR[kind] || '#4f8cff');
    const KIND_DASH = { code: '10 6', data: '3 9', net: '14 8' };
    const extra = sel.edgeId === e.id ? ' selected' : '';
    const d = wirePath(a.x, a.y, b.x, b.y);
    html += `<path class="wire${extra}" data-edge="${esc(e.id)}" d="${d}" stroke="${esc(color)}" fill="none" stroke-width="2.5" stroke-dasharray="${KIND_DASH[kind] || '10 6'}"></path>`;
    html += `<path class="wire hit" data-edge="${esc(e.id)}" d="${d}"></path>`;
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2 - 8;
    const kindTag = kind === 'net' ? '🌐' : kind === 'data' ? '⇄' : '⌗';
    html += `<text class="wire-label" x="${mx.toFixed(1)}" y="${my.toFixed(1)}" text-anchor="middle" pointer-events="none">${kindTag} ${esc(e.name || e.label || '接口')}</text>`;
  }
  wiresSvg.innerHTML = html;
}
function updateGhostWire() {
  let g = wiresSvg.querySelector('path.ghost');
  if (!g) {
    g = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    g.setAttribute('class', 'wire ghost');
    wiresSvg.appendChild(g);
  }
  const from = taskOf(wireDrag.fromTaskId);
  if (!from) return;
  const a = ifaceAnchor(from, 'out', 0, 1);
  g.setAttribute('d', wirePath(a.x, a.y, wireDrag.x, wireDrag.y));
}

/* ---------- 成果卡片（点连线弹出） ---------- */
const wirePop = $('#wire-pop');
function hideWirePop() { wirePop.hidden = true; }
function showWirePop(e, mx, my) {
  const from = taskOf(e.from);
  const srcName = from ? from.name : '?';
  const carry = e.carryName || e.fromPortName || '?';
  const run = state.runs.find((r) => r.taskId === e.from && r.status === 'success');
  const res = run && (run.portResults || []).find((pr) => pr.portId === e.fromPort);
  const KIND_NAME = { code: '代码', data: '数据', net: '网络' };
  const ifaceMeta = [KIND_NAME[e.kind] || e.kind, e.proto, e.auth].filter(Boolean).join(' · ');
  wirePop.innerHTML = `
    <div class="wp-title">🔗 接口 · ${esc(e.label || carry)}</div>
    <div class="wp-src">${esc(ifaceMeta)}${e.spec ? '<br>规范：' + esc(e.spec) : ''}</div>
    <div class="wp-src">来源：${esc(srcName)}${run ? ' · ' + fmtTime(run.startedAt) : ''}</div>
    ${res ? (res.kind === 'file'
      ? `<div class="wp-file">📎 ${esc(res.file ? res.file.name : '')}（${fmtSize(res.file ? res.file.size : null)}）&nbsp;<a href="/api/runs/${esc(run.id)}/file?portId=${esc(res.portId)}" download>⬇ 下载</a></div>`
      : `<pre class="wp-text">${esc(res.preview || '')}${res.chars > (res.preview || '').length ? '\n…' : ''}</pre>`)
      : '<div class="wp-none">尚无成功成果（上游跑完才会产生）</div>'}`;
  wirePop.style.left = '0px';
  wirePop.style.top = '0px';
  const bw = wirePop.offsetWidth, bh = wirePop.offsetHeight;
  wirePop.style.left = Math.min(mx + 14, innerWidth - bw - 10) + 'px';
  wirePop.style.top = Math.min(my + 14, innerHeight - bh - 10) + 'px';
  wirePop.hidden = false;
}

/* ---------- SSE + 运行缓冲 ---------- */
const runBuffers = new Map();
const pollTimers = new Map();
function mkOutSpan(stream, text) {
  const sp = document.createElement('span');
  if (stream === 'err') sp.className = 'err';
  else if (stream === 'out') sp.className = 'ok';
  sp.textContent = text;
  return sp;
}
function getBuffer(runId) {
  let b = runBuffers.get(runId);
  if (!b) { b = { chunks: [], el: null, rendered: 0 }; runBuffers.set(runId, b); }
  return b;
}
function renderPending(b) {
  if (!b.el || !b.el.isConnected) return;
  while (b.rendered < b.chunks.length) {
    const c = b.chunks[b.rendered++];
    b.el.appendChild(mkOutSpan(c.s, c.d));
  }
  b.el.scrollTop = b.el.scrollHeight;
}
function appendChunkToBuffer(runId, stream, text) {
  const b = getBuffer(runId);
  b.chunks.push({ s: stream, d: text });
  if (b.chunks.length > 8000) {
    const cut = b.chunks.length - 8000;
    b.chunks.splice(0, cut);
    b.rendered = Math.max(0, b.rendered - cut);
  }
  renderPending(b);
}
function pollRunOutput(runId) {
  if (pollTimers.has(runId)) return;
  const timer = setInterval(async () => {
    try {
      const { run } = await api('/api/runs/' + runId);
      const b = getBuffer(runId);
      const serverChunks = run.output || [];
      if (serverChunks.length > b.chunks.length) {
        for (const c of serverChunks.slice(b.chunks.length)) b.chunks.push(c);
        renderPending(b);
      }
      if (['success', 'error', 'killed', 'aborted', 'skipped'].includes(run.status)) {
        clearInterval(timer);
        pollTimers.delete(runId);
      }
    } catch (e) { /* ignore */ }
  }, 900);
  pollTimers.set(runId, timer);
}

/* ---------- 节点渲染（模块卡片） ---------- */
function renderNodes() {
  if (nodeDrag) return;
  world.innerHTML = '';
  for (const t of currentTasks()) {
    const st = nodeStatusOf(t);
    const eng = engineOf(t.engine);

    const el = document.createElement('div');
    el.className = 'node' + (sel.nodeId === t.id ? ' selected' : '');
    el.dataset.id = t.id;
    el.style.left = t.x + 'px';
    el.style.top = t.y + 'px';
    el.style.setProperty('--c', t.color || eng.color || PALETTE[0]);

    el.innerHTML = `
      <div class="node-head">
        <span class="node-status ${esc(st)}"></span>
        <span class="node-title">${esc(t.name)}</span>
        ${t.group ? `<span class="group-chip">${esc(t.group)}</span>` : ''}
        <span class="engine-badge" style="background:${esc(eng.color || '#64748b')}">${esc((t.engine || '').toUpperCase())}</span>
      </div>
      <div class="node-body" style="height:${BODY_H}px">
        <div class="node-desc">${esc(t.description || '')}</div>
        <div class="node-command">${esc(t.command || '')}</div>
      </div>
      <div class="node-actions">
        <button class="a-run" data-act="run" title="立即启动">▶ 启动</button>
        <button class="a-edit" data-act="edit" title="模块 / 接口 / 定时 / 日志 / 成果">⚙ 详情</button>
        <button class="a-del" data-act="del" title="删除">✕</button>
      </div>
      <span class="iface-handle" title="拖到另一个模块 → 建立接口关系"></span>`;
    world.appendChild(el);
    bindNodeEvents(el, t);
  }
  layoutWires();
}

function bindNodeEvents(el, t) {
  el.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.iface-handle')) {
      e.preventDefault();
      e.stopPropagation();
      startWireDrag(t.id, e);
      return;
    }
    if (e.target.closest('button')) return;
    e.preventDefault();
    e.stopPropagation();
    selectNode(t.id);
    bringToFront(el);
    nodeDrag = { id: t.id, px: e.clientX, py: e.clientY, ox: t.x, oy: t.y, moved: false };
    el.classList.add('dragging');
    window.addEventListener('pointermove', onNodeMove);
    window.addEventListener('pointerup', onNodeUp);
    window.addEventListener('pointercancel', onNodeUp);
  });
  el.addEventListener('dblclick', (e) => {
    e.stopPropagation();
    openTaskPanel(t.id);
  });
}

/* 节点拖动（window 级监听） */
function onNodeMove(e) {
  if (!nodeDrag) return;
  const t = taskOf(nodeDrag.id);
  const el = world.querySelector('.node[data-id="' + nodeDrag.id + '"]');
  if (!t || !el) return;
  const dx = e.clientX - nodeDrag.px;
  const dy = e.clientY - nodeDrag.py;
  if (Math.abs(dx) + Math.abs(dy) > 3) nodeDrag.moved = true;
  t.x = Math.round(nodeDrag.ox + dx / view.scale);
  t.y = Math.round(nodeDrag.oy + dy / view.scale);
  el.style.left = t.x + 'px';
  el.style.top = t.y + 'px';
  layoutWires();
}
function onNodeUp() {
  if (!nodeDrag) return;
  window.removeEventListener('pointermove', onNodeMove);
  window.removeEventListener('pointerup', onNodeUp);
  window.removeEventListener('pointercancel', onNodeUp);
  const id = nodeDrag.id;
  const moved = nodeDrag.moved;
  const t = taskOf(id);
  const el = world.querySelector('.node[data-id="' + id + '"]');
  if (el) el.classList.remove('dragging');
  nodeDrag = null;
  if (moved && t) {
    api('/api/tasks/' + t.id, { method: 'POST', body: JSON.stringify({ x: t.x, y: t.y }) })
      .catch((err) => toast('保存位置失败：' + err.message, 'err'));
  }
}

function tryCapture(el2, id) { try { el2.setPointerCapture(id); } catch (e) { /* ignore */ } }
function tryRelease(el2, id) { try { el2.releasePointerCapture(id); } catch (e) { /* ignore */ } }
function selectNode(id) {
  sel.nodeId = id;
  sel.edgeId = null;
  world.querySelectorAll('.node').forEach((n) => n.classList.toggle('selected', n.dataset.id === id));
  layoutWires();
}
function bringToFront(el) {
  world.querySelectorAll('.node').forEach((n) => n.classList.remove('focused'));
  el.classList.add('focused');
}

/* ---------- 拖拽接口手柄 → 建立接口关系（edge） ---------- */
function startWireDrag(taskId, e) {
  wireDrag.active = true;
  wireDrag.fromTaskId = taskId;
  wireDrag.fromSide = 'out';
  wireDrag.fromPort = null;
  wireDrag.x = e.clientX;
  wireDrag.y = e.clientY;
  wireDrag.target = null;
  window.addEventListener('pointermove', onWireMove);
  window.addEventListener('pointerup', onWireUp);
  updateGhostWire();
  board.classList.add('panning');
}
function onWireMove(e) {
  if (!wireDrag.active) return;
  wireDrag.x = e.clientX;
  wireDrag.y = e.clientY;
  updateGhostWire();
  const el2 = document.elementFromPoint(e.clientX, e.clientY);
  const nodeEl = el2 && el2.closest ? el2.closest('.node') : null;
  world.querySelectorAll('.node.iface-hit').forEach((x) => x.classList.remove('iface-hit'));
  wireDrag.target = null;
  if (nodeEl && nodeEl.dataset.id !== wireDrag.fromTaskId) {
    wireDrag.target = { taskId: nodeEl.dataset.id };
    nodeEl.classList.add('iface-hit');
  }
}
function onWireUp() {
  window.removeEventListener('pointermove', onWireMove);
  window.removeEventListener('pointerup', onWireUp);
  board.classList.remove('panning');
  world.querySelectorAll('.node.iface-hit').forEach((x) => x.classList.remove('iface-hit'));
  if (!wireDrag.active) return;
  wireDrag.active = false;
  const g = wiresSvg.querySelector('path.ghost');
  if (g) g.remove();
  const target = wireDrag.target;
  if (!target) return;
  const from = wireDrag.fromTaskId;
  const to = target.taskId;
  api('/api/edges', { method: 'POST', body: JSON.stringify({ from, to, name: '新接口', kind: 'code' }) })
    .then(() => { toast('已建立接口关系：' + ((taskOf(from) || {}).name || '') + ' ↔ ' + ((taskOf(to) || {}).name || ''), 'ok'); return refresh(); })
    .catch((err) => toast('建立接口失败：' + err.message, 'err'));
}

wiresSvg.addEventListener('pointerdown', (e) => {
  const p = e.target.closest ? (e.target.closest('path.wire.hit') || e.target.closest('path.wire:not(.hit):not(.ghost)')) : null;
  if (!p) return;
  e.stopPropagation();
  hideWirePop();
  const edge = edgeOf(p.dataset.edge);
  sel.edgeId = p.dataset.edge || null;
  sel.nodeId = null;
  layoutWires();
  if (e.detail === 2) {
    deleteEdge(sel.edgeId);
    return;
  }
  if (edge) showWirePop(edge, e.clientX, e.clientY);
});
document.addEventListener('pointerdown', (e) => {
  if (!e.target.closest('#wire-pop')) hideWirePop();
});

function deleteEdge(id) {
  const e = edgeOf(id);
  api('/api/edges/' + id, { method: 'DELETE' })
    .then(() => {
      sel.edgeId = null;
      toast('已断开连线' + (e ? '（' + (e.fromName || '') + ' → ' + (e.toName || '') + '）' : ''), 'ok');
      return refresh();
    })
    .catch((err) => toast('断开失败：' + err.message, 'err'));
}

/* ---------- 节点动作 ---------- */
world.addEventListener('click', (e) => {
  const actBtn = e.target.closest('[data-act]');
  const node = e.target.closest('.node');
  if (!node) return;
  e.stopPropagation();
  const id = node.dataset.id;
  if (actBtn) {
    const act = actBtn.dataset.act;
    if (act === 'run') runTask(id);
    else if (act === 'edit') openTaskPanel(id);
    else if (act === 'del') deleteTask(id);
  }
});

async function runTask(id) {
  try {
    const data = await api('/api/tasks/' + id + '/run', { method: 'POST' });
    toast('已入队「' + ((taskOf(id) || {}).name || '') + '」，等待节点认领', 'ok');
    openRunId = data.runId;
    pollRunOutput(data.runId);
    await refresh();
    if (selectedTaskId === id) renderPanel();
  } catch (err) {
    toast('启动失败：' + err.message, 'err');
  }
}
async function deleteTask(id) {
  const t = taskOf(id);
  if (!confirm('确定删除节点「' + (t ? t.name : '') + '」？\n（其定时、连线与运行记录也会清理）')) return;
  try {
    await api('/api/tasks/' + id, { method: 'DELETE' });
    if (selectedTaskId === id) closePanel();
    if (sel.nodeId === id) sel.nodeId = null;
    toast('已删除', 'ok');
    await refresh();
  } catch (err) { toast('删除失败：' + err.message, 'err'); }
}
function openTaskPanel(id) {
  selectedTaskId = id;
  const t = taskOf(id);
  if (t) {
    formData.name = t.name;
    formData.engine = t.engine || 'v8';
    formData.color = t.color || engineOf(t.engine).color || PALETTE[0];
    formData.description = t.description || '';
    formData.command = t.command || '';
    formData.scope = t.scope || '';
    formData.spec = t.spec || '';
    formData.group = t.group || '';
    expandedPorts.clear();
    portData.inputs = portsIn(t).map((p) => ({
      id: p.id, name: p.name, spec: p.spec || '',
      tests: (p.tests || []).map((x) => ({ id: x.id, name: x.name, command: x.command }))
    }));
    portData.outputs = portsOut(t).map((p) => ({
      id: p.id, name: p.name, spec: p.spec || '',
      tests: (p.tests || []).map((x) => ({ id: x.id, name: x.name, command: x.command })),
      capture: p.capture || 'stdout', glob: p.glob || ''
    }));
  }
  $('#panel-title').textContent = t ? ('模块详情 · ' + t.name) : '模块详情';
  renderPanel();
  panel.classList.remove('collapsed');
}
function closePanel() {
  selectedTaskId = null;
  openRunId = null;
  panel.classList.add('collapsed');
}

/* ---------- 画布平移 ---------- */
let pan = null;
board.addEventListener('pointerdown', (e) => {
  if (e.target.closest('.node') || e.target.closest('.port') || e.target.closest('#panel') || e.target.closest('#wires path')) return;
  if (pinch) return;
  if (e.button !== 0 && e.button !== 1) return;
  e.preventDefault();
  pan = { sx: e.clientX, sy: e.clientY, ox: view.x, oy: view.y };
  tryCapture(board, e.pointerId);
  board.classList.add('panning');
});
board.addEventListener('pointermove', (e) => {
  if (!pan) return;
  view.x = pan.ox + (e.clientX - pan.sx);
  view.y = pan.oy + (e.clientY - pan.sy);
  applyView();
});
board.addEventListener('pointerup', (e) => {
  if (!pan) return;
  tryRelease(board, e.pointerId);
  pan = null;
  board.classList.remove('panning');
});
board.addEventListener('wheel', (e) => {
  e.preventDefault();
  zoomBy(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX, e.clientY);
}, { passive: false });

/* 触摸：单指拖动 = 平移（Pointer Events 已覆盖），双指 = 缩放 + 平移 */
let pinch = null;
function touchMid(t1, t2) {
  return { x: (t1.clientX + t2.clientX) / 2, y: (t1.clientY + t2.clientY) / 2 };
}
function touchDist(t1, t2) {
  const dx = t1.clientX - t2.clientX, dy = t1.clientY - t2.clientY;
  return Math.hypot(dx, dy);
}
board.addEventListener('touchstart', (e) => {
  if (e.touches.length === 2) {
    e.preventDefault();
    pan = null;
    board.classList.remove('panning');
    const mid = touchMid(e.touches[0], e.touches[1]);
    pinch = { dist: touchDist(e.touches[0], e.touches[1]), mx: mid.x, my: mid.y };
  }
}, { passive: false });
board.addEventListener('touchmove', (e) => {
  if (!pinch || e.touches.length < 2) return;
  e.preventDefault();
  const mid = touchMid(e.touches[0], e.touches[1]);
  const dist = touchDist(e.touches[0], e.touches[1]);
  const [wx, wy] = screenToWorld(pinch.mx, pinch.my);
  const f = pinch.dist > 0 && dist > 0 ? dist / pinch.dist : 1;
  view.x += mid.x - pinch.mx;
  view.y += mid.y - pinch.my;
  view.scale = Math.min(3.5, Math.max(0.15, view.scale * f));
  view.x = mid.x - wx * view.scale;
  view.y = mid.y - wy * view.scale;
  pinch.dist = dist;
  pinch.mx = mid.x;
  pinch.my = mid.y;
  applyView();
}, { passive: false });
board.addEventListener('touchend', (e) => {
  if (e.touches.length < 2) pinch = null;
});
board.addEventListener('touchcancel', () => { pinch = null; });
board.addEventListener('click', (e) => {
  hideWirePop();
  if (e.target.closest('.node') || e.target.closest('#wires path')) return;
  sel.nodeId = null;
  sel.edgeId = null;
  world.querySelectorAll('.node').forEach((n) => n.classList.remove('selected'));
  layoutWires();
});
board.addEventListener('dblclick', (e) => {
  if (e.target.closest('.node')) return;
  showCtxMenu(e.clientX, e.clientY, 'canvas');
});

/* ---------- 节点库显示/隐藏 ---------- */
function setLibraryVisible(visible) {
  const lib = $('#library');
  const btn = $('#btn-toggle-lib');
  if (!lib || !btn) return;
  lib.classList.toggle('hidden', !visible);
  btn.classList.toggle('off', !visible);
  btn.title = visible ? '隐藏节点库 (Ctrl+B)' : '显示节点库 (Ctrl+B)';
  try { localStorage.setItem('taskpad:libHidden', visible ? '0' : '1'); } catch (e) { /* ignore */ }
}
function toggleLibrary() {
  const lib = $('#library');
  if (!lib) return;
  setLibraryVisible(lib.classList.contains('hidden'));
}
(function initLibraryVisible() {
  let hidden = false;
  try { hidden = localStorage.getItem('taskpad:libHidden') === '1'; } catch (e) { /* ignore */ }
  setLibraryVisible(!hidden);
})();
$('#btn-toggle-lib').addEventListener('click', toggleLibrary);

/* ---------- 导入架构编辑器 JSON ---------- */
async function importArchitecture(data) {
  const nodes = Array.isArray(data && data.nodes) ? data.nodes : [];
  if (!nodes.length) throw new Error('JSON 中没有 nodes 数组');
  const res = await api('/api/import/architecture', {
    method: 'POST',
    body: JSON.stringify({ project: data.project, nodes: data.nodes, edges: data.edges || [] })
  });
  toast('已导入「' + res.graph.name + '」：' + res.nodes + ' 节点 / ' + res.edges + ' 连线' + (res.skipped ? '（跳过 ' + res.skipped + '）' : ''), 'ok');
  await refresh();
  enterGraph(res.graph.id);
}
$('#btn-import-arch').addEventListener('click', () => $('#import-arch-input').click());
$('#import-arch-input').addEventListener('change', async (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  try {
    const data = JSON.parse(await f.text());
    await importArchitecture(data);
  } catch (err) {
    toast('导入失败：' + err.message, 'err');
  }
  e.target.value = '';
});

$('#btn-zoom-in').addEventListener('click', () => zoomBy(1.25, innerWidth / 2, innerHeight / 2));
$('#btn-zoom-out').addEventListener('click', () => zoomBy(1 / 1.25, innerWidth / 2, innerHeight / 2));
$('#btn-reset').addEventListener('click', fitView);
$('#btn-close-panel').addEventListener('click', closePanel);

/* ---------- 运行图 / 队列 / 清空 ---------- */
$('#btn-run-graph').addEventListener('click', async () => {
  try {
    const r = await api('/api/graph/run', { method: 'POST', body: JSON.stringify({ graphId: state.activeGraphId }) });
    toast('运行图已入队 ' + r.order + ' 个任务' + (r.cycle ? '（存在环，已尽量排序）' : ''), 'ok');
    await refresh();
  } catch (err) { toast('入队失败：' + err.message, 'err'); }
});
$('#btn-stop-graph').addEventListener('click', async () => {
  try {
    await api('/api/graph/stop', { method: 'POST' });
    toast('已停止队列', 'ok');
    await refresh();
  } catch (err) { toast(err.message, 'err'); }
});
$('#btn-clear-runs').addEventListener('click', async () => {
  if (!confirm('清空全部运行记录？（不影响任务与定时）')) return;
  try {
    await api('/api/runs/clear', { method: 'POST' });
    runBuffers.clear();
    toast('已清空运行记录', 'ok');
    await refresh();
  } catch (err) { toast('清空失败：' + err.message, 'err'); }
});
function updateQueueLabel() {
  const q = state.queue || {};
  const label = $('#queue-label');
  if (q.running) {
    const cur = q.currentTaskId ? ((taskOf(q.currentTaskId) || {}).name || '…') : '';
    label.textContent = '⏳ 运行图 ' + (q.remaining + (cur ? 1 : 0)) + ' · 当前: ' + cur;
    label.hidden = false;
  } else {
    label.hidden = true;
  }
}

/* ---------- 节点库 ---------- */
async function loadTemplates() {
  try {
    const { templates } = await api('/api/templates');
    state.templates = templates || [];
    renderLibrary();
  } catch (e) { /* ignore */ }
}
function renderLibrary() {
  const q = ($('#lib-search').value || '').trim().toLowerCase();
  const groups = {};
  for (const t of state.templates) {
    const eng = engineOf(t.engine);
    const key = eng.label || (t.engine || '').toUpperCase();
    if (q && !(t.name + ' ' + (t.command || '') + ' ' + (t.description || '')).toLowerCase().includes(q)) continue;
    (groups[key] = groups[key] || []).push(t);
  }
  $('#lib-groups').innerHTML = Object.keys(groups).map((gk) => {
    const sample = groups[gk][0];
    const eng = engineOf(sample.engine);
    return `
      <div class="lib-group-title"><span class="gdot" style="background:${esc(eng.color)}"></span>${esc(gk)}</div>
      ${groups[gk].map((t) => `
        <div class="lib-item" draggable="true" data-tpl="${esc(t.name)}" style="--c:${esc(t.color || eng.color)}">
          <span class="li-tag engine-badge" style="background:${esc(eng.color)}">${esc((t.engine || '').toUpperCase())}</span>
          <div class="li-name">${esc(t.name)}</div>
          <div class="li-cmd">${esc(t.command || '')}</div>
        </div>`).join('')}`;
  }).join('') || '<div class="no-runs">没有匹配的模板</div>';
}
$('#lib-search').addEventListener('input', renderLibrary);

async function addTaskFromTemplate(tpl, x, y) {
  const base = tpl.name;
  let name = base;
  let n = 2;
  while (state.tasks.some((t) => t.name === name)) name = base + ' #' + (n++);
  try {
    await api('/api/tasks', {
      method: 'POST',
      body: JSON.stringify({
        name, engine: tpl.engine || 'v8', color: tpl.color, description: tpl.description || '',
        command: tpl.command || '', x, y,
        graphId: state.activeGraphId,
        scope: '', spec: '',
        inputs: [{ id: 'in', name: 'dep', spec: '', tests: [] }],
        outputs: [{ id: 'out', name: 'out', capture: 'stdout', spec: '', tests: [] }]
      })
    });
    toast('已添加「' + name + '」', 'ok');
    await refresh();
  } catch (err) { toast('添加失败：' + err.message, 'err'); }
}
function centerPos() {
  const [wx, wy] = screenToWorld(innerWidth / 2 - 150, innerHeight / 2 - 60);
  return [Math.round(wx) + Math.round(Math.random() * 80), Math.round(wy) + Math.round(Math.random() * 50)];
}
function addCustomNode() {
  addTaskFromTemplate({ name: '自定义节点', engine: 'v8', color: PALETTE[0], description: '输入要执行的命令', command: 'echo "hello"' }, ...centerPos());
}
$('#lib-groups').addEventListener('click', (e) => {
  const item = e.target.closest('.lib-item');
  if (!item) return;
  const tpl = state.templates.find((t) => t.name === item.dataset.tpl);
  if (!tpl) return;
  addTaskFromTemplate(tpl, ...centerPos());
});
$('#btn-new-custom').addEventListener('click', addCustomNode);

/* ---------- 从节点库拖到画布 ---------- */
const DND_TPL = 'application/x-taskpad-tpl';
let draggingTpl = null;
let dropHint = null;

function showDropHint(cx, cy) {
  if (!dropHint) {
    dropHint = document.createElement('div');
    dropHint.className = 'drop-hint';
    document.body.appendChild(dropHint);
  }
  const tpl = state.templates.find((t) => t.name === draggingTpl);
  dropHint.innerHTML = '<span class="dh-plus">＋</span>' + esc(draggingTpl || '')
    + (tpl && tpl.engine ? ' <em>' + esc(tpl.engine.toUpperCase()) + '</em>' : '');
  dropHint.style.transform = 'translate(' + (cx + 16) + 'px,' + (cy + 10) + 'px)';
  dropHint.hidden = false;
}

function hideDropHint() {
  if (dropHint) dropHint.hidden = true;
  board.classList.remove('drop-active');
  document.body.classList.remove('dnd-lib');
}

$('#lib-groups').addEventListener('dragstart', (e) => {
  const item = e.target.closest('.lib-item');
  if (!item) return;
  draggingTpl = item.dataset.tpl;
  try {
    e.dataTransfer.setData(DND_TPL, draggingTpl);
    e.dataTransfer.setData('text/plain', draggingTpl); // 兜底：跨浏览器/跨窗口
    e.dataTransfer.effectAllowed = 'copy';
  } catch (err) { /* ignore */ }
  item.classList.add('dragging');
  document.body.classList.add('dnd-lib');
});

$('#lib-groups').addEventListener('dragend', () => {
  const d = $('#lib-groups').querySelector('.lib-item.dragging');
  if (d) d.classList.remove('dragging');
  draggingTpl = null;
  hideDropHint();
});

// 只在「从节点库拖出来」时接管画布的拖放
board.addEventListener('dragover', (e) => {
  if (!draggingTpl) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
  board.classList.add('drop-active');
  showDropHint(e.clientX, e.clientY);
});

board.addEventListener('dragleave', (e) => {
  if (!draggingTpl) return;
  if (!e.relatedTarget || !board.contains(e.relatedTarget)) hideDropHint();
});

board.addEventListener('drop', async (e) => {
  if (!draggingTpl) return;
  e.preventDefault();
  const name = (e.dataTransfer.getData(DND_TPL) || draggingTpl || '');
  const tpl = state.templates.find((t) => t.name === name);
  const cx = e.clientX;
  const cy = e.clientY;
  hideDropHint();
  draggingTpl = null;
  if (!tpl) return;
  // 落点换算成画布世界坐标；节点宽 300px，按「鼠标落在节点中心」摆放
  const [wx, wy] = screenToWorld(cx, cy);
  await addTaskFromTemplate(tpl, Math.round(wx) - 150, Math.round(wy) - 45);
});

/* ---------- 右键菜单 ---------- */
const ctx = $('#ctxmenu');
function showCtxMenu(x, y, kind, payload) {
  let html = '';
  if (kind === 'canvas') {
    html = `
      <div class="cm-item" data-ctx="add-v8">＋ 新建 V8 节点</div>
      <div class="cm-item" data-ctx="add-v9">＋ 新建 V9 节点</div>
      <div class="cm-item" data-ctx="add-custom">＋ 自定义空白节点</div>
      <div class="cm-sep"></div>
      <div class="cm-item" data-ctx="fit">⌂ 适配全部节点</div>
      <div class="cm-item" data-ctx="clear-runs">清空运行记录</div>`;
  } else if (kind === 'node') {
    html = `
      <div class="cm-item" data-ctx="run-node">▶ 运行此节点<span class="cm-kbd">R</span></div>
      <div class="cm-item" data-ctx="edit-node">⚙ 端口 / 定时 / 日志 / 成果</div>
      <div class="cm-item" data-ctx="dup-node">⧉ 复制节点<span class="cm-kbd">Ctrl+D</span></div>
      <div class="cm-sep"></div>
      <div class="cm-item danger" data-ctx="del-node">✕ 删除节点<span class="cm-kbd">Del</span></div>`;
  } else if (kind === 'wire') {
    html = `<div class="cm-item danger" data-ctx="del-wire">✕ 删除连线</div>`;
  }
  ctx.innerHTML = html;
  ctx.style.left = Math.min(x, innerWidth - 220) + 'px';
  ctx.style.top = Math.min(y, innerHeight - 220) + 'px';
  ctx.hidden = false;
  ctx._payload = payload;
}
document.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  const nodeEl = e.target.closest('.node');
  if (nodeEl) { showCtxMenu(e.clientX, e.clientY, 'node', nodeEl.dataset.id); return; }
  const wireEl = e.target.closest('path.wire.hit') || e.target.closest('path.wire:not(.ghost)');
  if (wireEl) { showCtxMenu(e.clientX, e.clientY, 'wire', wireEl.dataset.edge); return; }
  showCtxMenu(e.clientX, e.clientY, 'canvas', null);
});
ctx.addEventListener('click', (e) => {
  const item = e.target.closest('.cm-item');
  ctx.hidden = true;
  if (!item) return;
  const act = item.dataset.ctx;
  if (act === 'add-v8') addTaskFromTemplate({ name: 'V8 空白任务', engine: 'v8', color: '#3b82f6', description: '', command: '' }, ...centerPos());
  else if (act === 'add-v9') addTaskFromTemplate({ name: 'V9 空白任务', engine: 'v9', color: '#a855f7', description: '', command: '' }, ...centerPos());
  else if (act === 'add-custom') addCustomNode();
  else if (act === 'fit') fitView();
  else if (act === 'clear-runs') $('#btn-clear-runs').click();
  else if (act === 'run-node') runTask(ctx._payload);
  else if (act === 'edit-node') openTaskPanel(ctx._payload);
  else if (act === 'del-node') deleteTask(ctx._payload);
  else if (act === 'dup-node') duplicateNode(ctx._payload);
  else if (act === 'del-wire') deleteEdge(ctx._payload);
});
document.addEventListener('click', (e) => { if (!e.target.closest('#wire-pop')) { ctx.hidden = true; } });
async function duplicateNode(id) {
  const t = taskOf(id);
  if (!t) return;
  await addTaskFromTemplate({ name: t.name, engine: t.engine, color: t.color, description: t.description, command: t.command }, t.x + 50, t.y + 50);
}

/* ---------- 键盘 ---------- */
document.addEventListener('keydown', (e) => {
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT')) {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  if (e.key === 'Escape') {
    if (wireDrag.active) {
      wireDrag.active = false;
      window.removeEventListener('pointermove', onWireMove);
      window.removeEventListener('pointerup', onWireUp);
      board.classList.remove('panning');
      const g = wiresSvg.querySelector('path.ghost');
      if (g) g.remove();
      document.querySelectorAll('.port.hit').forEach((x) => x.classList.remove('hit'));
    }
    sel.nodeId = null; sel.edgeId = null;
    layoutWires();
    world.querySelectorAll('.node').forEach((n) => n.classList.remove('selected'));
    ctx.hidden = true;
    hideWirePop();
    $('#trace-modal').hidden = true;
    return;
  }
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (sel.edgeId) deleteEdge(sel.edgeId);
    else if (sel.nodeId) deleteTask(sel.nodeId);
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
    e.preventDefault();
    toggleLibrary();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') {
    e.preventDefault();
    if (sel.nodeId) duplicateNode(sel.nodeId);
    return;
  }
  if (e.key.toLowerCase() === 'r' && sel.nodeId) runTask(sel.nodeId);
});

/* ---------- 状态同步 ---------- */
function applyState(payload) {
  state.tasks = (payload.tasks || []).map((t) => Object.assign({ inputs: [], outputs: [], deps: [], scope: '', spec: '', subgraphId: null }, t));
  state.edges = payload.edges || [];
  state.graphs = payload.graphs || [];
  state.schedules = payload.schedules || [];
  state.runs = payload.runs || [];
  state.engines = payload.engines || state.engines;
  state.queue = payload.queue || state.queue;
  state.workers = payload.workers || state.workers || [];
  if (!state.engines.length) {
    state.engines = [
      { id: 'v8', label: 'V8', color: '#3b82f6' },
      { id: 'v9', label: 'V9', color: '#a855f7' }
    ];
  }
  if (!state.activeGraphId || !graphById(state.activeGraphId)) {
    state.activeGraphId = rootGraphId();
  }
  if (!fitDone) { fitDone = true; fitView(); }
  if (state.templates.length && !state.libWithEngines) { state.libWithEngines = true; renderLibrary(); }
  renderGraphBar();
  renderNodes();
  if (!panel.classList.contains('collapsed')) renderPanel();
  updateQueueLabel();
  updateWorkersLabel();
}

function updateWorkersLabel() {
  const el = $('#workers-label');
  if (!el) return;
  const list = state.workers || [];
  const now = Date.now();
  const online = list.filter((w) => w.lastSeen && now - new Date(w.lastSeen).getTime() < 90000);
  if (!list.length) {
    el.textContent = '🖥 无节点';
    el.className = 'workers-label off';
    return;
  }
  el.textContent = '🖥 节点 ' + online.length + (list.length > online.length ? ' / ' + list.length : '');
  el.className = 'workers-label ' + (online.length ? 'on' : 'off');
  el.title = list.map((w) => w.id + '（' + fmtTime(w.lastSeen) + '）').join('\n');
}
function handleSSE(msg) {
  const d = msg.data;
  if (!d || !d.type) return;
  if (d.type === 'state') { applyState(d.data); return; }
  if (d.type === 'queue') { state.queue = d.data || state.queue; updateQueueLabel(); renderNodes(); return; }
  if (d.type === 'run:start') {
    if (!state.runs.find((r) => r.id === d.run.id)) {
      state.runs.unshift(d.run);
      if (state.runs.length > 120) state.runs.length = 120;
    }
    renderNodes();
    if (!panel.classList.contains('collapsed')) renderPanel();
    return;
  }
  if (d.type === 'run:update') {
    const i = state.runs.findIndex((r) => r.id === d.run.id);
    if (i >= 0) state.runs[i] = d.run; else state.runs.unshift(d.run);
    renderNodes();
    if (!panel.classList.contains('collapsed')) renderPanel();
    return;
  }
  if (d.type === 'run:output') { appendChunkToBuffer(d.runId, d.stream, d.text); return; }
  if (d.type === 'run:end') {
    const i = state.runs.findIndex((r) => r.id === d.run.id);
    if (i >= 0) state.runs[i] = d.run; else state.runs.unshift(d.run);
    renderNodes();
    if (!panel.classList.contains('collapsed')) renderPanel();
  }
}
function connectSSE() {
  const es = new EventSource('/api/events');
  es.onopen = () => { $('#sse-dot').classList.add('on'); refresh(); };
  es.onerror = () => $('#sse-dot').classList.remove('on');
  es.onmessage = (e) => {
    try { handleSSE(JSON.parse(e.data)); } catch (err) { /* ignore */ }
  };
}
async function refresh() {
  try {
    const payload = await (await fetch('/api/state')).json();
    applyState(payload);
  } catch (err) { /* ignore */ }
}

/* ---------- 面板 ---------- */
function renderPanel() {
  renderEditPanel();
  bindPanelEvents();
  relinkOutput(openRunId);
}
function testBadge(pr) {
  if (!pr.tested) return '<span class="test-badge untested">未设测试</span>';
  return pr.valid ? '<span class="test-badge ok">✓ 测试通过</span>' : '<span class="test-badge bad">✗ 测试未通过</span>';
}
function ifaceTestEditorHTML(e, listIndex) {
  const tests = e.tests || [];
  return `
    <div class="edge-tests">
      ${tests.map((t, ti) => `
        <div class="test-row">
          <input type="text" data-etest-name="${esc(e.id)}:${ti}" value="${esc(t.name || '')}" placeholder="测试名">
          <input type="text" data-etest-cmd="${esc(e.id)}:${ti}" value="${esc(t.command || '')}" placeholder="shell 测试命令">
          <button data-del-etest="${esc(e.id)}:${ti}" title="删除测试">✕</button>
        </div>`).join('')}
      <button class="btn ghost block" data-add-etest="${esc(e.id)}">＋ 添加接口测试</button>
    </div>`;
}

function renderEditPanel() {
  const t = taskOf(selectedTaskId);
  if (!t) { closePanel(); return; }
  const scheds = state.schedules.filter((s) => s.taskId === t.id);
  const runs = state.runs.filter((r) => r.taskId === t.id);
  const st = nodeStatusOf(t);
  const eng = engineOf(t.engine);
  const subGraph = t.subgraphId ? graphById(t.subgraphId) : null;
  const edges = state.edges.filter((e) => e.from === t.id || e.to === t.id);
  const lastRun = [...runs].find((r) => r.status === 'success' || (r.portResults && r.portResults.length));

  panelBody.innerHTML = `
    <div class="section-title">
      <span class="node-status ${esc(st)}" style="display:inline-block"></span> ${esc(t.name)}
      <span class="engine-badge" style="background:${esc(eng.color || '#64748b')};float:right">${esc((t.engine || '').toUpperCase())}</span>
    </div>

    <button class="btn primary block big" data-bigrun="1">▶ 启动模块任务（等待节点认领）</button>

    <div class="section-title">📋 模块契约</div>
    <div class="field"><label>工作范围（agent 可操作范围 / 边界）</label>
      <textarea id="f-scope" placeholder="例如：仅读取该模块目录；只运行只读命令">${esc(formData.scope)}</textarea></div>
    <div class="field"><label>工作规范（agent 必须遵守的作业要求）</label>
      <textarea id="f-spec" placeholder="例如：输出 JSON 汇总；失败需在 stderr 说明">${esc(formData.spec)}</textarea></div>

    <div class="section-title">🧩 内部流程图</div>
    ${subGraph
      ? `<button class="btn block" id="btn-enter-subgraph">进入内部流程图（${esc(subGraph.name)}）</button>`
      : `<button class="btn ghost block" id="btn-create-subgraph">创建内部流程图</button>
         <div class="hint">模块内部的流程用子图描述（流程图），和模块间接口是两层。</div>`}

    <div class="section-title">🔗 接口关系（模块 ↔ 模块，edge）</div>
    <div class="hint" style="margin-bottom:6px">接口 = 两个模块之间的关系；接口测试属于接口本身。拖模块右侧圆形手柄到另一个模块即可建立关系。</div>
    ${edges.length ? edges.map((e) => {
      const dir = e.from === t.id ? 'provide' : 'use';
      const other = taskOf(e.from === t.id ? e.to : e.from);
      return `
      <div class="edge-editor" data-edge-editor="${esc(e.id)}">
        <div class="sched-item" style="margin-bottom:5px">
          <span class="s-chip">${dir === 'provide' ? '本模块 ↔ ' : '← '}${esc(other ? other.name : '?')} <b>${dir === 'provide' ? '（本模块提供）' : '（本模块使用）'}</b></span>
          <button class="sched-del" data-edge-del="${esc(e.id)}" title="删除接口关系">✕</button>
        </div>
        <div class="ee-row">
          <input data-edge-name value="${esc(e.name || e.label || '')}" placeholder="接口名">
          <select data-edge-kind>
            <option value="code" ${e.kind === 'code' ? 'selected' : ''}>代码</option>
            <option value="data" ${e.kind === 'data' ? 'selected' : ''}>数据</option>
            <option value="net" ${e.kind === 'net' ? 'selected' : ''}>网络</option>
          </select>
          <input data-edge-proto value="${esc(e.proto || '')}" placeholder="协议" style="display:${e.kind === 'net' ? '' : 'none'}">
          <input data-edge-auth value="${esc(e.auth || '')}" placeholder="鉴权" style="display:${e.kind === 'net' ? '' : 'none'}">
        </div>
        <textarea data-edge-spec rows="2" placeholder="接口规范：这个接口必须满足什么（格式/约束/测试条件）">${esc(e.spec || '')}</textarea>
        ${ifaceTestEditorHTML(e)}
      </div>`;
    }).join('') : '<div class="no-runs">暂无接口关系：拖模块右侧手柄到另一个模块</div>'}

    <div class="section-title">📦 产出成果（可选）</div>
    <div class="hint" style="margin-bottom:6px">模块执行后可产出成果（stdout 文本或匹配文件），供成果卡与下游展示。</div>
    <div id="pout-list">
      ${portData.outputs.map((p, i) => `
        <div class="port-edit">
          <span class="pe-dot out"></span>
          <input type="text" data-pout="${i}" value="${esc(p.name)}" style="flex:1">
          <select data-pout-cap="${i}">
            <option value="stdout" ${p.capture === 'stdout' ? 'selected' : ''}>stdout</option>
            <option value="file" ${p.capture === 'file' ? 'selected' : ''}>文件</option>
          </select>
          <input type="text" data-pout-glob="${i}" value="${esc(p.glob || '*')}" placeholder="glob" style="width:58px;display:${p.capture === 'file' ? '' : 'none'}">
          <button class="pe-del" data-del-pout="${i}" title="删除">✕</button>
        </div>`).join('') || '<div class="no-runs">无</div>'}
    </div>
    <button class="btn ghost block" id="btn-add-pout">＋ 添加产出</button>
    <button class="btn block" id="f-save-ports">保存产出</button>

    <div class="section-title">🧾 成果卡片（最近运行）</div>
    <div id="result-cards">
      ${lastRun && lastRun.portResults && lastRun.portResults.length
        ? lastRun.portResults.map((pr) => `
          <div class="result-card">
            <div class="rc-head">
              <span class="rc-name">${esc(pr.name)}</span>
              <span class="rc-kind ${esc(pr.kind)}">${pr.kind === 'file' ? '📎 文件' : '📄 文本'}</span>
            </div>
            ${pr.kind === 'file'
              ? `<div class="rc-file">${esc(pr.file ? pr.file.name : '')}（${fmtSize(pr.file ? pr.file.size : null)}）
                  <a href="/api/runs/${esc(lastRun.id)}/file?portId=${esc(pr.portId)}" download>⬇ 下载</a></div>`
              : `<pre class="rc-text">${esc(pr.preview || '')}${pr.chars > (pr.preview || '').length ? '\n…' : ''}</pre>`}
            <div class="rc-time">${fmtTime(lastRun.startedAt)} · exit=${lastRun.exitCode}</div>
          </div>`).join('')
        : '<div class="no-runs">运行成功后这里出现产出成果卡</div>'}
    </div>

    <div class="section-title">编辑信息</div>
    <div class="field"><label>模块名称</label>
      <input type="text" id="f-name" value="${esc(formData.name)}"></div>
    <div class="field"><label>执行引擎（V8 / V9）</label>
      <select id="f-engine">
        ${state.engines.map((e2) => `<option value="${esc(e2.id)}" ${formData.engine === e2.id ? 'selected' : ''}>${esc(e2.label)}</option>`).join('')}
      </select></div>
    <div class="field"><label>模块分组</label>
      <input type="text" id="f-group" value="${esc(formData.group || '')}" placeholder="core / adapter / script / config"></div>
    <div class="field"><label>颜色</label>
      <div class="swatches">${PALETTE.map((c) => `<span class="swatch" data-c="${c}" style="background:${c}${formData.color === c ? ';outline:2px solid #fff;outline-offset:2px' : ''}"></span>`).join('')}</div></div>
    <div class="field"><label>模块描述</label>
      <textarea id="f-desc">${esc(formData.description)}</textarea></div>
    <div class="field"><label>执行命令</label>
      <textarea id="f-cmd" class="cmd">${esc(formData.command)}</textarea></div>
    <button class="btn block" id="f-save">保存修改</button>
    <button class="btn danger block" id="f-del">删除节点</button>

    <div class="section-title">⏰ 定时任务</div>
    <div id="sched-list">
      ${scheds.length ? scheds.map(schedRowHTML).join('') : '<div class="no-runs">尚未设置定时</div>'}
    </div>
    <div class="field" style="margin-top:8px"><label>新定时（基于此模块）</label>
      <select id="s-type">
        ${['interval', 'daily', 'cron'].map((tp) => `<option value="${tp}" ${schedData.type === tp ? 'selected' : ''}>${{ interval: '间隔（每 N 分钟）', daily: '每日（HH:MM）', cron: 'cron 表达式' }[tp]}</option>`).join('')}
      </select></div>
    <div id="s-extra" class="field"></div>
    <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);margin-bottom:8px">
      <input type="checkbox" id="s-enabled" ${schedData.enabled ? 'checked' : ''}> 启用该定时
    </label>
    <button class="btn block" id="s-save">保存定时</button>

    <div class="section-title">运行记录（前 ${Math.min(runs.length, 30)} 条）</div>
    <div class="runs-list" id="runs-list">
      ${runs.length ? runs.slice(0, 30).map(runRowHTML).join('') : '<div class="no-runs">还没有运行记录</div>'}
    </div>
  `;
}

function schedRowHTML(s) {
  return `
    <div class="sched-item ${s.enabled ? 'on' : ''}">
      <span class="s-chip">${esc(s.label)}${s.enabled ? '' : '（停用）'}</span>
      <span class="sched-next">下次 ${s.enabled ? fmtNext(s.nextAt) : '—'}</span>
      <button class="sched-del" data-sched-toggle="${s.id}" title="启停">${s.enabled ? '⏸' : '▶'}</button>
      <button class="sched-del" data-sched-del="${s.id}" title="删除">✕</button>
    </div>`;
}
function runRowHTML(r) {
  const dur = r.durationMs != null ? ' · ' + fmtDur(r.durationMs) : '';
  const exit = r.exitCode != null ? `<span class="r-exit ${r.exitCode === 0 ? 'good' : 'bad'}">exit=${r.exitCode}</span>` : '';
  return `
    <div class="run-item" data-run="${esc(r.id)}">
      <div class="r-top">
        <span class="r-dot ${esc(r.status)}"></span>
        <span class="r-label">${esc(r.triggerLabel || '手动')}</span>
        <span class="r-time">${fmtTime(r.startedAt)}</span>
      </div>
      <div class="r-sub"><span>${esc(statusText(r.status))}</span>${dur}${exit}${r.worker ? `<span class="r-worker">🖥 ${esc(r.worker)}</span>` : ''}</div>
    </div>`;
}
async function toggleRunOutput(runId) {
  const itemEl = panelBody.querySelector(`[data-run="${runId}"]`);
  if (!itemEl) return;
  const existing = panelBody.querySelector('#out-' + runId);
  if (existing) { existing.remove(); if (openRunId === runId) openRunId = null; return; }
  openRunId = runId;
  const b = getBuffer(runId);
  const wrap = document.createElement('div');
  wrap.id = 'out-' + runId;
  const pre = document.createElement('div');
  pre.className = 'run-output';
  wrap.appendChild(pre);
  itemEl.after(wrap);
  b.el = pre;
  b.rendered = 0;
  if (!b.chunks.length) {
    try {
      const { run } = await api('/api/runs/' + runId);
      b.chunks = run.output || [];
    } catch (e) { /* ignore */ }
  }
  renderPending(b);
  pre.scrollTop = pre.scrollHeight;
}
function relinkOutput(runId) {
  if (!runId) return;
  const itemEl = panelBody.querySelector(`[data-run="${runId}"]`);
  if (!itemEl) return;
  if (panelBody.querySelector('#out-' + runId)) return;
  const b = getBuffer(runId);
  const wrap = document.createElement('div');
  wrap.id = 'out-' + runId;
  const pre = document.createElement('div');
  pre.className = 'run-output';
  wrap.appendChild(pre);
  itemEl.after(wrap);
  b.el = pre;
  b.rendered = 0;
  renderPending(b);
  pre.scrollTop = pre.scrollHeight;
}

/* ---------- 面板事件 ---------- */
$('#panel-body').addEventListener('click', (e) => {
  const runEl = e.target.closest('[data-run]');
  if (runEl) { toggleRunOutput(runEl.dataset.run); return; }
  const sDel = e.target.closest('[data-sched-del]');
  if (sDel) { deleteSchedule(sDel.dataset.schedDel); return; }
  const sToggle = e.target.closest('[data-sched-toggle]');
  if (sToggle) { toggleSchedule(sToggle.dataset.schedToggle); return; }
  const edgeDel = e.target.closest('[data-edge-del]');
  if (edgeDel) { deleteEdge(edgeDel.dataset.edgeDel); return; }
  const addTest = e.target.closest('[data-add-etest]');
  if (addTest) {
    const eid = addTest.dataset.addEtest;
    const ed = panelBody.querySelector('[data-edge-editor="' + eid + '"]');
    if (ed) {
      const wrap = ed.querySelector('.edge-tests');
      const row = document.createElement('div');
      row.className = 'test-row';
      row.innerHTML = `<input type="text" data-etest-name="${esc(eid)}:new" value="测试" placeholder="测试名">
        <input type="text" data-etest-cmd="${esc(eid)}:new" value="" placeholder="shell 命令">
        <button data-del-etest="${esc(eid)}:new" title="删除测试">✕</button>`;
      const btn = wrap.querySelector('[data-add-etest]');
      wrap.insertBefore(row, btn);
    }
    return;
  }
  const delTest = e.target.closest('[data-del-etest]');
  if (delTest) {
    const row = delTest.closest('.test-row');
    if (row) row.remove();
    return;
  }
  const delPout = e.target.closest('[data-del-pout]');
  if (delPout) { portData.outputs.splice(parseInt(delPout.dataset.delPout, 10), 1); renderPanel(); return; }
  const addPout = e.target.closest('#btn-add-pout');
  if (addPout) { portData.outputs.push({ id: 'p' + Math.random().toString(36).slice(2, 8), name: '产出' + (portData.outputs.length + 1), capture: 'stdout', glob: '*' }); renderPanel(); return; }
});
$('#panel-body').addEventListener('input', (e) => {
  const pout = e.target.closest('[data-pout]');
  if (pout) { portData.outputs[parseInt(pout.dataset.pout, 10)].name = pout.value; return; }
  const glob = e.target.closest('[data-pout-glob]');
  if (glob) { portData.outputs[parseInt(glob.dataset.poutGlob, 10)].glob = glob.value; return; }
});
$('#panel-body').addEventListener('change', (e) => {
  const cap = e.target.closest('[data-pout-cap]');
  if (cap) {
    const o = portData.outputs[parseInt(cap.dataset.poutCap, 10)];
    o.capture = cap.value;
    renderPanel();
    return;
  }
  const ed = e.target.closest('[data-edge-editor]');
  if (ed) {
    const kind = ed.querySelector('[data-edge-kind]').value;
    const protoEl = ed.querySelector('[data-edge-proto]');
    const authEl = ed.querySelector('[data-edge-auth]');
    if (protoEl) protoEl.style.display = kind === 'net' ? '' : 'none';
    if (authEl) authEl.style.display = kind === 'net' ? '' : 'none';
    saveEdgeEditor(ed);
  }
});

function saveEdgeEditor(ed) {
  const val = (sel) => { const el = ed.querySelector(sel); return el ? el.value : ''; };
  const tests = [...ed.querySelectorAll('.test-row')].map((row) => {
    const n = row.querySelector('[data-etest-name]');
    const c = row.querySelector('[data-etest-cmd]');
    return { name: n ? n.value : '', command: c ? c.value : '' };
  }).filter((t) => String(t.command).trim());
  api('/api/edges/' + ed.dataset.edgeEditor, {
    method: 'POST',
    body: JSON.stringify({
      name: val('[data-edge-name]'),
      label: val('[data-edge-name]'),
      kind: val('[data-edge-kind]') || 'code',
      proto: val('[data-edge-proto]'),
      auth: val('[data-edge-auth]'),
      spec: val('[data-edge-spec]'),
      tests
    })
  }).then(() => toast('接口已保存', 'ok')).catch((err) => toast('保存接口失败：' + err.message, 'err'));
}

async function deleteSchedule(id) {
  try {
    await api('/api/schedules/' + id, { method: 'DELETE' });
    toast('已删除定时', 'ok');
    await refresh();
  } catch (err) { toast(err.message, 'err'); }
}
async function toggleSchedule(id) {
  const s = state.schedules.find((x) => x.id === id);
  if (!s) return;
  try {
    await api('/api/schedules/' + id, { method: 'POST', body: JSON.stringify({ enabled: !s.enabled }) });
    toast(s.enabled ? '已停用' : '已启用', 'ok');
    await refresh();
  } catch (err) { toast(err.message, 'err'); }
}

function bindPanelEvents() {
  const body = panelBody;
  const onInput = () => {
    const v = (id) => { const el = body.querySelector(id); return el ? el.value : null; };
    if (v('#f-name') != null) formData.name = v('#f-name');
    if (v('#f-engine') != null) formData.engine = v('#f-engine');
    if (v('#f-desc') != null) formData.description = v('#f-desc');
    if (v('#f-cmd') != null) formData.command = v('#f-cmd');
    if (v('#f-scope') != null) formData.scope = v('#f-scope');
    if (v('#f-spec') != null) formData.spec = v('#f-spec');
    if (v('#f-group') != null) formData.group = v('#f-group');
  };
  ['#f-name', '#f-engine', '#f-desc', '#f-cmd', '#f-scope', '#f-spec', '#f-group'].forEach((id) => {
    const el = body.querySelector(id);
    if (el) el.addEventListener('input', onInput);
  });
  body.querySelectorAll('.swatch').forEach((sw) => {
    sw.addEventListener('click', () => {
      formData.color = sw.dataset.c;
      body.querySelectorAll('.swatch').forEach((x) => { x.style.outline = 'none'; });
      sw.style.outline = '2px solid #fff';
      sw.style.outlineOffset = '2px';
    });
  });
  const savePorts = body.querySelector('#f-save-ports');
  if (savePorts) savePorts.addEventListener('click', async () => {
    try {
      await api('/api/tasks/' + selectedTaskId, {
        method: 'POST',
        body: JSON.stringify({
          outputs: portData.outputs.map((p) => ({
            id: p.id, name: p.name,
            capture: p.capture, glob: p.capture === 'file' ? (p.glob || '*') : undefined
          }))
        })
      });
      toast('产出已保存', 'ok');
      await refresh();
      openTaskPanel(selectedTaskId);
    } catch (err) { toast('保存产出失败：' + err.message, 'err'); }
  });
  const saveBtn = body.querySelector('#f-save');
  if (saveBtn) saveBtn.addEventListener('click', async () => {
    if (!formData.name.trim()) { toast('请填写任务名称', 'warn'); return; }
    try {
      await api('/api/tasks/' + selectedTaskId, {
        method: 'POST',
        body: JSON.stringify({
          name: formData.name, engine: formData.engine, color: formData.color,
          description: formData.description, command: formData.command,
          scope: formData.scope, spec: formData.spec, group: formData.group
        })
      });
      toast('已保存', 'ok');
      await refresh();
      openTaskPanel(selectedTaskId);
    } catch (err) { toast('保存失败：' + err.message, 'err'); }
  });
  const createSub = body.querySelector('#btn-create-subgraph');
  if (createSub) createSub.addEventListener('click', async () => {
    try {
      const { graph } = await api('/api/graphs', {
        method: 'POST',
        body: JSON.stringify({ name: (taskOf(selectedTaskId) || {}).name + ' · 内部流程', parentNodeId: selectedTaskId })
      });
      toast('已创建内部流程图', 'ok');
      await refresh();
      enterGraph(graph.id);
    } catch (err) { toast('创建失败：' + err.message, 'err'); }
  });
  const enterSub = body.querySelector('#btn-enter-subgraph');
  if (enterSub) enterSub.addEventListener('click', () => {
    const t = taskOf(selectedTaskId);
    if (t && t.subgraphId) enterGraph(t.subgraphId);
  });
  const bigRun = body.querySelector('[data-bigrun]');
  if (bigRun) bigRun.addEventListener('click', () => runTask(selectedTaskId));
  const delBtn = body.querySelector('#f-del');
  if (delBtn) delBtn.addEventListener('click', () => deleteTask(selectedTaskId));

  const sType = body.querySelector('#s-type');
  if (sType) {
    sType.value = schedData.type;
    renderSchedExtra(body);
    sType.addEventListener('change', () => { schedData.type = sType.value; renderSchedExtra(body); });
  }
  const sEnabled = body.querySelector('#s-enabled');
  if (sEnabled) sEnabled.addEventListener('change', () => (schedData.enabled = sEnabled.checked));
  const sSave = body.querySelector('#s-save');
  if (sSave) sSave.addEventListener('click', async () => {
    const v = (id) => { const el = body.querySelector(id); return el ? el.value : null; };
    if (v('#s-minutes') != null) schedData.minutes = parseInt(v('#s-minutes'), 10) || 5;
    if (v('#s-time') != null) schedData.time = v('#s-time');
    if (v('#s-cron') != null) schedData.cron = v('#s-cron');
    try {
      await api('/api/schedules', {
        method: 'POST',
        body: JSON.stringify({
          taskId: selectedTaskId, type: schedData.type,
          minutes: schedData.minutes, time: schedData.time, cron: schedData.cron,
          enabled: schedData.enabled
        })
      });
      toast('定时已保存', 'ok');
      await refresh();
    } catch (err) { toast('保存定时失败：' + err.message, 'err'); }
  });
}
function renderSchedExtra(body) {
  const box = body.querySelector('#s-extra');
  if (!box) return;
  if (schedData.type === 'interval') {
    box.innerHTML = `<div class="field"><label>间隔（分钟）</label>
      <input type="number" id="s-minutes" min="1" value="${schedData.minutes}"></div>`;
  } else if (schedData.type === 'daily') {
    box.innerHTML = `<div class="field"><label>每日触发时间</label>
      <input type="time" id="s-time" value="${esc(schedData.time)}"></div>`;
  } else {
    box.innerHTML = `<div class="field"><label>cron 表达式（分 时 日 月 周）</label>
      <input type="text" id="s-cron" value="${esc(schedData.cron)}" spellcheck="false">
      <div class="hint">例如：<code>0 * * * *</code>、<code>*/5 * * * *</code></div></div>`;
  }
  const el = box.querySelector('input');
  if (el) el.addEventListener('input', () => {
    if (schedData.type === 'interval') schedData.minutes = parseInt(el.value, 10) || 5;
    if (schedData.type === 'daily') schedData.time = el.value;
    if (schedData.type === 'cron') schedData.cron = el.value;
  });
}

/* ---------- Trace 历史弹窗 ---------- */
const traceModal = $('#trace-modal');
const KIND_TEXT = { graph: '运行图', manual: '手动', scheduled: '定时' };
const expandedTraces = new Set();
$('#btn-traces').addEventListener('click', async () => {
  traceModal.hidden = false;
  await renderTraceList();
});
$('#tm-close').addEventListener('click', () => { traceModal.hidden = true; });
$('#tm-refresh').addEventListener('click', renderTraceList);
traceModal.addEventListener('click', (e) => {
  if (e.target.closest('[data-close]')) traceModal.hidden = true;
});
async function renderTraceList() {
  const body = $('#tm-body');
  body.innerHTML = '<div class="no-runs">加载中…</div>';
  try {
    const { traces } = await api('/api/traces');
    if (!traces.length) {
      body.innerHTML = '<div class="no-runs">还没有执行记录。点「运行图」或节点的 ▶ 启动试试。</div>';
      return;
    }
    body.innerHTML = traces.map((t) => `
      <div class="trace-item ${expandedTraces.has(t.id) ? 'open' : ''}" data-tid="${esc(t.id)}">
        <span class="r-dot ${esc(t.status)}"></span>
        <span class="t-kind">${esc(KIND_TEXT[t.kind] || t.kind)}</span>
        <span class="t-label">${esc(t.label || '')}</span>
        <span class="t-time">${fmtTime(t.startedAt)}</span>
        <span class="t-stat">${t.runCount} 步 · ${t.finishedAt ? fmtDur(t.durationMs) : '运行中'}</span>
        <span class="t-arrow">▸</span>
      </div>`).join('');
  } catch (e) {
    body.innerHTML = '<div class="no-runs">加载失败：' + esc(e.message) + '</div>';
  }
}
$('#tm-body').addEventListener('click', (e) => {
  const runEl = e.target.closest('[data-run-id]');
  if (runEl) { toggleTraceRun(runEl); return; }
  const trEl = e.target.closest('[data-tid]');
  if (trEl && !e.target.closest('.trace-detail')) toggleTraceDetail(trEl);
});
async function toggleTraceDetail(rowEl) {
  const tid = rowEl.dataset.tid;
  const body = $('#tm-body');
  if (expandedTraces.has(tid)) {
    expandedTraces.delete(tid);
    rowEl.classList.remove('open');
    const d = body.querySelector(`.trace-detail[data-tid-detail="${tid}"]`);
    if (d) d.remove();
    return;
  }
  expandedTraces.add(tid);
  rowEl.classList.add('open');
  let detail = body.querySelector(`.trace-detail[data-tid-detail="${tid}"]`);
  if (detail) detail.remove();
  detail = document.createElement('div');
  detail.className = 'trace-detail';
  detail.dataset.tidDetail = tid;
  detail.innerHTML = '<div class="no-runs">加载步骤…</div>';
  rowEl.after(detail);
  try {
    const { trace } = await api('/api/traces/' + tid);
    detail.innerHTML = trace.runs.map((r, i) => `
      <div class="trace-run" data-run-id="${esc(r.id)}">
        <span class="r-dot ${esc(r.status)}"></span>
        <span class="t-step">#${i + 1}</span>
        <b>${esc(r.taskName || '?')}</b>
        <span class="t-st ${esc(r.status)}">${esc(statusText(r.status))}</span>
        <span class="t-dur">${r.durationMs != null ? fmtDur(r.durationMs) : '…'}</span>
      </div>`).join('');
  } catch (e) {
    detail.innerHTML = '<div class="no-runs">加载失败</div>';
  }
}
async function toggleTraceRun(rowEl) {
  const runId = rowEl.dataset.runId;
  const body = $('#tm-body');
  const existing = body.querySelector(`.trace-out[data-out="${runId}"]`);
  if (existing) { existing.remove(); return; }
  const box = document.createElement('div');
  box.className = 'trace-out';
  box.dataset.out = runId;
  box.textContent = '加载输出…';
  rowEl.after(box);
  try {
    const { run } = await api('/api/runs/' + runId);
    box.textContent = '';
    for (const c of (run.output || [])) {
      const sp = document.createElement('span');
      if (c.s === 'err') sp.className = 'err';
      else if (c.s === 'out') sp.className = 'ok';
      sp.textContent = c.d;
      box.appendChild(sp);
    }
    box.scrollTop = box.scrollHeight;
  } catch (e) {
    box.textContent = '加载失败';
  }
}

/* ---------- Agent 工位：认领 / 上传成果 / 提交 ---------- */
const workerModal = $('#worker-modal');
const wm = {
  id: (function () { try { return localStorage.getItem('taskpad:workerId') || ''; } catch (e) { return ''; } })() || ('agent-' + Math.random().toString(36).slice(2, 6)),
  run: null,
  timer: null,
  uploads:[]
};
function wmSaveId() { try { localStorage.setItem('taskpad:workerId', wm.id); } catch (e) { /* ignore */ } }
function wmOpen() {
  workerModal.hidden = false;
  wmRender();
}
function wmRenderIdle() {
  $('#wm-body').innerHTML = `
    <div class="wm-row">
      <label style="color:var(--muted);font-size:11.5px">节点名</label>
      <input id="wm-id" value="${esc(wm.id)}" placeholder="你的 agent 名称">
      <button class="btn primary" id="wm-claim">🤝 认领任务</button>
    </div>
    <div class="hint">网站只做调度与存储，不执行命令。认领后：本地执行 run.command → 上传成果文件 → 勾选接口测试 → 提交结果。<br>
    同样的流程也可以直接用 curl / worker.py 完成（见右上「🤖 Agent 指南」）。</div>`;
  $('#wm-claim').addEventListener('click', wmClaim);
}
function wmRender() {
  if (wm.run) wmRenderTask();
  else wmRenderIdle();
}
async function wmClaim() {
  const idEl = $('#wm-id');
  if (idEl) { wm.id = idEl.value.trim() || wm.id; wmSaveId(); }
  try {
    const { run } = await api('/api/runs/claim', { method: 'POST', body: JSON.stringify({ workerId: wm.id }) });
    if (!run) { toast('暂无待认领任务（点击节点 ▶ 启动后任务会进入待认领）', 'warn'); return; }
    wm.run = run;
    wm.uploads = [];
    wmStartHeartbeat();
    wmRenderTask();
    await refresh();
  } catch (err) { toast('认领失败：' + err.message, 'err'); }
}
function wmStartHeartbeat() {
  wmStopHeartbeat();
  wm.timer = setInterval(async () => {
    if (!wm.run) return;
    try {
      const r = await api('/api/runs/' + wm.run.runId + '/heartbeat', { method: 'POST', body: JSON.stringify({ workerId: wm.id }) });
      const hb = $('#wm-hb');
      if (hb) { hb.textContent = '♥ 心跳 ' + fmtTime(new Date().toISOString()); hb.className = 'wm-hb on'; }
      if (r.abort) {
        if (hb) { hb.textContent = '⏹ 已被停止'; hb.className = 'wm-hb abort'; }
        wmStopHeartbeat();
        toast('该任务已被停止，可勾选「中止」提交', 'warn');
      }
    } catch (e) {
      const hb = $('#wm-hb');
      if (hb) { hb.textContent = '心跳失败：' + e.message; hb.className = 'wm-hb abort'; }
    }
  }, 5000);
  setTimeout(() => wmHeartbeatOnce(), 300);
}
function wmStopHeartbeat() { if (wm.timer) { clearInterval(wm.timer); wm.timer = null; } }
async function wmHeartbeatOnce() {
  if (!wm.run) return;
  try {
    await api('/api/runs/' + wm.run.runId + '/heartbeat', { method: 'POST', body: JSON.stringify({ workerId: wm.id }) });
    const hb = $('#wm-hb');
    if (hb) { hb.textContent = '♥ 已认领'; hb.className = 'wm-hb on'; }
  } catch (e) { /* ignore */ }
}
function wmRenderTask() {
  const r = wm.run;
  const g = graphById(r.graphId);
  const ifaces = r.interfaces || [];
  const outs = r.outputs || [];
  const KIND_NAME = { code: '代码', data: '数据', net: '网络' };
  const ifaceHTML = ifaces.length ? ifaces.map((i) => {
    const tests = (i.tests || []).map((t) => `
      <label class="wm-test"><input type="checkbox" data-wm-test="${esc(i.edgeId)}:iface:${esc(t.id || t.name || '')}" data-test-name="${esc(t.name || '')}"> 接口测试：${esc(t.name || t.command)}</label>`).join('');
    return `<div class="wm-iface">
      <div class="if-head"><span class="if-kind">${esc(KIND_NAME[i.kind] || i.kind)}</span> ${esc(i.name)}
        <span class="if-dir">${i.direction === 'provide' ? '本模块提供 ↔ ' + esc(i.otherTaskName || '') : '← 使用 ' + esc(i.otherTaskName || '') + ' 的接口'}</span></div>
      ${i.spec ? '<div class="if-spec">规范：' + esc(i.spec) + '</div>' : ''}
      ${i.kind === 'net' ? '<div class="if-spec">' + esc(i.proto || '') + (i.auth ? ' · 鉴权 ' + esc(i.auth) : '') + '</div>' : ''}
      ${tests}
    </div>`;
  }).join('') : '<div class="no-runs">本模块暂无接口关系</div>';
  const outputHTML = outs.map((o) => {
    const upload = o.capture === 'file' ? `
      <div class="wm-file">
        <input type="file" data-wm-file="${esc(o.id)}">
        <button class="btn ghost" data-wm-upload="${esc(o.id)}">⬆ 上传成果</button>
      </div>
      <div class="wm-uploaded" data-wm-uploaded="${esc(o.id)}"></div>` : '';
    return `<div class="wm-iface">
      <div class="if-head"><span class="if-kind">${o.capture === 'file' ? '文件' : 'stdout'}</span> ${esc(o.name)}</div>
      ${o.spec ? '<div class="if-spec">规范：' + esc(o.spec) + '</div>' : ''}
      ${upload}
    </div>`;
  }).join('') || '<div class="no-runs">无</div>';
  $('#wm-body').innerHTML = `
    <div class="wm-task-head">${esc(r.taskName)} <span class="engine-badge" style="background:${esc(engineOf(r.engine).color || '#64748b')}">${esc((r.engine || '').toUpperCase())}</span></div>
    <div class="wm-meta">项目：${esc(g ? g.name : '—')}${r.scope ? ' · 工作范围：' + esc(r.scope) : ''}${r.spec ? '<br>工作规范：' + esc(r.spec) : ''}</div>
    <div class="wm-cmd">${esc(r.command)}</div>
    <div class="wm-sec">接口关系（模块 ↔ 模块）</div>
    ${ifaceHTML}
    <div class="wm-sec">产出成果</div>
    ${outputHTML}
    <div class="wm-sec">提交结果</div>
    <div class="wm-submit">
      <textarea id="wm-stdout" rows="4" placeholder="stdout（文本成果）"></textarea>
      <textarea id="wm-stderr" rows="2" placeholder="stderr"></textarea>
      <div class="wm-row" style="margin-top:6px">
        <label style="color:var(--muted);font-size:11.5px">exitCode</label>
        <input type="number" id="wm-exit" value="0" style="width:90px;flex:0 0 auto">
        <button class="btn primary" id="wm-submit">提交任务（含接口测试结果）</button>
        <button class="btn ghost" id="wm-next">继续接任务</button>
        <button class="btn ghost" id="wm-abort">放弃任务</button>
      </div>
      <div id="wm-result"></div>
    </div>`;
  $('#wm-submit').addEventListener('click', wmSubmit);
  $('#wm-next').addEventListener('click', () => { wm.run = null; wm.uploads = []; wmRenderIdle(); });
  $('#wm-abort').addEventListener('click', wmAbort);
  document.querySelectorAll('[data-wm-upload]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const portId = btn.dataset.wmUpload;
      const input = document.querySelector('[data-wm-file="' + portId + '"]');
      if (!input || !input.files || !input.files[0]) { toast('先选择要上传的成果文件', 'warn'); return; }
      wmUpload(portId, input.files[0]);
    });
  });
}

async function wmUpload(portId, file) {
  if (!wm.run) return;
  try {
    const res = await fetch('/api/runs/' + wm.run.runId + '/files/' + encodeURIComponent(file.name), { method: 'POST', body: file });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
    wm.uploads.push({ portId, name: data.name, size: data.size });
    const el = document.querySelector('[data-wm-uploaded="' + portId + '"]');
    if (el) el.textContent = '✓ 已上传：' + data.name + '（' + fmtSize(data.size) + '）';
    toast('成果已上传：' + data.name, 'ok');
  } catch (err) { toast('上传失败：' + err.message, 'err'); }
}
async function wmSubmit() {
  if (!wm.run) return;
  const tests = [...document.querySelectorAll('[data-wm-test]')].map((cb) => {
    const [edgeId, side, testId] = cb.dataset.wmTest.split(':');
    return { edgeId, portId: '', side, testId, name: cb.dataset.testName || '', passed: cb.checked, exitCode: cb.checked ? 0 : 1, output: '' };
  });
  const exitCode = parseInt(($('#wm-exit') || {}).value, 10);
  const stdout = ($('#wm-stdout') || {}).value || '';
  const stderr = ($('#wm-stderr') || {}).value || '';
  try {
    const res = await api('/api/runs/' + wm.run.runId + '/submit', {
      method: 'POST',
      body: JSON.stringify({ workerId: wm.id, exitCode: isNaN(exitCode) ? 0 : exitCode, stdout, stderr, tests })
    });
    wmStopHeartbeat();
    const ok = res.run.status === 'success';
    const box = $('#wm-result');
    if (box) {
      box.innerHTML = '<div class="wm-result ' + (ok ? '' : 'err') + '">已提交：' + esc(statusText(res.run.status)) +
        ' · exit=' + esc(String(res.run.exitCode)) + ' · 成果 ' + (res.run.portResults || []).length + ' 个</div>';
    }
    toast('任务已提交：' + statusText(res.run.status), ok ? 'ok' : 'warn');
    const sub = $('#wm-submit');
    if (sub) { sub.disabled = true; sub.textContent = '已提交 ✓'; }
    const ab = $('#wm-abort');
    if (ab) ab.style.display = 'none';
    await refresh();
  } catch (err) { toast('提交失败：' + err.message, 'err'); }
}
async function wmAbort() {
  if (!wm.run) return;
  if (!confirm('放弃当前任务？（会回到待认领，其它 agent 可接）')) return;
  try {
    await api('/api/runs/' + wm.run.runId + '/kill', { method: 'POST' });
    wmStopHeartbeat();
    wm.run = null;
    toast('已放弃任务', 'ok');
    await refresh();
    wmRender();
  } catch (err) { toast('放弃失败：' + err.message, 'err'); }
}
$('#btn-worker').addEventListener('click', wmOpen);
$('#wm-close').addEventListener('click', () => { workerModal.hidden = true; });
$('#wm-refresh').addEventListener('click', () => wmRender());
workerModal.addEventListener('click', (e) => { if (e.target.closest('[data-close]')) workerModal.hidden = true; });
workerModal.addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.target.id === 'wm-id') { e.preventDefault(); wmClaim(); } });

/* ---------- 时钟 / 启动 ---------- */
function tickClock() {
  const now = new Date();
  const p = (n) => String(n).padStart(2, '0');
  $('#clock').textContent =
    `${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())} ${p(now.getHours())}:${p(now.getMinutes())}:${p(now.getSeconds())}`;
}
tickClock();
setInterval(tickClock, 1000);
applyView();
loadTemplates();
/* ---------- 图栏事件 ---------- */
$('#btn-graph-back').addEventListener('click', goBackGraph);
$('#btn-graph-new').addEventListener('click', async () => {
  const name = prompt('新图名称：', '新图');
  if (!name || !name.trim()) return;
  try {
    const { graph } = await api('/api/graphs', { method: 'POST', body: JSON.stringify({ name: name.trim() }) });
    toast('已创建图「' + graph.name + '」', 'ok');
    await refresh();
    enterGraph(graph.id);
  } catch (err) { toast('创建失败：' + err.message, 'err'); }
});
$('#btn-graph-rename').addEventListener('click', async () => {
  const g = currentGraph();
  if (!g) return;
  const name = prompt('重命名当前图：', g.name);
  if (!name || !name.trim()) return;
  try {
    await api('/api/graphs/' + g.id, { method: 'POST', body: JSON.stringify({ name: name.trim() }) });
    toast('已重命名', 'ok');
    await refresh();
  } catch (err) { toast('重命名失败：' + err.message, 'err'); }
});
$('#btn-graph-del').addEventListener('click', async () => {
  const g = currentGraph();
  if (!g || g.id === rootGraphId()) return;
  if (!confirm('删除图「' + g.name + '」及其内部流程图和节点？')) return;
  try {
    await api('/api/graphs/' + g.id, { method: 'DELETE' });
    toast('已删除图', 'ok');
    state.activeGraphId = rootGraphId();
    await refresh();
    enterGraph(rootGraphId());
  } catch (err) { toast('删除失败：' + err.message, 'err'); }
});
$('#graph-path').addEventListener('click', (e) => {
  const el = e.target.closest('[data-goto-graph]');
  if (el) enterGraph(el.dataset.gotoGraph);
});

refresh().then(() => connectSSE());
// 周期刷新：更新节点在线状态等（SSE 已覆盖大部分变更）
setInterval(() => { if (!document.hidden) refresh(); }, 20000);