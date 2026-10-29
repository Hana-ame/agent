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
  pathEl.innerHTML = path.map((g, i) => `
    <span class="gp-item ${i === path.length - 1 ? 'current' : ''}" data-goto-graph="${esc(g.id)}">${esc(g.name)}</span>
    ${i < path.length - 1 ? '<span class="gp-sep">▸</span>' : ''}`).join('');
  const back = $('#btn-graph-back');
  if (back) back.disabled = path.length <= 1;
  const del = $('#btn-graph-del');
  if (del) del.disabled = path.length <= 1;
  const sel = $('#graph-select');
  if (sel) {
    const roots = (state.graphs || []).filter((g) => !g.parentNodeId);
    sel.innerHTML = roots.map((g) => `<option value="${esc(g.id)}" ${g.id === state.activeGraphId ? 'selected' : ''}>${esc(g.name)}</option>`).join('');
    sel.value = state.activeGraphId;
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

/* ---------- 节点几何 ---------- */
function nodeHeight(t) {
  const cnt = Math.max(portsIn(t).length, portsOut(t).length, 2);
  return HEADER_H + cnt * PORT_STEP + 48;
}
function portIndex(list, portId) {
  const i = list.findIndex((p) => p.id === portId);
  return i < 0 ? 0 : i;
}
function portPos(task, side, portId) {
  const list = side === 'out' ? portsOut(task) : portsIn(task);
  const i = portIndex(list, portId);
  const x = side === 'out' ? task.x + NODE_W : task.x;
  const y = task.y + PORTS_Y0 + i * PORT_STEP;
  return { x: x * view.scale + view.x, y: y * view.scale + view.y };
}

/* ---------- 连线（SVG） ---------- */
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
  let html = '';
  for (const e of currentEdges()) {
    const from = taskOf(e.from);
    const to = taskOf(e.to);
    if (!from || !to) continue;
    const a = portPos(from, 'out', e.fromPort);
    const b = portPos(to, 'in', e.toPort);
    const fromStatus = nodeStatusOf(from);
    const kind = e.kind || 'code';
    const KIND_COLOR = { code: '#4f8cff', data: '#22c55e', net: '#f59e0b' };
    const color = fromStatus === 'running' ? '#f59e0b' : (KIND_COLOR[kind] || engineOf(from.engine).color || '#4f8cff');
    const KIND_DASH = { code: '10 6', data: '3 9', net: '14 8' };
    const extra = sel.edgeId === e.id ? ' selected' : '';
    const d = wirePath(a.x, a.y, b.x, b.y);
    html += `<path class="wire${extra}" data-edge="${esc(e.id)}" d="${d}" stroke="${esc(color)}" fill="none" stroke-width="2.5" stroke-dasharray="${KIND_DASH[kind] || '10 6'}"></path>`;
    html += `<path class="wire hit" data-edge="${esc(e.id)}" d="${d}"></path>`;
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2 - 8;
    const kindTag = kind === 'net' ? '🌐' : kind === 'data' ? '⇄' : '⌗';
    html += `<text class="wire-label" x="${mx.toFixed(1)}" y="${my.toFixed(1)}" text-anchor="middle" pointer-events="none">${kindTag} ${esc(e.label || e.carryName || e.fromPortName || '')}</text>`;
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
  const a = portPos(from, wireDrag.fromSide, wireDrag.fromPort);
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

/* ---------- 节点渲染 ---------- */
function renderNodes() {
  if (nodeDrag) return;
  world.innerHTML = '';
  for (const t of currentTasks()) {
    const st = nodeStatusOf(t);
    const eng = engineOf(t.engine);
    const ins = portsIn(t);
    const outs = portsOut(t);
    const cnt = Math.max(ins.length, outs.length, 2);

    const el = document.createElement('div');
    el.className = 'node' + (sel.nodeId === t.id ? ' selected' : '');
    el.dataset.id = t.id;
    el.style.left = t.x + 'px';
    el.style.top = t.y + 'px';
    el.style.setProperty('--c', t.color || eng.color || PALETTE[0]);

    const bodyH = cnt * PORT_STEP + 38;
    el.innerHTML = `
      <div class="node-head">
        <span class="node-status ${esc(st)}"></span>
        <span class="node-title">${esc(t.name)}</span>
        ${t.group ? `<span class="group-chip">${esc(t.group)}</span>` : ''}
        <span class="engine-badge" style="background:${esc(eng.color || '#64748b')}">${esc((t.engine || '').toUpperCase())}</span>
      </div>
      <div class="node-body" style="height:${bodyH}px">
        <div class="port-col in">
          ${ins.map((p, i) => `
            <div class="port-row in" style="top:${(PORTS_Y0 - HEADER_H - 7) + i * PORT_STEP}px">
              <span class="port input" data-port="${esc(p.id)}" data-side="in" title="输入端口"></span>
              <span class="port-name">${esc(p.name)}</span>
            </div>`).join('')}
        </div>
        <div class="port-col out">
          ${outs.map((p, i) => `
            <div class="port-row out" style="top:${(PORTS_Y0 - HEADER_H - 7) + i * PORT_STEP}px">
              <span class="port-name">${esc(p.name)}</span>
              <span class="port output" data-port="${esc(p.id)}" data-side="out" title="输出端口 ${p.capture === 'file' ? '(文件成果)' : '(stdout成果)'}"></span>
            </div>`).join('')}
        </div>
        <div class="node-desc">${esc(t.description || '')}</div>
        <div class="node-command">${esc(t.command || '')}</div>
      </div>
      ${t.subgraphId ? '<span class="subgraph-badge">⧉ 内部流程</span>' : ''}
      <div class="node-actions">
        <button class="a-run" data-act="run" title="立即启动">▶ 启动</button>
        <button class="a-edit" data-act="edit" title="端口 / 定时 / 日志 / 成果">⚙ 详情</button>
        <button class="a-del" data-act="del" title="删除">✕</button>
      </div>`;
    world.appendChild(el);
    bindNodeEvents(el, t);
  }
  layoutWires();
}

function bindNodeEvents(el, t) {
  // 按下节点任意位置（端口/按钮除外）：选中并开始拖动
  el.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.port') || e.target.closest('button')) return;
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

  el.querySelectorAll('.port').forEach((port) => {
    port.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const side = port.dataset.side;
      startWireDrag(t.id, port.dataset.port, side, e);
    });
  });
}

/* 节点拖动（window 级监听，避免 pointer capture 与子元素监听不一致） */
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

/* ---------- 端口拖拽 → 连线（edges API） ---------- */
function startWireDrag(taskId, portId, side, e) {
  wireDrag.active = true;
  wireDrag.fromTaskId = taskId;
  wireDrag.fromPort = portId;
  wireDrag.fromSide = side;
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
  const portEl = el2 && el2.closest ? el2.closest('.port') : null;
  document.querySelectorAll('.port.hit').forEach((x) => x.classList.remove('hit'));
  wireDrag.target = null;
  if (portEl) {
    const nodeEl = portEl.closest('.node');
    const id = nodeEl ? nodeEl.dataset.id : null;
    const side = portEl.dataset.side;
    if (id && id !== wireDrag.fromTaskId && wireDrag.fromSide !== side) {
      wireDrag.target = { taskId: id, portId: portEl.dataset.port, side };
      portEl.classList.add('hit');
    }
  }
}
function onWireUp() {
  window.removeEventListener('pointermove', onWireMove);
  window.removeEventListener('pointerup', onWireUp);
  board.classList.remove('panning');
  document.querySelectorAll('.port.hit').forEach((x) => x.classList.remove('hit'));
  if (!wireDrag.active) return;
  wireDrag.active = false;
  const g = wiresSvg.querySelector('path.ghost');
  if (g) g.remove();

  const target = wireDrag.target;
  if (!target) return;
  let from = null, fromPort = null, to = null, toPort = null;
  if (wireDrag.fromSide === 'out' && target.side === 'in') {
    from = wireDrag.fromTaskId; fromPort = wireDrag.fromPort; to = target.taskId; toPort = target.portId;
  } else if (wireDrag.fromSide === 'in' && target.side === 'out') {
    from = target.taskId; fromPort = target.portId; to = wireDrag.fromTaskId; toPort = wireDrag.fromPort;
  }
  if (!from || !to) return;
  api('/api/edges', { method: 'POST', body: JSON.stringify({ from, fromPort, to, toPort }) })
    .then(() => { toast('已连线：' + ((taskOf(from) || {}).name || '') + ' → ' + ((taskOf(to) || {}).name || ''), 'ok'); return refresh(); })
    .catch((err) => toast('连线失败：' + err.message, 'err'));
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
  $('#panel-title').textContent = t ? ('任务详情 · ' + t.name) : '任务详情';
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
        <div class="lib-item" data-tpl="${esc(t.name)}" style="--c:${esc(t.color || eng.color)}">
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
function portEditorHTML(list, side) {
  const isIn = side === 'in';
  return list.map((p, i) => {
    const key = side + ':' + p.id;
    const open = expandedPorts.has(key);
    const tests = p.tests || [];
    return `
      <div class="port-edit">
        <span class="pe-dot ${isIn ? 'in' : 'out'}"></span>
        <input type="text" data-${isIn ? 'pin' : 'pout'}="${i}" value="${esc(p.name)}" style="flex:1">
        ${isIn ? '' : `
          <select data-pout-cap="${i}">
            <option value="stdout" ${p.capture === 'stdout' ? 'selected' : ''}>stdout</option>
            <option value="file" ${p.capture === 'file' ? 'selected' : ''}>文件</option>
          </select>
          <input type="text" data-pout-glob="${i}" value="${esc(p.glob || '*')}" placeholder="glob" title="文件匹配模式" style="width:58px;display:${p.capture === 'file' ? '' : 'none'}">`}
        <button class="pe-tests ${tests.length ? '' : 'warn'}" data-toggle-port="${esc(key)}" title="接口规范与接口测试">测试 ${tests.length}</button>
        <button class="pe-del" data-del-${isIn ? 'pin' : 'pout'}="${i}" title="删除端口">✕</button>
      </div>
      ${open ? `<div class="port-detail">
        <div class="pd-label">接口规范（这个接口必须满足什么）</div>
        <textarea data-port-spec="${esc(key)}" rows="2" placeholder="例如：JSON，必须包含 id/title 字段；或 CSV 第一行为表头">${esc(p.spec || '')}</textarea>
        <div class="pd-label">接口测试（由 agent 执行 shell 命令，exit 0 = 通过）</div>
        ${tests.map((t, ti) => `
          <div class="test-row">
            <input type="text" data-test-name="${esc(key)}:${ti}" value="${esc(t.name || '')}" placeholder="测试名">
            <input type="text" data-test-cmd="${esc(key)}:${ti}" value="${esc(t.command || '')}" placeholder="shell 命令">
            <button data-del-test="${esc(key)}:${ti}" title="删除测试">✕</button>
          </div>`).join('')}
        <div class="hint">输入端口变量：<code>$__INPUT</code> / <code>$__INPUT_FILE</code>；输出端口变量：<code>$__RESULT</code> / <code>$__RESULT_FILE</code></div>
        <button class="btn ghost block" data-add-test="${esc(key)}">＋ 添加接口测试</button>
      </div>` : ''}`;
  }).join('') || '<div class="no-runs">无</div>';
}

function renderEditPanel() {
  const t = taskOf(selectedTaskId);
  if (!t) { closePanel(); return; }
  const scheds = state.schedules.filter((s) => s.taskId === t.id);
  const runs = state.runs.filter((r) => r.taskId === t.id);
  const st = nodeStatusOf(t);
  const eng = engineOf(t.engine);
  const subGraph = t.subgraphId ? graphById(t.subgraphId) : null;

  const inEdges = state.edges.filter((e) => e.to === t.id);
  const outEdges = state.edges.filter((e) => e.from === t.id);
  const lastOkRun = [...runs].find((r) => r.status === 'success' || (r.portResults && r.portResults.length));

  panelBody.innerHTML = `
    <div class="section-title">
      <span class="node-status ${esc(st)}" style="display:inline-block"></span> ${esc(t.name)}
      <span class="engine-badge" style="background:${esc(eng.color || '#64748b')};float:right">${esc((t.engine || '').toUpperCase())}</span>
    </div>

    <button class="btn primary block big" data-bigrun="1">▶ 启动任务（等待节点认领）</button>

    <div class="section-title">📋 作业契约</div>
    <div class="field"><label>工作范围（agent 可操作的范围 / 边界）</label>
      <textarea id="f-scope" placeholder="例如：仅允许读取 v8/ 目录；只运行只读命令；不得修改数据">${esc(formData.scope)}</textarea></div>
    <div class="field"><label>工作规范（agent 必须遵守的作业要求）</label>
      <textarea id="f-spec" placeholder="例如：完成后必须输出 JSON 汇总；失败需在 stderr 说明原因">${esc(formData.spec)}</textarea></div>

    <div class="section-title">🧩 内部流程图</div>
    ${subGraph
      ? `<button class="btn block" id="btn-enter-subgraph">进入内部流程图（${esc(subGraph.name)}）</button>`
      : `<button class="btn ghost block" id="btn-create-subgraph">创建内部流程图</button>
         <div class="hint">节点内部可以有自己的流程图（子图），用于描述该 agent 的工作步骤。</div>`}

    <div class="section-title">↔ 接口（端口）：规范 + 测试</div>
    <div class="field"><label>输入接口</label>
      <div id="pin-list">${portEditorHTML(portData.inputs, 'in')}</div>
      <button class="btn ghost block" id="btn-add-pin">＋ 添加输入接口</button>
    </div>
    <div class="field"><label>输出接口</label>
      <div id="pout-list">${portEditorHTML(portData.outputs, 'out')}</div>
      <button class="btn ghost block" id="btn-add-pout">＋ 添加输出接口</button>
    </div>
    <button class="btn block" id="f-save-ports">保存接口（规范 + 测试）</button>

    <div class="section-title">📦 模块分组</div>
    <div class="field"><label>模块分组 / 类别</label>
      <input type="text" id="f-group" value="${esc(formData.group || '')}" placeholder="如 core / adapter / script / config"></div>

    <div class="section-title">🔗 连线与接口契约</div>
    <div class="hint" style="margin-bottom:6px">edge 规定上游输出接口 → 下游输入接口之间必须满足什么；数据必须通过接口测试才会投递给下游。</div>
    ${[...inEdges.map((e) => ({ e, dir: 'in' })), ...outEdges.map((e) => ({ e, dir: 'out' }))].map(({ e, dir }) => `
      <div class="sched-item"><span class="s-chip">${dir === 'in'
        ? '← ' + esc(e.fromName) + ' [' + esc(e.fromPortName || e.carryName) + '] → 本节点 [' + esc(e.toPortName) + ']'
        : '本节点 [' + esc(e.fromPortName || e.carryName) + '] → ' + esc(e.toName) + ' [' + esc(e.toPortName) + ']'}</span>
        <button class="sched-del" data-edge-del="${esc(e.id)}" title="断开">✕</button></div>
      <div class="edge-editor" data-edge-editor="${esc(e.id)}">
        <div class="ee-row">
          <input data-edge-label value="${esc(e.label || '')}" placeholder="接口名">
          <select data-edge-kind>
            <option value="code" ${e.kind === 'code' ? 'selected' : ''}>代码</option>
            <option value="data" ${e.kind === 'data' ? 'selected' : ''}>数据</option>
            <option value="net" ${e.kind === 'net' ? 'selected' : ''}>网络</option>
          </select>
          <input data-edge-proto value="${esc(e.proto || '')}" placeholder="协议" style="display:${e.kind === 'net' ? '' : 'none'}">
          <input data-edge-auth value="${esc(e.auth || '')}" placeholder="鉴权" style="display:${e.kind === 'net' ? '' : 'none'}">
        </div>
        <textarea data-edge-spec rows="2" placeholder="接口规范：这个接口必须满足什么（格式/约束/测试条件）">${esc(e.spec || '')}</textarea>
      </div>`).join('')}
    ${inEdges.length || outEdges.length ? '' : '<div class="no-runs">暂无连线，拖端口连一条</div>'}

    <div class="section-title">🧾 成果卡片（最近运行）</div>
    <div id="result-cards">
      ${lastOkRun && lastOkRun.portResults && lastOkRun.portResults.length
        ? lastOkRun.portResults.map((pr) => `
          <div class="result-card">
            <div class="rc-head">
              <span class="rc-name">${esc(pr.name)}</span>
              <span class="rc-kind ${esc(pr.kind)}">${pr.kind === 'file' ? '📎 文件' : '📄 文本'}</span>
              ${testBadge(pr)}
            </div>
            ${pr.kind === 'file'
              ? `<div class="rc-file">${esc(pr.file ? pr.file.name : '')}（${fmtSize(pr.file ? pr.file.size : null)}）
                  <a href="/api/runs/${esc(lastOkRun.id)}/file?portId=${esc(pr.portId)}" download>⬇ 下载</a></div>`
              : `<pre class="rc-text">${esc(pr.preview || '')}${pr.chars > (pr.preview || '').length ? '\n…' : ''}</pre>`}
            ${(pr.tests || []).length ? `<div class="hint">${(pr.tests || []).map((x) => (x.passed ? '✓ ' : '✗ ') + esc(x.name)).join(' · ')}</div>` : ''}
            <div class="rc-time">${fmtTime(lastOkRun.startedAt)} · exit=${lastOkRun.exitCode}</div>
          </div>`).join('')
        : '<div class="no-runs">运行成功后这里会出现各输出接口的成果卡（并通过接口测试校验）</div>'}
    </div>

    <div class="section-title">编辑信息</div>
    <div class="field"><label>任务名称</label>
      <input type="text" id="f-name" value="${esc(formData.name)}"></div>
    <div class="field"><label>执行引擎（V8 / V9）</label>
      <select id="f-engine">
        ${state.engines.map((e2) => `<option value="${esc(e2.id)}" ${formData.engine === e2.id ? 'selected' : ''}>${esc(e2.label)}</option>`).join('')}
      </select></div>
    <div class="field"><label>颜色</label>
      <div class="swatches">${PALETTE.map((c) => `<span class="swatch" data-c="${c}" style="background:${c}${formData.color === c ? ';outline:2px solid #fff;outline-offset:2px' : ''}"></span>`).join('')}</div></div>
    <div class="field"><label>描述</label>
      <textarea id="f-desc">${esc(formData.description)}</textarea></div>
    <div class="field"><label>执行命令</label>
      <textarea id="f-cmd" class="cmd">${esc(formData.command)}</textarea></div>
    <button class="btn block" id="f-save">保存修改</button>
    <button class="btn danger block" id="f-del">删除节点</button>

    <div class="section-title">⏰ 定时任务</div>
    <div id="sched-list">
      ${scheds.length ? scheds.map(schedRowHTML).join('') : '<div class="no-runs">尚未设置定时</div>'}
    </div>
    <div class="field" style="margin-top:8px"><label>新定时（基于此任务）</label>
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
      ${runs.length ? runs.slice(0, 30).map(runRowHTML).join('') : '<div class="no-runs">还没有运行记录，点「启动任务」运行一次</div>'}
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
  const togglePort = e.target.closest('[data-toggle-port]');
  if (togglePort) {
    const k = togglePort.dataset.togglePort;
    if (expandedPorts.has(k)) expandedPorts.delete(k); else expandedPorts.add(k);
    renderPanel();
    return;
  }
  const addTest = e.target.closest('[data-add-test]');
  if (addTest) {
    const key = addTest.dataset.addTest;
    const [side, pid] = key.split(':');
    const list = side === 'in' ? portData.inputs : portData.outputs;
    const port = list.find((x) => x.id === pid);
    if (port) {
      port.tests = port.tests || [];
      port.tests.push({ id: 't' + Math.random().toString(36).slice(2, 8), name: '测试' + (port.tests.length + 1), command: '' });
      renderPanel();
    }
    return;
  }
  const delTest = e.target.closest('[data-del-test]');
  if (delTest) {
    const [key, idx] = delTest.dataset.delTest.split(':');
    const [side, pid] = key.split(':');
    const list = side === 'in' ? portData.inputs : portData.outputs;
    const port = list.find((x) => x.id === pid);
    if (port && port.tests) { port.tests.splice(parseInt(idx, 10), 1); renderPanel(); }
    return;
  }
  const delPin = e.target.closest('[data-del-pin]');
  if (delPin) { portData.inputs.splice(parseInt(delPin.dataset.delPin, 10), 1); renderPanel(); return; }
  const delPout = e.target.closest('[data-del-pout]');
  if (delPout) { portData.outputs.splice(parseInt(delPout.dataset.delPout, 10), 1); renderPanel(); return; }
  const addPin = e.target.closest('#btn-add-pin');
  if (addPin) { portData.inputs.push({ id: 'p' + Math.random().toString(36).slice(2, 8), name: 'in' + (portData.inputs.length + 1), spec: '', tests: [] }); renderPanel(); return; }
  const addPout = e.target.closest('#btn-add-pout');
  if (addPout) { portData.outputs.push({ id: 'p' + Math.random().toString(36).slice(2, 8), name: 'out' + (portData.outputs.length + 1), spec: '', tests: [], capture: 'stdout', glob: '*' }); renderPanel(); return; }
});
$('#panel-body').addEventListener('input', (e) => {
  const pin = e.target.closest('[data-pin]');
  if (pin) { portData.inputs[parseInt(pin.dataset.pin, 10)].name = pin.value; return; }
  const pout = e.target.closest('[data-pout]');
  if (pout) { portData.outputs[parseInt(pout.dataset.pout, 10)].name = pout.value; return; }
  const glob = e.target.closest('[data-pout-glob]');
  if (glob) { portData.outputs[parseInt(glob.dataset.poutGlob, 10)].glob = glob.value; return; }
  const spec = e.target.closest('[data-port-spec]');
  if (spec) {
    const [side, pid] = spec.dataset.portSpec.split(':');
    const list = side === 'in' ? portData.inputs : portData.outputs;
    const port = list.find((x) => x.id === pid);
    if (port) port.spec = spec.value;
    return;
  }
  const tName = e.target.closest('[data-test-name]');
  if (tName) {
    const [key, idx] = tName.dataset.testName.split(':');
    const [side, pid] = key.split(':');
    const list = side === 'in' ? portData.inputs : portData.outputs;
    const port = list.find((x) => x.id === pid);
    if (port && port.tests && port.tests[parseInt(idx, 10)]) port.tests[parseInt(idx, 10)].name = tName.value;
    return;
  }
  const tCmd = e.target.closest('[data-test-cmd]');
  if (tCmd) {
    const [key, idx] = tCmd.dataset.testCmd.split(':');
    const [side, pid] = key.split(':');
    const list = side === 'in' ? portData.inputs : portData.outputs;
    const port = list.find((x) => x.id === pid);
    if (port && port.tests && port.tests[parseInt(idx, 10)]) port.tests[parseInt(idx, 10)].command = tCmd.value;
    return;
  }
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
    const val = (sel) => { const el = ed.querySelector(sel); return el ? el.value : ''; };
    const kind = val('[data-edge-kind]') || 'code';
    api('/api/edges/' + ed.dataset.edgeEditor, {
      method: 'POST',
      body: JSON.stringify({
        label: val('[data-edge-label]'),
        kind,
        proto: kind === 'net' ? val('[data-edge-proto]') : '',
        auth: kind === 'net' ? val('[data-edge-auth]') : '',
        spec: val('[data-edge-spec]')
      })
    }).then(() => toast('接口定义已保存', 'ok')).catch((err) => toast('保存接口失败：' + err.message, 'err'));
    // 切换为网络类型时显示协议/鉴权输入
    const protoEl = ed.querySelector('[data-edge-proto]');
    const authEl = ed.querySelector('[data-edge-auth]');
    if (protoEl) protoEl.style.display = kind === 'net' ? '' : 'none';
    if (authEl) authEl.style.display = kind === 'net' ? '' : 'none';
  }
});

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
          inputs: portData.inputs.map((p) => ({ id: p.id, name: p.name, spec: p.spec || '', tests: p.tests || [] })),
          outputs: portData.outputs.map((p) => ({
            id: p.id, name: p.name, spec: p.spec || '', tests: p.tests || [],
            capture: p.capture, glob: p.capture === 'file' ? (p.glob || '*') : undefined
          }))
        })
      });
      toast('接口已保存（规范 + 测试）', 'ok');
      await refresh();
      openTaskPanel(selectedTaskId);
    } catch (err) { toast('保存接口失败：' + err.message, 'err'); }
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