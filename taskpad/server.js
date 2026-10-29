'use strict';

/**
 * TaskPad — 任务白板（零依赖 Node.js）
 *
 * - 静态文件服务（public/）
 * - REST API：任务 / 运行 / 定时任务
 * - 定时调度器：interval（每 N 分钟）、daily（每日 HH:MM）、cron（5 段）
 * - SSE 实时推送运行日志
 * - JSON 文件持久化（data/store.json）
 *
 * 安全提示：本应用会以本机 shell 执行任务里的命令，仅限本机/内网使用。
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');

const ROOT = __dirname;
const PUBLIC = path.join(ROOT, 'public');
const DATA_DIR = path.join(ROOT, 'data');
const STORE_FILE = path.join(DATA_DIR, 'store.json');
const ARTIFACT_DIR = path.join(DATA_DIR, 'artifacts');

const RUN_CAP = 400;           // 最多保留运行记录
const TRACE_CAP = 200;          // 最多保留 trace 历史
const OUTPUT_BUDGET = 400000;  // 单次运行输出上限（字符，超出丢最旧）
const MISS_GRACE_MS = 120000;  // 定时错过多少毫秒内仍补跑

fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(ARTIFACT_DIR, { recursive: true });

// ---------- 配置 ----------
let config = {};
try {
  config = JSON.parse(fs.readFileSync(path.join(ROOT, 'config.json'), 'utf8'));
} catch (e) { /* 用默认 */ }
config.port = config.port || 8330;
// 项目根：任务里的命令默认以此为基准执行（taskpad 的上一级目录 = 工作区）
config.root = config.root || '..';
const PROJECT_ROOT = path.resolve(ROOT, config.root);
config.engines = config.engines || {
  v8: { label: 'V8', cwd: '.', env: { TASK_ENGINE: 'v8' }, timeoutMs: 1800000, color: '#3b82f6' },
  v9: { label: 'V9', cwd: '.', env: { TASK_ENGINE: 'v9' }, timeoutMs: 1800000, color: '#a855f7' }
};
for (const key of Object.keys(config.engines)) {
  const e = config.engines[key];
  e.cwd = path.resolve(PROJECT_ROOT, e.cwd || '.');
  e.timeoutMs = e.timeoutMs || 1800000;
  fs.mkdirSync(e.cwd, { recursive: true });
}

// ---------- 存储 ----------
function emptyStore() { return { tasks: [], schedules: [], runs: [] }; }
let store = emptyStore();
const storeFileExisted = fs.existsSync(STORE_FILE);
if (storeFileExisted) {
  try {
    store = JSON.parse(fs.readFileSync(STORE_FILE, 'utf8'));
  } catch (e) {
    fs.renameSync(STORE_FILE, STORE_FILE + '.backup-' + Date.now());
  }
}
['tasks', 'schedules', 'runs', 'traces'].forEach((k) => { if (!Array.isArray(store[k])) store[k] = []; });
store.tasks.forEach((t) => { if (!Array.isArray(t.deps)) t.deps = []; });

// ---------- 迁移：图结构 + 节点 scope/spec + 端口 spec/tests ----------
function migrateGraphs() {
  if (!Array.isArray(store.graphs) || !store.graphs.length) {
    store.graphs = [{ id: uid(), name: '主图', parentNodeId: null, createdAt: nowIso() }];
  }
  const rootId = store.graphs[0].id;
  for (const t of store.tasks) {
    if (!t.graphId) t.graphId = rootId;
    if (typeof t.scope !== 'string') t.scope = '';
    if (typeof t.spec !== 'string') t.spec = '';
    if (t.subgraphId === undefined) t.subgraphId = null;
    t.inputs = normalizePorts(t.inputs || [], false);
    t.outputs = normalizePorts(t.outputs || [], true);
    t.deps = [];
  }
  for (const e of store.edges) {
    if (!e.graphId) e.graphId = rootId;
    if (typeof e.contract !== 'string') e.contract = '';
  }
}

// ---------- 迁移旧数据：deps → edges + 默认端口 ----------
function migrateStore() {
  store.edges = store.edges || [];
  for (const t of store.tasks) {
    if (!Array.isArray(t.inputs) || !t.inputs.length) t.inputs = [{ id: 'in', name: 'dep' }];
    if (!Array.isArray(t.outputs) || !t.outputs.length) t.outputs = [{ id: 'out', name: 'out', capture: 'stdout' }];
    if (Array.isArray(t.deps) && t.deps.length) {
      for (const d of t.deps) {
        const src = store.tasks.find((x) => x.id === d);
        const fromPort = src && src.outputs && src.outputs[0] ? src.outputs[0].id : 'out';
        if (!store.edges.some((e) => e.from === d && e.to === t.id)) {
          store.edges.push({ id: uid(), from: d, fromPort, to: t.id, toPort: 'in', carryName: 'out' });
        }
      }
      t.deps = [];
    }
  }
}

function normalizeTests(list) {
  const out = [];
  for (const t of list || []) {
    const command = String(t.command || '').trim();
    if (!command) continue;
    out.push({ id: String(t.id || uid()).slice(0, 12), name: String(t.name || '').trim() || 'test', command });
  }
  return out;
}
function normalizePorts(list, isOutput) {
  const out = [];
  for (const p of list || []) {
    const name = String(p.name || '').trim() || 'port';
    out.push({
      id: String(p.id || uid()).slice(0, 12),
      name,
      spec: String(p.spec || ''),
      tests: normalizeTests(p.tests),
      capture: isOutput ? (p.capture === 'file' ? 'file' : 'stdout') : undefined,
      glob: isOutput && p.capture === 'file' ? String(p.glob || '*') : undefined
    });
  }
  if (out.length) return out;
  return isOutput
    ? [{ id: 'out', name: 'out', spec: '', tests: [], capture: 'stdout' }]
    : [{ id: 'in', name: 'dep', spec: '', tests: [] }];
}

let saveChain = Promise.resolve();
function save() {
  saveChain = saveChain
    .then(() => new Promise((resolve, reject) => {
      const tmp = STORE_FILE + '.tmp';
      fs.writeFile(tmp, JSON.stringify(store), (err) => {
        if (err) return reject(err);
        fs.rename(tmp, STORE_FILE, (e) => (e ? reject(e) : resolve()));
      });
    }))
    .catch((err) => console.error('[save] failed:', err.message));
  return saveChain;
}

function uid() { return crypto.randomBytes(8).toString('hex'); }
function nowIso() { return new Date().toISOString(); }

// ---------- 运行 ----------
const procs = new Map();      // runId -> { proc, timer }
const killedSet = new Set();  // runId 被手动终止
const sseClients = new Set(); // SSE 客户端写流

function broadcast(payload) {
  const data = 'data: ' + JSON.stringify(payload) + '\n\n';
  for (const client of sseClients) {
    try { client.write(data); } catch (e) { sseClients.delete(client); }
  }
}

function abortRun(runId, reason) {
  const r = store.runs.find((x) => x.id === runId);
  if (!r) return;
  if (r.status === 'pending') {
    r.status = 'aborted';
    r.finishedAt = nowIso();
    r.durationMs = r.startedAt ? (new Date(r.finishedAt).getTime() - new Date(r.startedAt).getTime()) : 0;
    r.output.push({ t: Date.now(), s: 'err', d: '[中止] ' + (reason || '手动中止') + '\n' });
    if (r.traceId) updateTrace(r.traceId);
    save();
    broadcast({ type: 'run:end', run: briefRun(r) });
  } else if (r.status === 'running') {
    r.abortRequested = true;
    save();
    broadcast({ type: 'run:update', run: briefRun(r) });
  }
}

// 失联清扫：running 但心跳超时的运行回到 pending，等其它节点认领
setInterval(() => {
  const now = Date.now();
  let changed = false;
  for (const r of store.runs) {
    if (r.status !== 'running' || !r.heartbeatAt) continue;
    if (now - new Date(r.heartbeatAt).getTime() > 90000) {
      r.status = 'pending';
      r.worker = null;
      r.claimedAt = null;
      r.heartbeatAt = null;
      r.output.push({ t: Date.now(), s: 'err', d: '[失联] 节点心跳超时，运行重新入队\n' });
      changed = true;
      broadcast({ type: 'run:update', run: briefRun(r) });
    }
  }
  if (changed) save();
}, 15000).unref();

function briefRun(r) {
  return {
    id: r.id, taskId: r.taskId, taskName: r.taskName, engine: r.engine, trigger: r.trigger,
    triggerLabel: r.triggerLabel, status: r.status, startedAt: r.startedAt, finishedAt: r.finishedAt,
    worker: r.worker || null, claimedAt: r.claimedAt || null, abortRequested: !!r.abortRequested,
    exitCode: r.exitCode, durationMs: r.durationMs, outputChars: r.output ? r.output.reduce((n, o) => n + o.d.length, 0) : 0,
    portResults: Array.isArray(r.portResults) ? r.portResults.map((pr) => ({
      portId: pr.portId, name: pr.name, kind: pr.kind,
      preview: pr.kind === 'text' ? String(pr.preview || '').slice(0, 200) : '',
      chars: pr.chars || 0,
      file: pr.file ? { name: pr.file.name, size: pr.file.size } : null,
      valid: pr.valid !== false, tested: !!pr.tested,
      tests: Array.isArray(pr.tests) ? pr.tests.map((t) => ({ name: t.name, passed: !!t.passed })) : []
    })) : [],
    tests: Array.isArray(r.tests) ? r.tests.map((t) => ({ portId: t.portId, side: t.side, name: t.name, passed: !!t.passed })) : []
  };
}

function simpleGlobMatch(name, pattern) {
  if (pattern === '*' || pattern === '') return true;
  if (pattern === name) return true;
  const rx = '^' + pattern.split('*').map((x) => x.replace(/[.+^${}()|[\]\\]/g, '\\$&')).join('.*') + '$';
  return new RegExp(rx).test(name);
}

// 运行结束后捕获各输出端口的“成果”
function captureResults(runId) {
  const r = store.runs.find((x) => x.id === runId);
  if (!r) return;
  const task = store.tasks.find((t) => t.id === r.taskId);
  if (!task) return;
  const stdoutText = (r.output || []).map((o) => o.d).join('');
  const allTests = Array.isArray(r.tests) ? r.tests : [];
  const results = [];
  for (const out of task.outputs || []) {
    const cap = out.capture || 'stdout';
    const portTests = allTests.filter((t) => t.side === 'out' && t.portId === out.id);
    const valid = portTests.length ? portTests.every((t) => t.passed) : true;
    const tests = portTests.map((t) => ({ id: t.testId, name: t.name, passed: !!t.passed, exitCode: t.exitCode, output: String(t.output || '').slice(0, 2000) }));
    if (cap === 'stdout') {
      results.push({ portId: out.id, name: out.name, kind: 'text', preview: stdoutText.slice(0, 2000), chars: stdoutText.length, valid, tests, tested: portTests.length > 0 });
    } else if (cap === 'file') {
      const names = Object.keys(r.artifacts || {}).filter((fn) => simpleGlobMatch(fn, out.glob || '*'));
      if (names.length) {
        const fn = names[0];
        const art = r.artifacts[fn];
        results.push({ portId: out.id, name: out.name, kind: 'file', file: { name: fn, path: art.path, size: art.size }, valid, tests, tested: portTests.length > 0 });
      }
    }
  }
  r.portResults = results;
}

// ---------- 历史 trace ----------
function computeTraceStatus(trace) {
  let hasErr = false, hasSkip = false, hasKill = false, running = false;
  for (const rid of trace.runIds) {
    const r = store.runs.find((x) => x.id === rid);
    if (!r) continue;
    if (r.status === 'error') hasErr = true;
    else if (r.status === 'killed' || r.status === 'aborted') hasKill = true;
    else if (r.status === 'skipped') hasSkip = true;
    if (r.status === 'running' || r.status === 'pending' || r.status === 'queued') running = true;
  }
  if (running) return 'running';
  if (hasKill) return 'killed';
  if (hasErr) return 'error';
  if (hasSkip) return 'skipped';
  return 'success';
}
function updateTrace(traceId) {
  const t = store.traces.find((x) => x.id === traceId);
  if (!t) return;
  let fin = null;
  for (const rid of t.runIds) {
    const r = store.runs.find((x) => x.id === rid);
    if (r && r.finishedAt && (!fin || r.finishedAt > fin)) fin = r.finishedAt;
  }
  t.finishedAt = fin;
  t.durationMs = (t.startedAt && fin) ? (new Date(fin).getTime() - new Date(t.startedAt).getTime()) : null;
  t.status = computeTraceStatus(t);
  // 清理已删除任务的 runId
  t.runIds = t.runIds.filter((rid) => store.runs.some((x) => x.id === rid));
  t.taskIds = t.taskIds.filter((tid) => store.tasks.some((x) => x.id === tid));
}
function ensureTrace(kind, label, run, traceId) {
  let t = traceId ? store.traces.find((x) => x.id === traceId) : null;
  if (!t) {
    t = {
      id: uid(), kind, label: label || kind,
      startedAt: run ? run.startedAt : nowIso(), finishedAt: null,
      status: 'running', runIds: [], taskIds: [], durationMs: null, createdAt: nowIso()
    };
    store.traces.unshift(t);
    if (store.traces.length > TRACE_CAP) store.traces.length = TRACE_CAP;
  }
  if (run) {
    if (!t.runIds.includes(run.id)) t.runIds.push(run.id);
    if (!t.taskIds.includes(run.taskId)) t.taskIds.push(run.taskId);
  }
  updateTrace(t.id);
  return t.id;
}
function briefTrace(t) {
  return {
    id: t.id, kind: t.kind, label: t.label, startedAt: t.startedAt, finishedAt: t.finishedAt,
    status: t.status, runCount: (t.runIds || []).length, taskCount: (t.taskIds || []).length, durationMs: t.durationMs
  };
}

function finalizeRun(runId, status, exitCode, note) {
  const r = store.runs.find((x) => x.id === runId);
  if (!r) return;
  if (note) r.output.push({ t: Date.now(), s: 'err', d: note });
  r.status = status;
  r.exitCode = exitCode;
  r.finishedAt = nowIso();
  r.durationMs = Date.now() - new Date(r.startedAt).getTime();
  procs.delete(runId);
  captureResults(runId);
  save();
  if (r.traceId) updateTrace(r.traceId);
  broadcast({ type: 'run:end', run: briefRun(r) });
}

function pushOutput(runId, stream, text) {
  const r = store.runs.find((x) => x.id === runId);
  if (!r) return;
  r.output.push({ t: Date.now(), s: stream, d: text });
  let total = 0, cut = 0;
  for (let i = r.output.length - 1; i >= 0; i--) {
    total += r.output[i].d.length;
    if (total > OUTPUT_BUDGET) { cut = i + 1; break; }
  }
  if (cut) r.output.splice(0, cut);
  save();
  broadcast({ type: 'run:output', runId, stream, text });
}

function tryKillProc(p) {
  try {
    if (process.platform === 'win32') {
      try { spawn('taskkill', ['/pid', String(p.pid), '/T', '/F']); } catch (e) { /* ignore */ }
    }
    p.kill('SIGTERM');
    setTimeout(() => { try { p.kill('SIGKILL'); } catch (e) { /* ignore */ } }, 5000).unref();
  } catch (e) { /* ignore */ }
}

function addSkippedRun(task, triggerLabel) {
  const run = {
    id: uid(), taskId: task.id, taskName: task.name, engine: task.engine,
    command: task.command, trigger: 'scheduled', triggerLabel,
    status: 'skipped', startedAt: nowIso(), finishedAt: nowIso(), exitCode: null, durationMs: 0,
    output: [{ t: Date.now(), s: 'err', d: '[跳过] 该任务已有运行正在进行，本次定时触发跳过。' }]
  };
  store.runs.unshift(run);
  if (store.runs.length > RUN_CAP) store.runs.length = RUN_CAP;
  save();
  broadcast({ type: 'run:start', run: briefRun(run) });
  broadcast({ type: 'run:end', run: briefRun(run) });
}

/**
 * 启动一次任务运行。若该任务已有 running 运行则返回 { skipped: true }。
 */
async function startRun(task, trigger, triggerLabel) {
  // 启用守卫：同一任务已有活动运行（pending/running）则不重复创建
  const active = store.runs.find((r) => r.taskId === task.id && (r.status === 'pending' || r.status === 'running'));
  if (active) return { skipped: true };

  const run = {
    id: uid(), taskId: task.id, taskName: task.name, engine: task.engine,
    command: task.command, trigger, triggerLabel,
    status: 'pending', startedAt: nowIso(), finishedAt: null, exitCode: null, durationMs: null,
    worker: null, claimedAt: null, heartbeatAt: null, abortRequested: false,
    output: [], portResults: [], artifacts: {}
  };
  store.runs.unshift(run);
  if (store.runs.length > RUN_CAP) store.runs.length = RUN_CAP;
  await save();
  broadcast({ type: 'run:start', run: briefRun(run) });
  return { run };
}

// ---------- 节点（worker）管理 ----------
store.workers = store.workers || [];
function touchWorker(id) {
  const now = nowIso();
  let w = store.workers.find((x) => x.id === id);
  if (!w) {
    w = { id, firstSeen: now, lastSeen: now, currentRunId: null, _persistAt: 0 };
    store.workers.unshift(w);
    if (store.workers.length > 100) store.workers.length = 100;
  } else {
    w.lastSeen = now;
  }
  // 空闲轮询时也保持在线，但最多每 20s 落一次盘
  if (Date.now() - (w._persistAt || 0) > 20000) {
    w._persistAt = Date.now();
    save();
  }
  return w;
}

// 认领时组装上游输入（文本内联 / 文件给下载地址）
function claimInputs(task) {
  const envList = [];
  for (const e of store.edges.filter((x) => x.to === task.id)) {
    const src = store.tasks.find((t) => t.id === e.from);
    if (!src) continue;
    const inPort = (task.inputs || []).find((i) => i.id === e.toPort);
    const outPort = (src.outputs || []).find((o) => o.id === e.fromPort);
    const key = String((inPort && inPort.name) || e.toPort).replace(/[^\w\u4e00-\u9fa5]+/g, '_');
    const base = {
      port: e.toPort, portName: inPort ? inPort.name : e.toPort,
      spec: inPort ? inPort.spec || '' : '',
      tests: (inPort && inPort.tests) || [],
      contract: e.contract || '',
      srcTaskId: src.id, srcTaskName: src.name, srcPortName: outPort ? outPort.name : e.fromPort,
      srcPortSpec: outPort ? outPort.spec || '' : ''
    };
    const lastRun = store.runs.find((r) => r.taskId === src.id && r.status === 'success');
    if (!lastRun) { envList.push(Object.assign({}, base, { kind: 'missing', reason: '上游尚未成功运行' })); continue; }
    const res = (lastRun.portResults || []).find((pr) => pr.portId === e.fromPort);
    if (!res) { envList.push(Object.assign({}, base, { kind: 'missing', reason: '上游未产出该端口成果' })); continue; }
    if (res.valid === false) { envList.push(Object.assign({}, base, { kind: 'invalid', reason: '上游成果未通过接口测试', tests: res.tests || [] })); continue; }
    if (res.kind === 'text') {
      const txt = (lastRun.output || []).map((o) => o.d).join('').slice(0, 100000);
      envList.push(Object.assign({}, base, { env: '__IN_' + key, kind: 'text', value: txt, srcRunId: lastRun.id }));
    } else if (res.kind === 'file' && res.file) {
      envList.push(Object.assign({}, base, { env: '__IN_' + key + '_FILE', kind: 'file', srcRunId: lastRun.id, portId: res.portId, fileName: res.file.name, fileUrl: '/api/runs/' + lastRun.id + '/file?portId=' + encodeURIComponent(res.portId) }));
    }
  }
  return envList;
}

// 认领
function claimRun(workerId) {
  const run = store.runs.find((r) => r.status === 'pending');
  if (!run) return null;
  const task = store.tasks.find((t) => t.id === run.taskId);
  if (!task) return null;
  const engine = config.engines[task.engine] || config.engines.v8;
  run.status = 'running';
  run.worker = workerId;
  run.claimedAt = nowIso();
  run.heartbeatAt = nowIso();
  const w = touchWorker(workerId);
  w.currentRunId = run.id;
  // engineCwd 仅作提示（worker 在自己机器上解析）
  let engineCwd = engine.cwd ? (engine.cwd.split('/').pop() === '' ? '.' : path.relative(PROJECT_ROOT, engine.cwd)) : '.';
  if (engineCwd.startsWith('..')) engineCwd = '.';
  save();
  const graph = store.graphs.find((g) => g.id === task.graphId);
  return {
    runId: run.id, taskId: task.id, taskName: task.name, command: task.command,
    engine: task.engine, engineLabel: engine.label, engineCwd,
    graphId: task.graphId || null, graphName: graph ? graph.name : null,
    scope: task.scope || '', spec: task.spec || '',
    inputs: claimInputs(task),
    outputs: (task.outputs || []).map((o) => ({
      id: o.id, name: o.name, capture: o.capture || 'stdout', glob: o.glob || '*',
      spec: o.spec || '', tests: o.tests || []
    }))
  };
}

// ---------- 定时 ----------
function pad2(n) { return String(n).padStart(2, '0'); }

function parseCron(expr) {
  const parts = String(expr || '').trim().split(/\s+/);
  if (parts.length !== 5) return null;
  const days = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat'];
  function field(spec, min, max) {
    const set = new Set();
    for (const token of spec.split(',')) {
      if (token === '*') { for (let i = min; i <= max; i++) set.add(i); continue; }
      const m = token.match(/^(\d+|\*)\/(\d+)$/);
      if (m) { const step = parseInt(m[2], 10); for (let i = min; i <= max; i += step) set.add(i); continue; }
      let a = parseInt(token.split('-')[0], 10);
      let b = token.includes('-') ? parseInt(token.split('-')[1], 10) : a;
      if (Number.isNaN(a)) continue;
      for (let i = a; i <= b && i <= max; i++) set.add(i);
    }
    return set;
  }
  const minutes = field(parts[0], 0, 59);
  const hours = field(parts[1], 0, 23);
  const dom = field(parts[2], 1, 31);
  const months = field(parts[3], 1, 12);
  let dow = field(parts[4], 0, 7);
  if (dow.has(7)) dow.add(0);
  return { minutes, hours, dom, months, dow };
}

function cronMatches(c, d) {
  if (!c.months.has(d.getMonth() + 1)) return false;
  const domMatch = c.dom.has(d.getDate());
  const dowMatch = c.dow.has(d.getDay());
  const domFull = c.dom.size >= 31;
  const dowFull = c.dow.size >= 7;
  if (domFull && dowFull) return true;
  if (!domFull && !dowFull) return domMatch && dowMatch;
  // 一个受限时：用受限的那个（cron 语义：DOM 或 DOW，谁受限谁生效）
  return domFull ? dowMatch : domMatch;
}

function nextDailyAt(timeStr, after) {
  const m = String(timeStr || '').match(/^(\d{1,2}):(\d{2})$/);
  if (!m) return null;
  const h = parseInt(m[1], 10), mi = parseInt(m[2], 10);
  if (h > 23 || mi > 59) return null;
  let d = new Date(after);
  d.setHours(h, mi, 0, 0);
  if (d.getTime() <= after) d = new Date(d.getTime() + 24 * 3600 * 1000);
  return d.getTime();
}

function nextCronAt(expr, after) {
  const c = parseCron(expr);
  if (!c) return null;
  let t = new Date(Math.floor(after / 60000) * 60000 + 60000);
  let steps = 0;
  while (steps < 400000) {
    if (cronMatches(c, t)) return t.getTime();
    t = new Date(t.getTime() + 60000);
    steps++;
  }
  return null;
}

function scheduleLabel(s) {
  if (s.type === 'interval') return '每 ' + s.minutes + ' 分钟';
  if (s.type === 'daily') return '每日 ' + s.time;
  if (s.type === 'cron') return 'cron ' + s.cron;
  return '';
}

function computeNextAt(s, now) {
  if (!s.enabled) return null;
  if (s.type === 'interval') return now + (parseInt(s.minutes, 10) || 5) * 60000;
  if (s.type === 'daily') return nextDailyAt(s.time, now);
  if (s.type === 'cron') return nextCronAt(s.cron, now);
  return null;
}

function advanceSchedule(s, now) {
  if (s.type === 'interval') s.nextAt = now + (parseInt(s.minutes, 10) || 5) * 60000;
  else if (s.type === 'daily') s.nextAt = nextDailyAt(s.time, now + 1000);
  else if (s.type === 'cron') s.nextAt = nextCronAt(s.cron, now + 1000);
  else s.nextAt = null;
}

function fireSchedule(s) {
  const task = store.tasks.find((t) => t.id === s.taskId);
  if (!task) return;
  const label = scheduleLabel(s);
  const traceId = uid();
  const tr = { id: traceId, kind: 'scheduled', label, startedAt: nowIso(), finishedAt: null, status: 'running', runIds: [], taskIds: [], durationMs: null, createdAt: nowIso() };
  store.traces.unshift(tr);
  if (store.traces.length > TRACE_CAP) store.traces.length = TRACE_CAP;
  startRun(task, 'scheduled', label).then((res) => {
    if (res.run) {
      res.run.traceId = ensureTrace('scheduled', label, res.run, traceId);
    } else if (res.skipped) {
      addSkippedRun(task, label);
    }
    save();
  }).catch(() => { /* ignore */ });
}

// ---------- 图队列：按依赖顺序依次执行 ----------
const graphQueue = { running: false, remaining: [], currentTaskId: null };

function queueBroadcast() {
  broadcast({ type: 'queue', data: { running: graphQueue.running, remaining: graphQueue.remaining.length, currentTaskId: graphQueue.currentTaskId } });
}

function taskDeps(id) {
  return store.edges.filter((e) => e.to === id).map((e) => e.from);
}
function topoOrder(ids) {
  const valid = new Set(ids);
  const order = [];
  const visited = new Set();
  const stack = new Set();
  let cycle = false;
  function visit(id) {
    if (visited.has(id)) return;
    if (stack.has(id)) { cycle = true; return; }
    stack.add(id);
    for (const d of taskDeps(id)) if (valid.has(d)) visit(d);
    stack.delete(id);
    if (!visited.has(id)) { visited.add(id); order.push(id); }
  }
  for (const id of ids) visit(id);
  return { order, cycle };
}

const TERMINAL_RUN_STATUS = ['success', 'error', 'killed', 'aborted', 'skipped'];
function runAndWait(task, traceId) {
  return new Promise((resolve) => {
    startRun(task, 'graph', '运行图').then(({ run, skipped }) => {
      if (skipped || !run) return resolve({ skipped: true, run: null });
      run.traceId = ensureTrace('graph', '运行图', run, traceId);
      save();
      const timer = setInterval(() => {
        const r = store.runs.find((x) => x.id === run.id);
        if (r && TERMINAL_RUN_STATUS.includes(r.status)) { clearInterval(timer); resolve({ run }); }
      }, 400);
    });
  });
}

async function graphQueueLoop() {
  if (graphQueue.running) return;
  graphQueue.running = true;
  const traceId = uid();
  const tr = { id: traceId, kind: 'graph', label: '运行图', startedAt: nowIso(), finishedAt: null, status: 'running', runIds: [], taskIds: [], durationMs: null, createdAt: nowIso() };
  store.traces.unshift(tr);
  if (store.traces.length > TRACE_CAP) store.traces.length = TRACE_CAP;
  save();
  queueBroadcast();
  while (graphQueue.remaining.length) {
    const taskId = graphQueue.remaining.shift();
    graphQueue.currentTaskId = taskId;
    queueBroadcast();
    const task = store.tasks.find((t) => t.id === taskId);
    if (task) {
      const res = await runAndWait(task, traceId);
      if (res && res.skipped && !res.run) {
        // 记录“跳过”项并归入同一 trace
        const run = {
          id: uid(), taskId: task.id, taskName: task.name, engine: task.engine, command: task.command,
          trigger: 'graph', triggerLabel: '运行图',
          status: 'skipped', startedAt: nowIso(), finishedAt: nowIso(), exitCode: null, durationMs: 0,
          output: [{ t: Date.now(), s: 'err', d: '[跳过] 该任务已有运行正在进行。' }]
        };
        run.traceId = ensureTrace('graph', '运行图', run, traceId);
        store.runs.unshift(run);
        if (store.runs.length > RUN_CAP) store.runs.length = RUN_CAP;
        save();
        broadcast({ type: 'run:start', run: briefRun(run) });
        broadcast({ type: 'run:end', run: briefRun(run) });
      }
    }
  }
  graphQueue.currentTaskId = null;
  graphQueue.running = false;
  updateTrace(traceId);
  queueBroadcast();
}

function tick() {
  const now = Date.now();
  for (const s of store.schedules) {
    if (!s.enabled || typeof s.nextAt !== 'number') continue;
    if (now >= s.nextAt) {
      const missed = now - s.nextAt > MISS_GRACE_MS;
      if (!missed) fireSchedule(s);
      advanceSchedule(s, now);
      save();
    }
  }
}
setInterval(tick, 1000).unref();

// ---------- API ----------
function statePayload() {
  const tasks = store.tasks.map((t) => {
    const sched = store.schedules.filter((s) => s.taskId === t.id)
      .map((s) => ({ id: s.id, type: s.type, minutes: s.minutes, time: s.time, cron: s.cron, enabled: s.enabled, nextAt: s.nextAt, label: scheduleLabel(s) }));
    const lastRun = store.runs.find((r) => r.taskId === t.id);
    return Object.assign({}, t, { schedules: sched, lastRun: lastRun ? briefRun(lastRun) : null });
  });
  return {
    graphs: store.graphs.map((g) => ({ id: g.id, name: g.name, parentNodeId: g.parentNodeId || null, createdAt: g.createdAt })),
    tasks,
    edges: store.edges.map((e) => {
      const from = store.tasks.find((t) => t.id === e.from);
      const to = store.tasks.find((t) => t.id === e.to);
      return {
        id: e.id, from: e.from, fromPort: e.fromPort, to: e.to, toPort: e.toPort, carryName: e.carryName,
        contract: e.contract || '', graphId: e.graphId || null,
        fromName: from ? from.name : '?', toName: to ? to.name : '?',
        fromPortName: (from && (from.outputs || []).find((o) => o.id === e.fromPort) || {}).name || e.fromPort,
        toPortName: (to && (to.inputs || []).find((i) => i.id === e.toPort) || {}).name || e.toPort
      };
    }),
    workers: store.workers.map((w) => ({ id: w.id, firstSeen: w.firstSeen, lastSeen: w.lastSeen, currentRunId: w.currentRunId })),
    queue: { running: graphQueue.running, remaining: graphQueue.remaining.length, currentTaskId: graphQueue.currentTaskId },
    traces: store.traces.slice(0, 60).map(briefTrace),
    schedules: store.schedules.map((s) => ({ id: s.id, taskId: s.taskId, taskName: (store.tasks.find((t) => t.id === s.taskId) || {}).name || '?', type: s.type, minutes: s.minutes, time: s.time, cron: s.cron, enabled: s.enabled, nextAt: s.nextAt, label: scheduleLabel(s) })),
    runs: store.runs.slice(0, 100).map(briefRun),
    engines: Object.keys(config.engines).map((k) => ({ id: k, label: config.engines[k].label, color: config.engines[k].color })),
    now: nowIso()
  };
}

function json(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > 2 * 1024 * 1024) { reject(new Error('body too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => {
      try { resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {}); }
      catch (e) { reject(new Error('invalid JSON body')); }
    });
    req.on('error', reject);
  });
}

function readRaw(req, limit) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > limit) { reject(new Error('file too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8'
};

async function handleApi(req, res, method, p) {
  // SSE
  if (p === '/api/events') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no'
    });
    res.write('retry: 3000\n\n');
    res.write('data: ' + JSON.stringify({ type: 'state', data: statePayload() }) + '\n\n');
    sseClients.add(res);
    const hb = setInterval(() => { try { res.write(': ping\n\n'); } catch (e) { /* ignore */ } }, 25000);
    req.on('close', () => { clearInterval(hb); sseClients.delete(res); });
    return;
  }

  if (method === 'GET' && p === '/api/health') return json(res, 200, { ok: true, time: nowIso() });
  if (method === 'GET' && p === '/api/state') return json(res, 200, statePayload());

  if (method === 'GET' && p === '/api/templates') return json(res, 200, { templates: TEMPLATES });

  if (method === 'POST' && p === '/api/graph/run') {
    const body = await readBody(req).catch(() => ({}));
    let ids;
    if (body && Array.isArray(body.taskIds) && body.taskIds.length && body.taskIds.every((id) => store.tasks.some((t) => t.id === id))) {
      ids = body.taskIds;
    } else if (body && body.graphId && store.graphs.some((g) => g.id === body.graphId)) {
      ids = store.tasks.filter((t) => t.graphId === body.graphId).map((t) => t.id);
    } else {
      ids = store.tasks.map((t) => t.id);
    }
    const { order, cycle } = topoOrder(ids);
    graphQueue.remaining.push(...order);
    graphQueueLoop();
    return json(res, 200, { ok: true, order: order.length, cycle });
  }

  if (method === 'POST' && p === '/api/graph/stop') {
    graphQueue.remaining = [];
    // 中止当前运行（running → 请求节点停止；pending → 直接中止）
    if (graphQueue.currentTaskId) {
      const cur = store.runs.find((r) => r.taskId === graphQueue.currentTaskId && (r.status === 'running' || r.status === 'pending'));
      if (cur) abortRun(cur.id, '队列已停止');
    }
    queueBroadcast();
    return json(res, 200, { ok: true });
  }

  if (method === 'GET' && p === '/api/traces') {
    return json(res, 200, { traces: store.traces.slice(0, 200).map(briefTrace) });
  }

  if (method === 'GET' && p.startsWith('/api/traces/')) {
    const mm = p.match(/^\/api\/traces\/([\w-]+)$/);
    if (mm) {
      const t = store.traces.find((x) => x.id === mm[1]);
      if (!t) return json(res, 404, { error: 'trace not found' });
      const runs = t.runIds.map((rid) => {
        const r = store.runs.find((x) => x.id === rid);
        return r ? briefRun(r) : null;
      }).filter(Boolean);
      return json(res, 200, { trace: Object.assign(briefTrace(t), { runs }) });
    }
  }

  if (method === 'GET' && p === '/api/runs') return json(res, 200, { runs: store.runs.slice(0, 200).map(briefRun) });

  let m;
  if (method === 'GET' && (m = p.match(/^\/api\/runs\/([\w-]+)$/))) {
    const r = store.runs.find((x) => x.id === m[1]);
    if (!r) return json(res, 404, { error: 'run not found' });
    return json(res, 200, { run: r });
  }

  if (method === 'GET' && (m = p.match(/^\/api\/runs\/([\w-]+)\/file$/))) {
    const u = new URL(req.url, 'http://localhost');
    const portId = u.searchParams.get('portId');
    const r = store.runs.find((x) => x.id === m[1]);
    if (!r) return json(res, 404, { error: 'run not found' });
    const task = store.tasks.find((t) => t.id === r.taskId);
    const engine = config.engines[task ? task.engine : ''] || config.engines.v8;
    const res2 = (r.portResults || []).find((pr) => pr.portId === portId && pr.kind === 'file' && pr.file);
    if (!res2) return json(res, 404, { error: 'result not found' });
    const fp = path.resolve(res2.file.path);
    const inArtifacts = fp.startsWith(ARTIFACT_DIR);
    const inEngine = fp.startsWith(engine.cwd);
    if (!inArtifacts && !inEngine) return json(res, 403, { error: 'forbidden' });
    fs.readFile(fp, (err, data) => {
      if (err) { json(res, 404, { error: 'file missing' }); return; }
      res.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': 'attachment; filename="' + encodeURIComponent(res2.file.name) + '"'
      });
      res.end(data);
    });
    return;
  }

  if (method === 'POST' && p === '/api/runs/clear') {
    store.runs = [];
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { ok: true });
  }

  if (method === 'POST' && p === '/api/runs/claim') {
    const body = await readBody(req);
    const workerId = String(body.workerId || '').trim();
    if (!workerId) return json(res, 400, { error: 'workerId required' });
    touchWorker(workerId);
    const claimed = claimRun(workerId);
    if (!claimed) return json(res, 200, { run: null });
    const run = store.runs.find((x) => x.id === claimed.runId);
    broadcast({ type: 'run:update', run: briefRun(run) });
    return json(res, 200, { run: claimed });
  }

  if (method === 'POST' && (m = p.match(/^\/api\/runs\/([\w-]+)\/heartbeat$/))) {
    const body = await readBody(req);
    const workerId = String(body.workerId || '').trim();
    const run = store.runs.find((x) => x.id === m[1]);
    if (!run) return json(res, 404, { error: 'run not found' });
    if (run.status !== 'running') return json(res, 409, { error: 'run not running', status: run.status });
    if (run.worker && workerId && run.worker !== workerId) return json(res, 409, { error: 'claimed by another worker' });
    run.heartbeatAt = nowIso();
    touchWorker(workerId || run.worker || 'unknown');
    save();
    return json(res, 200, { ok: true, abort: !!run.abortRequested });
  }

  if (method === 'POST' && (m = p.match(/^\/api\/runs\/([\w-]+)\/submit$/))) {
    const body = await readBody(req);
    const run = store.runs.find((x) => x.id === m[1]);
    if (!run) return json(res, 404, { error: 'run not found' });
    if (TERMINAL_RUN_STATUS.includes(run.status)) return json(res, 200, { ok: true, already: true, run: briefRun(run) });
    if (run.worker && body.workerId && run.worker !== body.workerId) return json(res, 409, { error: 'claimed by another worker' });
    const stdout = String(body.stdout == null ? '' : body.stdout);
    const stderr = String(body.stderr == null ? '' : body.stderr);
    if (stdout) run.output.push({ t: Date.now(), s: 'out', d: stdout });
    if (stderr) run.output.push({ t: Date.now(), s: 'err', d: stderr });
    const aborted = !!body.aborted || !!run.abortRequested;
    // 接口测试结果（由节点执行并上报）：{portId, side:'in'|'out', testId, name, passed, exitCode, output}
    const testResults = Array.isArray(body.tests) ? body.tests.map((t) => ({
      portId: String(t.portId || ''), side: t.side === 'in' ? 'in' : 'out',
      testId: String(t.testId || ''), name: String(t.name || ''),
      passed: !!t.passed, exitCode: typeof t.exitCode === 'number' ? t.exitCode : null,
      output: String(t.output || '').slice(0, 4000)
    })) : [];
    run.tests = testResults;
    let status = aborted ? 'aborted' : (body.exitCode === 0 || body.exitCode === '0' ? 'success' : 'error');
    if (status === 'success') {
      const outFails = testResults.filter((t) => t.side === 'out' && !t.passed);
      if (outFails.length) status = 'error';
      const inFail = testResults.find((t) => t.side === 'in' && !t.passed);
      if (inFail) {
        run.output.push({ t: Date.now(), s: 'err', d: '[接口测试未通过] 输入端口 ' + (inFail.name || inFail.portId) + '：\n' + (inFail.output || '').slice(0, 2000) + '\n' });
        status = 'error';
      }
    }
    run.worker = run.worker || String(body.workerId || '');
    if (run.worker) touchWorker(run.worker);
    const w = store.workers.find((x) => x.id === run.worker);
    if (w) w.currentRunId = null;
    finalizeRun(run.id, status, typeof body.exitCode === 'number' ? body.exitCode : null, null);
    return json(res, 200, { ok: true, run: briefRun(run) });
  }

  if (method === 'POST' && (m = p.match(/^\/api\/runs\/([\w-]+)\/files\/([^\/]+)$/))) {
    const run = store.runs.find((x) => x.id === m[1]);
    if (!run) return json(res, 404, { error: 'run not found' });
    let name = decodeURIComponent(m[2]).replace(/[\\/:*?"<>|]/g, '_').slice(0, 120);
    if (!name || name === '.' || name === '..') return json(res, 400, { error: 'bad file name' });
    const data = await readRaw(req, 256 * 1024 * 1024);
    const dir = path.join(ARTIFACT_DIR, run.id);
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, name);
    fs.writeFileSync(fp, data);
    run.artifacts = run.artifacts || {};
    run.artifacts[name] = { path: fp, size: data.length };
    save();
    return json(res, 200, { ok: true, name, size: data.length });
  }

  if (method === 'POST' && (m = p.match(/^\/api\/runs\/([\w-]+)\/kill$/))) {
    const runId = m[1];
    const run = store.runs.find((x) => x.id === runId);
    if (!run) return json(res, 404, { error: 'run not found' });
    if (run.status === 'pending') {
      abortRun(runId, '手动中止');
    } else if (run.status === 'running') {
      run.abortRequested = true;
      save();
      broadcast({ type: 'run:update', run: briefRun(run) });
    }
    return json(res, 200, { ok: true, runId });
  }

  // ---------- 图（graph）管理 ----------
  if (method === 'POST' && p === '/api/graphs') {
    const body = await readBody(req);
    const name = String(body.name || '').trim() || ('子图 ' + (store.graphs.length));
    const parentNodeId = body.parentNodeId ? String(body.parentNodeId) : null;
    const parentNode = parentNodeId ? store.tasks.find((t) => t.id === parentNodeId) : null;
    if (parentNodeId && !parentNode) return json(res, 400, { error: '父节点不存在' });
    const g = { id: uid(), name, parentNodeId, createdAt: nowIso() };
    store.graphs.push(g);
    if (parentNode) parentNode.subgraphId = g.id;
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { graph: g });
  }

  if ((method === 'POST' || method === 'PUT') && (m = p.match(/^\/api\/graphs\/([\w-]+)$/))) {
    const g = store.graphs.find((x) => x.id === m[1]);
    if (!g) return json(res, 404, { error: 'graph not found' });
    const body = await readBody(req);
    if (typeof body.name === 'string' && body.name.trim()) g.name = body.name.trim();
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { graph: g });
  }

  if (method === 'DELETE' && (m = p.match(/^\/api\/graphs\/([\w-]+)$/))) {
    const gid = m[1];
    if (!store.graphs.some((x) => x.id === gid)) return json(res, 404, { error: 'graph not found' });
    if (store.graphs[0].id === gid) return json(res, 400, { error: '不能删除主图' });
    // 收集自身 + 所有子孙图
    const doomed = new Set([gid]);
    let grew = true;
    while (grew) {
      grew = false;
      for (const g of store.graphs) {
        if (g.parentNodeId && !doomed.has(g.id)) {
          const owner = store.tasks.find((t) => t.id === g.parentNodeId);
          if (doomed.has(g.id) || doomed.has(g.parentNodeId) || (owner && doomed.has(owner.graphId))) { doomed.add(g.id); grew = true; }
        }
      }
    }
    const doomedNodes = store.tasks.filter((t) => doomed.has(t.graphId)).map((t) => t.id);
    store.tasks = store.tasks.filter((t) => !doomed.has(t.graphId));
    store.edges = store.edges.filter((e) => !doomedNodes.includes(e.from) && !doomedNodes.includes(e.to));
    store.graphs = store.graphs.filter((g) => !doomed.has(g.id));
    store.schedules = store.schedules.filter((sc) => !doomedNodes.includes(sc.taskId));
    // 清理父节点引用
    for (const t of store.tasks) {
      if (t.subgraphId && doomed.has(t.subgraphId)) t.subgraphId = null;
    }
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { ok: true, deletedGraphs: doomed.size, deletedNodes: doomedNodes.length });
  }

  if (method === 'POST' && p === '/api/tasks') {
    const body = await readBody(req);
    const name = String(body.name || '').trim();
    if (!name) return json(res, 400, { error: '任务名不能为空' });
    const task = {
      id: uid(),
      name,
      description: String(body.description || '').trim(),
      command: String(body.command || '').trim() || ('echo "[TaskPad] ' + name.replace(/"/g, '') + ' 无命令，仅自检"'),
      engine: config.engines[body.engine] ? String(body.engine) : 'v8',
      color: String(body.color || config.engines[config.engines.v8 ? 'v8' : Object.keys(config.engines)[0]].color || '#3b82f6'),
      x: typeof body.x === 'number' ? body.x : 0,
      y: typeof body.y === 'number' ? body.y : 0,
      graphId: store.graphs.some((g) => g.id === body.graphId) ? String(body.graphId) : store.graphs[0].id,
      scope: String(body.scope || ''),
      spec: String(body.spec || ''),
      subgraphId: null,
      createdAt: nowIso()
    };
    if (Array.isArray(body.inputs)) task.inputs = normalizePorts(body.inputs);
    if (Array.isArray(body.outputs)) task.outputs = normalizePorts(body.outputs, true);
    store.tasks.push(task);
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { task });
  }

  if ((method === 'POST' || method === 'PUT') && (m = p.match(/^\/api\/tasks\/([\w-]+)$/))) {
    const task = store.tasks.find((t) => t.id === m[1]);
    if (!task) return json(res, 404, { error: 'task not found' });
    const body = await readBody(req);
    ['name', 'description', 'command', 'color'].forEach((k) => {
      if (typeof body[k] === 'string' && body[k] !== undefined) task[k] = body[k];
    });
    if (typeof body.name === 'string' && body.name.trim()) task.name = body.name.trim();
    if (typeof body.engine === 'string' && config.engines[body.engine]) task.engine = body.engine;
    if (typeof body.x === 'number') task.x = body.x;
    if (typeof body.y === 'number') task.y = body.y;
    if (typeof body.scope === 'string') task.scope = body.scope;
    if (typeof body.spec === 'string') task.spec = body.spec;
    if (typeof body.graphId === 'string' && store.graphs.some((g) => g.id === body.graphId)) task.graphId = body.graphId;
    if (Array.isArray(body.inputs)) task.inputs = normalizePorts(body.inputs);
    if (Array.isArray(body.outputs)) task.outputs = normalizePorts(body.outputs, true);
    // 端口被删除时清理相关边
    const keepInputs = new Set((task.inputs || []).map((i) => i.id));
    const keepOutputs = new Set((task.outputs || []).map((o) => o.id));
    store.edges = store.edges.filter((e) => !(e.to === task.id && !keepInputs.has(e.toPort)) && !(e.from === task.id && !keepOutputs.has(e.fromPort)));
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { task });
  }

  if (method === 'DELETE' && (m = p.match(/^\/api\/tasks\/([\w-]+)$/))) {
    const i = store.tasks.findIndex((t) => t.id === m[1]);
    if (i < 0) return json(res, 404, { error: 'task not found' });
    store.tasks.splice(i, 1);
    store.edges = store.edges.filter((e) => e.from !== m[1] && e.to !== m[1]);
    store.schedules = store.schedules.filter((s) => s.taskId !== m[1]);
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { ok: true });
  }

  if (method === 'POST' && (m = p.match(/^\/api\/tasks\/([\w-]+)\/run$/))) {
    const task = store.tasks.find((t) => t.id === m[1]);
    if (!task) return json(res, 404, { error: 'task not found' });
    const res2 = await startRun(task, 'manual', '手动');
    if (res2.skipped) return json(res, 409, { error: '该任务已在运行中' });
    res2.run.traceId = ensureTrace('manual', '手动', res2.run);
    save();
    return json(res, 200, { runId: res2.run.id });
  }

  if (method === 'POST' && p === '/api/edges') {
    const body = await readBody(req);
    const from = store.tasks.find((t) => t.id === body.from);
    const to = store.tasks.find((t) => t.id === body.to);
    if (!from || !to) return json(res, 400, { error: '源或目标任务不存在' });
    if (from.id === to.id) return json(res, 400, { error: '不能连接自身' });
    const outPort = (from.outputs || []).find((o) => o.id === body.fromPort);
    const inPort = (to.inputs || []).find((i) => i.id === body.toPort);
    if (!outPort) return json(res, 400, { error: '源输出端口不存在' });
    if (!inPort) return json(res, 400, { error: '目标输入端口不存在' });
    if (store.edges.some((e) => e.from === from.id && e.fromPort === outPort.id && e.to === to.id && e.toPort === inPort.id)) {
      return json(res, 400, { error: '该连线已存在' });
    }
    const edge = {
      id: uid(), from: from.id, fromPort: outPort.id, to: to.id, toPort: inPort.id,
      carryName: outPort.name, graphId: to.graphId || from.graphId, contract: String(body.contract || '')
    };
    store.edges.push(edge);
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { edge });
  }

  if ((method === 'POST' || method === 'PUT') && (m = p.match(/^\/api\/edges\/([\w-]+)$/))) {
    const e = store.edges.find((x) => x.id === m[1]);
    if (!e) return json(res, 404, { error: 'edge not found' });
    const body = await readBody(req);
    if (typeof body.contract === 'string') e.contract = body.contract;
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { edge: e });
  }

  if (method === 'DELETE' && (m = p.match(/^\/api\/edges\/([\w-]+)$/))) {
    const i = store.edges.findIndex((e) => e.id === m[1]);
    if (i < 0) return json(res, 404, { error: 'edge not found' });
    store.edges.splice(i, 1);
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { ok: true });
  }

  if (method === 'POST' && p === '/api/schedules') {
    const body = await readBody(req);
    const task = store.tasks.find((t) => t.id === body.taskId);
    if (!task) return json(res, 400, { error: 'task not found' });
    const type = ['interval', 'daily', 'cron'].includes(body.type) ? body.type : 'interval';
    const s = {
      id: uid(), taskId: task.id, type, enabled: body.enabled !== false,
      minutes: parseInt(body.minutes, 10) || 5,
      time: String(body.time || '08:00'),
      cron: String(body.cron || '0 * * * *'),
      nextAt: null, createdAt: nowIso()
    };
    s.nextAt = computeNextAt(s, Date.now());
    store.schedules.push(s);
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { schedule: s });
  }

  if ((method === 'POST' || method === 'PUT') && (m = p.match(/^\/api\/schedules\/([\w-]+)$/))) {
    const s = store.schedules.find((x) => x.id === m[1]);
    if (!s) return json(res, 404, { error: 'schedule not found' });
    const body = await readBody(req);
    if (['interval', 'daily', 'cron'].includes(body.type)) s.type = body.type;
    if (typeof body.enabled === 'boolean') s.enabled = body.enabled;
    if (body.minutes) s.minutes = parseInt(body.minutes, 10) || 5;
    if (body.time) s.time = String(body.time);
    if (body.cron) s.cron = String(body.cron);
    s.nextAt = computeNextAt(s, Date.now());
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { schedule: s });
  }

  if (method === 'DELETE' && (m = p.match(/^\/api\/schedules\/([\w-]+)$/))) {
    const i = store.schedules.findIndex((x) => x.id === m[1]);
    if (i < 0) return json(res, 404, { error: 'schedule not found' });
    store.schedules.splice(i, 1);
    save();
    broadcast({ type: 'state', data: statePayload() });
    return json(res, 200, { ok: true });
  }

  return json(res, 404, { error: 'not found', path: p });
}

function serveStatic(req, res, urlPath) {
  if (urlPath === '/favicon.ico') { res.writeHead(204); res.end(); return; }
  let rel = decodeURIComponent(urlPath);
  if (rel === '/') rel = '/index.html';
  const file = path.normalize(path.join(PUBLIC, rel));
  if (!file.startsWith(PUBLIC)) { res.writeHead(403); res.end('forbidden'); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }); res.end('404 Not Found'); return; }
    const ext = path.extname(file).toLowerCase();
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cache-Control': 'no-store'
    });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  const u = new URL(req.url, 'http://localhost');
  const p = u.pathname.replace(/\/+$/, '') || '/';
  if (p.startsWith('/api/') || p === '/api') {
    handleApi(req, res, req.method, p).catch((err) => {
      try { json(res, 500, { error: String(err && err.message || err) }); } catch (e) { /* ignore */ }
    });
    return;
  }
  serveStatic(req, res, p);
});

// ---------- 任务模板（节点库） ----------
const TEMPLATES = [
  {
    name: 'v8 离线 demo', engine: 'v8', color: '#3b82f6',
    command: 'python3 -m dag8 demo',
    description: '离线 demo：command 节点多波次并行，不调模型。'
  },
  {
    name: 'v8 lint + dry-run', engine: 'v8', color: '#22c55e',
    command: 'python3 -m dag8 lint configs/demo.json && python3 -m dag8 run configs/demo.json --dry-run',
    description: '静态检查配置 + dry-run 验证占位符渲染。'
  },
  {
    name: 'v8 三边流水线', engine: 'v8', color: '#eab308',
    command: 'python3 -m dag8 run configs/demo.json --workers 2 --rpm 3',
    description: '起草 → 挑错 → 定稿；波次并行 + 限速。'
  },
  {
    name: 'v9 知识推理 demo', engine: 'v9', color: '#a855f7',
    command: 'python3 -m v9.demo',
    description: '抓 Wikipedia 三实体 → 判定关系 → 写知识库。'
  },
  {
    name: 'v9 测试套件', engine: 'v9', color: '#06b6d4',
    command: 'python3 -m pytest v9/tests -q',
    description: '运行 v9 全部测试（20 个用例）。'
  },
  {
    name: '自检 echo', engine: 'v8', color: '#f43f5e',
    command: 'echo "TaskPad 自检 OK" && date',
    description: '最小示例，什么都不依赖。'
  }
];

// ---------- 首次初始化：预置示例任务 ----------
function seedIfEmpty() {
  if (storeFileExisted || store.tasks.length > 0) return;
  const defs = [
      {
        name: 'v8 离线 demo（最快验证）', engine: 'v8', color: '#3b82f6',
        command: 'python3 -m dag8 demo',
        description: '离线 demo：command 节点多波次并行，不调模型。用于确认服务链路正常。',
        x: 80, y: 60, deps: []
      },
      {
        name: 'v8 lint + dry-run', engine: 'v8', color: '#22c55e',
        command: 'python3 -m dag8 lint configs/demo.json && python3 -m dag8 run configs/demo.json --dry-run',
        description: '先静态检查配置，再 dry-run 验证占位符渲染（不调 pi）。',
        x: 520, y: 60, deps: []
      },
      {
        name: 'v8 三边流水线', engine: 'v8', color: '#eab308',
        command: 'python3 -m dag8 run configs/demo.json --workers 2 --rpm 3',
        description: '起草 → 挑错 → 定稿；依赖 lint 通过后再跑。',
        x: 120, y: 420, deps: []
      },
      {
        name: 'v9 知识推理 demo', engine: 'v9', color: '#a855f7',
        command: 'python3 -m v9.demo',
        description: 'v9 开放世界推理：抓 Wikipedia 三实体 → 判定关系 → 写知识库。需要外网。',
        x: 560, y: 420, deps: []
      },
      {
        name: 'v9 测试套件', engine: 'v9', color: '#06b6d4',
        command: 'python3 -m pytest v9/tests -q',
        description: '运行 v9 全部测试（20 个用例，约 1s）。',
        x: 980, y: 60, deps: []
      },
      {
        name: '自检：echo 任务', engine: 'v8', color: '#f43f5e',
        command: 'echo "TaskPad 自检 OK" && date',
        description: '最小示例：输出一句话和当前时间，什么都不依赖。',
        x: 980, y: 420, deps: []
      }
    ];
    store.tasks = defs.map((d) => Object.assign({ id: uid(), description: '', createdAt: nowIso(), deps: [] }, d));
    for (const t of store.tasks) {
      t.graphId = store.graphs[0].id;
      t.scope = '示例任务：在 v8/v9 项目内执行只读检查';
      t.spec = '按任务描述执行命令，输出结果供下游消费。';
      t.subgraphId = null;
      t.inputs = [{ id: 'in', name: 'dep', spec: '上游任意成果（可选）', tests: [] }];
      t.outputs = [{ id: 'out', name: 'out', spec: '命令 stdout 文本', tests: [], capture: 'stdout' }];
    }
    // 示例数据流：lint 的 stdout → 三边流水线 的 dep 输入（运行图时先跑 lint）
    const lintId = (store.tasks.find((t) => t.name.indexOf('lint') >= 0) || {}).id;
    const pipe = store.tasks.find((t) => t.name.indexOf('三边') >= 0);
    if (lintId && pipe) {
      store.edges.push({ id: uid(), from: lintId, fromPort: 'out', to: pipe.id, toPort: 'in', carryName: 'out' });
    }
    save();
}
migrateStore();
migrateGraphs();
seedIfEmpty();

server.listen(config.port, () => {
  console.log('TaskPad 任务白板已启动');
  console.log('  地址:   http://127.0.0.1:' + config.port);
  console.log('  数据:   ' + STORE_FILE);
  console.log('  引擎:   ' + Object.keys(config.engines).map((k) => k + '→' + config.engines[k].cwd).join(' | '));
});