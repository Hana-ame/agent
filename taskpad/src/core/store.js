'use strict';

const fs = require('fs');
const path = require('path');
const { uid, nowIso } = require('./ids');
const { normalizePorts, normalizeTests } = require('./ports');
const { seedIfEmpty } = require('./seed');

const ARRAYS = ['graphs', 'tasks', 'edges', 'schedules', 'runs', 'traces', 'workers'];

/**
 * 存储层：唯一持有 data/store.json 的模块。
 *
 * 设计要点（参考 ComfyUI 的 model_management / n8n 的 repositories）：
 *   - 内存里就一份 db，其他模块拿到的是引用，不做深拷贝；
 *   - 写盘串行化 + .tmp → rename 原子替换，杜绝半截文件；
 *   - 容量截断（runs/traces/workers）在写入前统一做，不让数组无限增长；
 *   - 迁移只在启动时跑一次，之后所有数据都视为「已归一」。
 *
 * 本模块不感知 HTTP、不感知 SSE、不感知 run 语义。
 */
function createStore(cfg, deps) {
  const log = (deps && deps.log) || console;

  const db = {
    graphs: [], tasks: [], edges: [], schedules: [], runs: [], traces: [], workers: []
  };

  let fileExisted = fs.existsSync(cfg.storeFile);
  if (fileExisted) {
    try {
      const raw = JSON.parse(fs.readFileSync(cfg.storeFile, 'utf8'));
      if (raw && typeof raw === 'object') Object.assign(db, raw);
    } catch (e) {
      // 坏文件不覆盖，改名留证后再从空库启动
      const backup = cfg.storeFile + '.backup-' + Date.now();
      try { fs.renameSync(cfg.storeFile, backup); } catch (_) { /* ignore */ }
      log.error('[store] store.json 解析失败，已备份到 ' + backup);
    }
  }

  for (const k of ARRAYS) if (!Array.isArray(db[k])) db[k] = [];

  // ---------- 迁移 ----------
  function migrateLegacyDeps() {
    for (const t of db.tasks) {
      if (!Array.isArray(t.inputs) || !t.inputs.length) t.inputs = [{ id: 'in', name: 'dep' }];
      if (!Array.isArray(t.outputs) || !t.outputs.length) t.outputs = [{ id: 'out', name: 'out', capture: 'stdout' }];
      if (!Array.isArray(t.deps) || !t.deps.length) { t.deps = []; continue; }
      for (const d of t.deps) {
        const src = db.tasks.find((x) => x.id === d);
        const fromPort = src && src.outputs && src.outputs[0] ? src.outputs[0].id : 'out';
        if (!db.edges.some((e) => e.from === d && e.to === t.id)) {
          db.edges.push({ id: uid(), from: d, fromPort, to: t.id, toPort: 'in', carryName: 'out' });
        }
      }
      t.deps = [];
    }
  }

  function migrateGraphs() {
    if (!db.graphs.length) {
      db.graphs = [{ id: uid(), name: '主图', parentNodeId: null, createdAt: nowIso() }];
    }
    const rootId = db.graphs[0].id;
    for (const t of db.tasks) {
      if (!t.graphId) t.graphId = rootId;
      if (typeof t.scope !== 'string') t.scope = '';
      if (typeof t.spec !== 'string') t.spec = '';
      if (typeof t.group !== 'string') t.group = '';
      if (t.subgraphId === undefined) t.subgraphId = null;
      t.inputs = normalizePorts(t.inputs || [], false);
      t.outputs = normalizePorts(t.outputs || [], true);
      t.deps = [];
    }
    for (const e of db.edges) {
      if (!e.graphId) e.graphId = rootId;
      if (typeof e.contract !== 'string') e.contract = '';
      if (!e.name) e.name = e.label || '接口';
      if (!e.label) e.label = e.name;
      if (!Array.isArray(e.tests)) e.tests = normalizeTests(e.tests || []);
      else e.tests = normalizeTests(e.tests);
    }
  }

  function migrate() {
    migrateLegacyDeps();
    migrateGraphs();
    if (!fileExisted) seedIfEmpty(db, cfg);
    trim();
    return db;
  }

  // ---------- 容量控制 ----------
  function trim() {
    const L = cfg.limits;
    if (db.runs.length > L.runCap) db.runs.length = L.runCap;
    if (db.traces.length > L.traceCap) db.traces.length = L.traceCap;
    if (db.workers.length > L.workerCap) db.workers.length = L.workerCap;
  }

  // ---------- 原子写盘（串行队列） ----------
  let chain = Promise.resolve();
  let pending = 0;

  function save() {
    pending++;
    chain = chain
      .then(() => new Promise((resolve, reject) => {
        trim();
        const tmp = cfg.storeFile + '.tmp';
        fs.writeFile(tmp, JSON.stringify(db), (err) => {
          if (err) return reject(err);
          fs.rename(tmp, cfg.storeFile, (e) => (e ? reject(e) : resolve()));
        });
      }))
      .then(() => { pending--; })
      .catch((err) => {
        pending--;
        log.error('[store] save failed: ' + (err && err.message));
      });
    return chain;
  }

  /** 等待所有排队写入落盘（退出/测试用） */
  function flush() {
    return chain;
  }

  // ---------- 查询辅助（避免各处重复 find） ----------
  const findTask = (id) => db.tasks.find((t) => t.id === id) || null;
  const findGraph = (id) => db.graphs.find((g) => g.id === id) || null;
  const findEdge = (id) => db.edges.find((e) => e.id === id) || null;
  const findRun = (id) => db.runs.find((r) => r.id === id) || null;
  const findTrace = (id) => db.traces.find((t) => t.id === id) || null;
  const findSchedule = (id) => db.schedules.find((s) => s.id === id) || null;

  function init() {
    fs.mkdirSync(cfg.dataDir, { recursive: true });
    fs.mkdirSync(cfg.artifactDir, { recursive: true });
    migrate();
    // 启动时立刻落盘一次：把迁移结果固化，避免每次启动重复推断
    save();
    return db;
  }

  return {
    db,
    init,
    save,
    flush,
    trim,
    fileExisted: () => fileExisted,
    findTask, findGraph, findEdge, findRun, findTrace, findSchedule
  };
}

module.exports = { createStore };
