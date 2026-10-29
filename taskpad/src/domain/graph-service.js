'use strict';

const { uid, nowIso } = require('../core/ids');
const { normalizePorts, normalizeTests } = require('../core/ports');
const { emitSafe } = require('../core/events');
const { badRequest, notFound } = require('../core/errors');

const KINDS = ['code', 'data', 'net'];
const COLORS = ['#3b82f6', '#22c55e', '#eab308', '#a855f7', '#06b6d4', '#f43f5e', '#f97316', '#8b5cf6'];

/**
 * 图 / 模块 / 接口 的领域服务。
 *
 * 数据模型：Graph（项目或子图）→ Task（模块）→ Edge（模块间接口契约）→ 子图（Task 内部流程）。
 * 本模块只管「结构正确性 + 引用完整性」，不碰运行状态（那是 run-service 的事）。
 * 所有改结构的入口都在这里，保证级联清理只有一份实现。
 */
function createGraphService({ db, store, bus, cfg }) {
  function touch() {
    store.save();
    emitSafe(bus, 'state:dirty', {});
  }

  const colorOf = (group) => {
    let h = 0;
    const gs = String(group || '');
    for (let i = 0; i < gs.length; i++) h = (h * 31 + gs.charCodeAt(i)) >>> 0;
    return COLORS[h % COLORS.length];
  };

  // ---------- 图 ----------
  function createGraph(body) {
    const name = String(body.name || '').trim() || ('子图 ' + db.graphs.length);
    const parentNodeId = body.parentNodeId ? String(body.parentNodeId) : null;
    const parentNode = parentNodeId ? db.tasks.find((t) => t.id === parentNodeId) : null;
    if (parentNodeId && !parentNode) throw badRequest('父节点不存在');

    const g = { id: uid(), name, parentNodeId, createdAt: nowIso() };
    db.graphs.push(g);
    // 双向绑定：父节点的 subgraphId 指回子图
    if (parentNode) parentNode.subgraphId = g.id;
    touch();
    return g;
  }

  function updateGraph(id, body) {
    const g = db.graphs.find((x) => x.id === id);
    if (!g) throw notFound('graph not found');
    if (typeof body.name === 'string' && body.name.trim()) g.name = body.name.trim();
    touch();
    return g;
  }

  /**
   * 级联删除：子图 → 子图内的节点 → 相关边 / 定时任务 → 上层节点的 subgraphId 置空。
   * 用不动点迭代求后代闭包，避免递归栈和顺序依赖。
   */
  function deleteGraphCascade(gid) {
    const root = db.graphs.find((x) => x.id === gid);
    if (!root) throw notFound('graph not found');
    if (db.graphs[0].id === gid) throw badRequest('不能删除主图');

    const doomed = new Set([gid]);
    let grew = true;
    while (grew) {
      grew = false;
      for (const g of db.graphs) {
        if (doomed.has(g.id) || !g.parentNodeId) continue;
        const owner = db.tasks.find((t) => t.id === g.parentNodeId);
        if (owner && doomed.has(owner.graphId)) { doomed.add(g.id); grew = true; }
      }
    }

    const doomedNodes = db.tasks.filter((t) => doomed.has(t.graphId)).map((t) => t.id);
    const nodeSet = new Set(doomedNodes);
    db.tasks = db.tasks.filter((t) => !doomed.has(t.graphId));
    db.edges = db.edges.filter((e) => !nodeSet.has(e.from) && !nodeSet.has(e.to));
    db.graphs = db.graphs.filter((g) => !doomed.has(g.id));
    db.schedules = db.schedules.filter((s) => !nodeSet.has(s.taskId));
    for (const t of db.tasks) {
      if (t.subgraphId && doomed.has(t.subgraphId)) t.subgraphId = null;
    }
    touch();
    return { deletedGraphs: doomed.size, deletedNodes: doomedNodes.length };
  }

  function graphTaskIds(graphId) {
    return db.tasks.filter((t) => t.graphId === graphId).map((t) => t.id);
  }

  // ---------- 模块（节点） ----------
  function createTask(body) {
    const name = String(body.name || '').trim();
    if (!name) throw badRequest('任务名不能为空');

    const engine = cfg.engines[body.engine] ? String(body.engine) : (cfg.engines.v8 ? 'v8' : cfg.engineIds[0]);
    const task = {
      id: uid(),
      name,
      description: String(body.description || '').trim(),
      command: String(body.command || '').trim() || ('echo "[TaskPad] ' + name.replace(/"/g, '') + ' 无命令，仅自检"'),
      engine,
      color: String(body.color || (cfg.engines[engine] && cfg.engines[engine].color) || '#3b82f6'),
      x: typeof body.x === 'number' ? body.x : 0,
      y: typeof body.y === 'number' ? body.y : 0,
      graphId: db.graphs.some((g) => g.id === body.graphId) ? String(body.graphId) : db.graphs[0].id,
      scope: String(body.scope || ''),
      spec: String(body.spec || ''),
      group: String(body.group || ''),
      subgraphId: null,
      createdAt: nowIso(),
      // 端口必须有默认值：否则新模块没有 outputs，运行结束时捕获不到任何成果
      inputs: normalizePorts(Array.isArray(body.inputs) ? body.inputs : [], false),
      outputs: normalizePorts(Array.isArray(body.outputs) ? body.outputs : [], true)
    };
    db.tasks.push(task);
    touch();
    return task;
  }

  function updateTask(id, body) {
    const task = db.tasks.find((t) => t.id === id);
    if (!task) throw notFound('task not found');

    for (const k of ['description', 'command', 'color']) {
      if (typeof body[k] === 'string') task[k] = body[k];
    }
    if (typeof body.name === 'string' && body.name.trim()) task.name = body.name.trim();
    if (typeof body.engine === 'string' && cfg.engines[body.engine]) task.engine = body.engine;
    if (typeof body.x === 'number') task.x = body.x;
    if (typeof body.y === 'number') task.y = body.y;
    if (typeof body.scope === 'string') task.scope = body.scope;
    if (typeof body.spec === 'string') task.spec = body.spec;
    if (typeof body.group === 'string') task.group = body.group;
    if (typeof body.graphId === 'string' && db.graphs.some((g) => g.id === body.graphId)) task.graphId = body.graphId;
    if (Array.isArray(body.inputs)) task.inputs = normalizePorts(body.inputs, false);
    if (Array.isArray(body.outputs)) task.outputs = normalizePorts(body.outputs, true);

    // 端口被删掉 → 悬挂的边必须同时清掉，否则拓扑排序和前端连线都会炸
    const keepInputs = new Set((task.inputs || []).map((i) => i.id));
    const keepOutputs = new Set((task.outputs || []).map((o) => o.id));
    db.edges = db.edges.filter((e) => {
      if (e.to === task.id && !keepInputs.has(e.toPort)) return false;
      if (e.from === task.id && !keepOutputs.has(e.fromPort)) return false;
      return true;
    });
    touch();
    return task;
  }

  function deleteTask(id) {
    const i = db.tasks.findIndex((t) => t.id === id);
    if (i < 0) throw notFound('task not found');
    db.tasks.splice(i, 1);
    db.edges = db.edges.filter((e) => e.from !== id && e.to !== id);
    db.schedules = db.schedules.filter((s) => s.taskId !== id);
    touch();
    return { ok: true };
  }

  // ---------- 接口（边） ----------
  function createEdge(body) {
    const from = db.tasks.find((t) => t.id === body.from);
    const to = db.tasks.find((t) => t.id === body.to);
    if (!from || !to) throw badRequest('源或目标任务不存在');
    if (from.id === to.id) throw badRequest('不能连接自身');

    const name = String(body.name || body.label || '接口');
    if (db.edges.some((e) => e.from === from.id && e.to === to.id && (e.name || e.label) === name)) {
      throw badRequest('该接口关系已存在');
    }

    const edge = {
      id: uid(),
      from: from.id, fromPort: body.fromPort ? String(body.fromPort) : 'out',
      to: to.id, toPort: body.toPort ? String(body.toPort) : 'in',
      carryName: name,
      graphId: to.graphId || from.graphId,
      name, label: name,
      kind: KINDS.includes(body.kind) ? body.kind : 'code',
      proto: String(body.proto || 'http'),
      auth: String(body.auth || 'no'),
      spec: String(body.spec || ''),
      tests: normalizeTests(body.tests),
      contract: String(body.contract || '')
    };
    db.edges.push(edge);
    touch();
    return edge;
  }

  function updateEdge(id, body) {
    const e = db.edges.find((x) => x.id === id);
    if (!e) throw notFound('edge not found');
    if (typeof body.contract === 'string') e.contract = body.contract;
    if (typeof body.label === 'string') e.label = body.label;
    if (typeof body.name === 'string') e.name = body.name;
    if (typeof body.spec === 'string') e.spec = body.spec;
    if (typeof body.proto === 'string') e.proto = body.proto;
    if (typeof body.auth === 'string') e.auth = body.auth;
    if (KINDS.includes(body.kind)) e.kind = body.kind;
    if (Array.isArray(body.tests)) e.tests = normalizeTests(body.tests);
    touch();
    return e;
  }

  function deleteEdge(id) {
    const i = db.edges.findIndex((e) => e.id === id);
    if (i < 0) throw notFound('edge not found');
    db.edges.splice(i, 1);
    touch();
    return { ok: true };
  }

  // ---------- 架构图导入 ----------
  function importArchitecture(body) {
    const nodesIn = Array.isArray(body.nodes) ? body.nodes : [];
    if (!nodesIn.length) throw badRequest('JSON 中没有 nodes 数组');

    const name = String(body.project || '导入的架构').slice(0, 60);
    const g = { id: uid(), name, parentNodeId: null, createdAt: nowIso() };
    db.graphs.push(g);

    const idMap = new Map();
    for (const n of nodesIn) {
      const label = String(n.label || n.id || '节点');
      const task = {
        id: uid(),
        name: label.slice(0, 80),
        description: n.group ? ('[' + n.group + ']') : '',
        group: String(n.group || ''),
        command: 'echo "[TaskPad] ' + label.replace(/"/g, '').slice(0, 60) + ' 无命令"',
        engine: 'v8',
        color: colorOf(n.group),
        x: Math.round((Number(n.x) || 0) * 1.7),
        y: Math.round((Number(n.y) || 0) * 1.7),
        graphId: g.id,
        scope: '来自架构图：' + name,
        spec: String(n.desc || ''),
        subgraphId: null,
        createdAt: nowIso(),
        inputs: normalizePorts([], false),
        outputs: normalizePorts([], true)
      };
      db.tasks.push(task);
      idMap.set(String(n.id), task.id);
    }

    let edgeCount = 0;
    let skipped = 0;
    for (const e of (body.edges || [])) {
      const from = idMap.get(String(e.s));
      const to = idMap.get(String(e.t));
      if (!from || !to) { skipped++; continue; }
      db.edges.push({
        id: uid(), from, fromPort: 'out', to, toPort: 'in',
        carryName: 'out', graphId: g.id,
        name: String(e.name || e.label || '接口'),
        label: String(e.label || e.name || '接口'),
        kind: KINDS.includes(e.kind) ? e.kind : 'code',
        proto: String(e.proto || 'http'),
        auth: String(e.auth || 'no'),
        spec: String(e.spec || ''),
        contract: ''
      });
      edgeCount++;
    }
    touch();
    return { graph: g, nodes: nodesIn.length, edges: edgeCount, skipped };
  }

  // ---------- 拓扑 ----------
  /** from → to 表示 to 依赖 from（调用方用 valid 集合做图内隔离） */
  function taskDeps(id) {
    return db.edges.filter((e) => e.to === id).map((e) => e.from);
  }

  /**
   * DFS 三色标记拓扑排序。
   * 有环时仍返回一个确定性顺序（并置 cycle=true），不抛错 —— 用户需要看到图而不是一个 500。
   */
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

  return {
    colorOf,
    createGraph, updateGraph, deleteGraphCascade, graphTaskIds,
    createTask, updateTask, deleteTask,
    createEdge, updateEdge, deleteEdge,
    importArchitecture,
    taskDeps, topoOrder
  };
}

module.exports = { createGraphService, KINDS };
