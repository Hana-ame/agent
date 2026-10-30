'use strict';

const { labelOf } = require('./cron');

/**
 * 对外视图投影层。
 *
 * 存在的理由：store 里的实体是「全字段内部结构」（含 output 全文、artifacts 路径等），
 * 直接吐给前端会导致 state 包体随运行记录线性膨胀。这里统一裁剪为
 * 「前端/Agent 需要的最小字段集」，并且只有这一处定义，避免前后端字段漂移。
 */

function briefRun(r) {
  if (!r) return null;
  return {
    id: r.id,
    taskId: r.taskId,
    taskName: r.taskName,
    engine: r.engine,
    trigger: r.trigger,
    triggerLabel: r.triggerLabel,
    status: r.status,
    startedAt: r.startedAt,
    finishedAt: r.finishedAt,
    worker: r.worker || null,
    claimedAt: r.claimedAt || null,
    abortRequested: !!r.abortRequested,
    exitCode: r.exitCode,
    durationMs: r.durationMs,
    outputChars: r.output ? r.output.reduce((n, o) => n + o.d.length, 0) : 0,
    portResults: Array.isArray(r.portResults) ? r.portResults.map((pr) => ({
      portId: pr.portId,
      name: pr.name,
      kind: pr.kind,
      preview: pr.kind === 'text' ? String(pr.preview || '').slice(0, 200) : '',
      chars: pr.chars || 0,
      file: pr.file ? { name: pr.file.name, size: pr.file.size } : null,
      valid: pr.valid !== false,
      tested: !!pr.tested,
      tests: Array.isArray(pr.tests) ? pr.tests.map((t) => ({ name: t.name, passed: !!t.passed })) : []
    })) : [],
    tests: Array.isArray(r.tests)
      ? r.tests.map((t) => ({ portId: t.portId, edgeId: t.edgeId || '', side: t.side, name: t.name, passed: !!t.passed }))
      : [],
    gate: r.gate || null,
    traceId: r.traceId || null
  };
}

function briefTrace(t) {
  if (!t) return null;
  return {
    id: t.id,
    kind: t.kind,
    label: t.label,
    startedAt: t.startedAt,
    finishedAt: t.finishedAt,
    status: t.status,
    runCount: (t.runIds || []).length,
    taskCount: (t.taskIds || []).length,
    durationMs: t.durationMs
  };
}

function briefSchedule(s, taskName) {
  return {
    id: s.id,
    taskId: s.taskId,
    taskName: taskName || '?',
    type: s.type,
    minutes: s.minutes,
    time: s.time,
    cron: s.cron,
    enabled: s.enabled,
    nextAt: s.nextAt,
    label: labelOf(s)
  };
}

/**
 * 全量状态（SSE 首次推送 + GET /api/state + 变更后广播）
 * @param {object} db
 * @param {{engines:Array, queue:object}} surface
 */
function statePayload(db, surface) {
  const tasks = db.tasks.map((t) => {
    const schedules = db.schedules
      .filter((s) => s.taskId === t.id)
      .map((s) => ({
        id: s.id, type: s.type, minutes: s.minutes, time: s.time,
        cron: s.cron, enabled: s.enabled, nextAt: s.nextAt, label: labelOf(s)
      }));
    const lastRun = db.runs.find((r) => r.taskId === t.id);
    return Object.assign({}, t, { schedules, lastRun: briefRun(lastRun) });
  });

  return {
    graphs: db.graphs.map((g) => ({
      id: g.id, name: g.name, parentNodeId: g.parentNodeId || null, layer: g.layer || null, createdAt: g.createdAt
    })),
    tasks,
    edges: db.edges.map((e) => {
      const from = db.tasks.find((t) => t.id === e.from);
      const to = db.tasks.find((t) => t.id === e.to);
      const fromPort = from && (from.outputs || []).find((o) => o.id === e.fromPort);
      const toPort = to && (to.inputs || []).find((i) => i.id === e.toPort);
      return {
        id: e.id, from: e.from, fromPort: e.fromPort, to: e.to, toPort: e.toPort, carryName: e.carryName,
        contract: e.contract || '', name: e.name || e.label || '', label: e.label || e.name || '',
        kind: e.kind || 'code', proto: e.proto || '', auth: e.auth || '', spec: e.spec || '',
        tests: (e.tests || []).map((t) => ({ id: t.id, name: t.name, command: t.command })),
        graphId: e.graphId || null,
        fromName: from ? from.name : '?',
        toName: to ? to.name : '?',
        fromPortName: (fromPort || {}).name || e.fromPort,
        toPortName: (toPort || {}).name || e.toPort
      };
    }),
    workers: db.workers.map((w) => ({
      id: w.id, firstSeen: w.firstSeen, lastSeen: w.lastSeen, currentRunId: w.currentRunId
    })),
    queue: surface.queue || { running: false, remaining: 0, currentTaskId: null },
    traces: db.traces.slice(0, 60).map(briefTrace),
    schedules: db.schedules.map((s) => briefSchedule(s, (db.tasks.find((t) => t.id === s.taskId) || {}).name)),
    runs: db.runs.slice(0, 100).map(briefRun),
    engines: surface.engines || [],
    now: new Date().toISOString()
  };
}

module.exports = { briefRun, briefTrace, briefSchedule, statePayload };
