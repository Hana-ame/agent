'use strict';

const { emitSafe } = require('../core/events');
const { uid } = require('../core/ids');
const { advanceSchedule, computeNextAt, labelOf } = require('../core/cron');
const { badRequest } = require('../core/errors');

/**
 * 调度器：两件事，别的一概不管。
 *   1. 时间维度：interval / daily / cron → 到点起一次运行；
 *   2. 拓扑维度：整图运行 → 按拓扑序逐个起运行，等上一个到达终态再推进。
 *
 * 和旧实现的差别：图队列不再用 setInterval(400ms) 轮询数据库，
 * 而是由 run-service 在运行终态时发 run:end 事件唤醒。轮询 → 事件，延迟和空转都没了。
 */
function createScheduler({ db, store, bus, cfg, runService, graphService }) {
  const queue = { running: false, remaining: [], currentTaskId: null, traceId: null };
  let tickTimer = null;
  let sweepTimer = null;

  function queueDirty() {
    emitSafe(bus, 'queue:dirty', {});
    emitSafe(bus, 'state:dirty', {});
  }

  function queueSnapshot() {
    return { running: queue.running, remaining: queue.remaining.length, currentTaskId: queue.currentTaskId };
  }

  // ---------- 定时 ----------
  function fireSchedule(s) {
    const task = db.tasks.find((t) => t.id === s.taskId);
    if (!task) return;
    const label = labelOf(s);
    const res = runService.startRun(task, 'scheduled', label);
    if (res.blocked) return;                       // 纯架构模块，定时到了也没东西可跑
    if (res.skipped) {
      const skipped = runService.addSkippedRun(task, label);
      runService.ensureTrace('scheduled', label, skipped);
      store.save();
      return;
    }
    runService.ensureTrace('scheduled', label, res.run);
    store.save();
  }

  function tick(nowMs) {
    const now = nowMs || Date.now();
    let changed = false;
    for (const s of db.schedules) {
      if (!s.enabled || typeof s.nextAt !== 'number') continue;
      if (now < s.nextAt) continue;
      const missed = now - s.nextAt > cfg.limits.missGraceMs;
      if (!missed) fireSchedule(s);
      advanceSchedule(s, now);
      changed = true;
    }
    if (changed) store.save();
  }

  /** 新建/修改定时任务时统一算下次触发时刻 */
  function refreshNextAt(s, now) {
    s.nextAt = computeNextAt(s, now || Date.now());
    return s.nextAt;
  }

  // ---------- 整图运行 ----------
  function resolveTaskIds(body) {
    if (body && Array.isArray(body.taskIds) && body.taskIds.length
      && body.taskIds.every((id) => db.tasks.some((t) => t.id === id))) {
      return body.taskIds.slice(0, cfg.limits.graphTailCap);
    }
    if (body && body.graphId && db.graphs.some((g) => g.id === body.graphId)) {
      return graphService.graphTaskIds(body.graphId).slice(0, cfg.limits.graphTailCap);
    }
    return db.tasks.map((t) => t.id).slice(0, cfg.limits.graphTailCap);
  }

  function runGraph(body) {
    const ids = resolveTaskIds(body);
    if (!ids.length) throw badRequest('没有可运行的模块');
    const { order, cycle } = graphService.topoOrder(ids);
    queue.remaining.push(...order);
    kick();
    return { ok: true, order: order.length, cycle, queued: queue.remaining.length };
  }

  function stopGraph() {
    queue.remaining = [];
    if (queue.currentTaskId) {
      const cur = db.runs.find((r) => r.taskId === queue.currentTaskId
        && (r.status === 'running' || r.status === 'pending'));
      if (cur) runService.abort(cur.id, '队列已停止');
    }
    queueDirty();
    return { ok: true };
  }

  async function kick() {
    if (queue.running) return;
    queue.running = true;
    const trace = runService.createTrace('graph', '运行图');
    queue.traceId = trace.id;
    store.save();
    queueDirty();

    try {
      while (queue.remaining.length) {
        const taskId = queue.remaining.shift();
        queue.currentTaskId = taskId;
        queueDirty();

        const task = db.tasks.find((t) => t.id === taskId);
        if (!task) continue;

        const res = runService.startRun(task, 'graph', '运行图');
        if (res.blocked) {
          console.warn('[scheduler] 跳过未接执行入口的模块：' + task.name);
          continue;
        }
        if (res.skipped) {
          // 已有活动运行：记一条 skipped 并归入同一 trace，然后推进下一个
          const skipped = runService.addSkippedRun(task, '运行图');
          runService.ensureTrace('graph', '运行图', skipped, trace.id);
          store.save();
          continue;
        }

        const run = res.run;
        runService.ensureTrace('graph', '运行图', run, trace.id);
        store.save();

        const engine = cfg.engines[task.engine];
        const budget = (engine && engine.timeoutMs ? engine.timeoutMs : 1800000) + 120000;
        const waited = await runService.waitForRun(run.id, budget);
        if (waited.state === 'stale') {
          console.warn('[scheduler] 运行 ' + run.id + ' 心跳超时已回收，跳过该节点继续');
        } else if (waited.state === 'timeout') {
          console.warn('[scheduler] 运行 ' + run.id + ' 等待超时，跳过该节点继续');
          runService.abort(run.id, '图队列等待超时');
        }
      }
    } finally {
      queue.currentTaskId = null;
      queue.running = false;
      queue.traceId = null;
      runService.updateTrace(trace.id);
      store.save();
      queueDirty();
    }
  }

  // ---------- 生命周期 ----------
  function start() {
    if (!tickTimer) {
      tickTimer = setInterval(() => { try { tick(); } catch (e) { console.error('[scheduler] tick failed:', e.message); } }, cfg.limits.tickIntervalMs);
      if (tickTimer.unref) tickTimer.unref();
    }
    if (!sweepTimer) {
      sweepTimer = setInterval(() => {
        try { runService.sweep(); } catch (e) { console.error('[scheduler] sweep failed:', e.message); }
      }, cfg.limits.sweepIntervalMs);
      if (sweepTimer.unref) sweepTimer.unref();
    }
  }

  function stop() {
    if (tickTimer) { clearInterval(tickTimer); tickTimer = null; }
    if (sweepTimer) { clearInterval(sweepTimer); sweepTimer = null; }
  }

  return {
    queue, queueSnapshot, queueDirty,
    fireSchedule, tick, refreshNextAt,
    runGraph, stopGraph, kick,
    start, stop, uid
  };
}

module.exports = { createScheduler };
