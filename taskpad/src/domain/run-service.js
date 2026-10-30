'use strict';

const path = require('path');
const { uid, nowIso, envKey } = require('../core/ids');
const { simpleGlobMatch } = require('../core/ports');
const { emitSafe } = require('../core/events');
const { notFound, conflict, badRequest } = require('../core/errors');
const { briefRun, briefTrace } = require('../core/views');

const ACTIVE = ['pending', 'running'];
const TERMINAL = ['success', 'error', 'killed', 'aborted', 'skipped'];

/**
 * 运行 + Worker 协调领域服务。
 *
 * 这块是整个系统里唯一有「并发语义」的地方，规则写死在这里：
 *   1. 一个模块同时只能有一个活动运行（pending/running），重复启动返回 skipped；
 *   2. Worker 认领即获得租约，心跳续租，超时由 sweep 收回队列（可能被别的节点接管）；
 *   3. 提交时做门禁：任何一条测试失败（in / out / iface）都判 error，成果不向下游投递；
 *   4. 状态流转只能按 pending → running → terminal 走，terminal 不可覆盖。
 */
function createRunService({ db, store, bus, cfg, runtimeInfo }) {
  const L = cfg.limits;

  function touch() {
    store.save();
  }

  function markState() {
    store.save();
    emitSafe(bus, 'state:dirty', {});
  }

  function emitRun(topic, run, extra) {
    emitSafe(bus, topic, Object.assign({ run: briefRun(run), runId: run.id }, extra || {}));
  }

  const findRun = (id) => db.runs.find((r) => r.id === id) || null;

  // ---------- Worker ----------
  function touchWorker(id) {
    const now = nowIso();
    let w = db.workers.find((x) => x.id === id);
    if (!w) {
      w = { id, firstSeen: now, lastSeen: now, currentRunId: null, _persistAt: 0 };
      db.workers.unshift(w);
      if (db.workers.length > L.workerCap) db.workers.length = L.workerCap;
    } else {
      w.lastSeen = now;
    }
    // 空闲轮询很频繁，最多每 20s 落一次盘
    if (Date.now() - (w._persistAt || 0) > L.workerPersistMs) {
      w._persistAt = Date.now();
      store.save();
    }
    return w;
  }

  // ---------- 契约下发 ----------
  /** 模块相关的接口关系（edge = 契约，不是 in/out 数据流） */
  function claimInterfaces(task) {
    const list = [];
    for (const e of db.edges) {
      if (e.from !== task.id && e.to !== task.id) continue;
      const other = db.tasks.find((t) => t.id === (e.from === task.id ? e.to : e.from));
      list.push({
        edgeId: e.id,
        name: e.name || e.label || '接口',
        kind: e.kind || 'code',
        proto: e.proto || '',
        auth: e.auth || '',
        spec: e.spec || '',
        contract: e.contract || '',
        tests: (e.tests || []).map((t) => ({ id: t.id, name: t.name, command: t.command })),
        direction: e.from === task.id ? 'provide' : 'use',
        otherTaskId: other ? other.id : null,
        otherTaskName: other ? other.name : '?'
      });
    }
    return list;
  }

  /**
   * 上游成果注入。
   * 与 claimInterfaces 的分工：interfaces 回答「我这个模块对外承诺什么」，
   * inputs 回答「上一步实际交给我什么」。两者缺一，Worker 就只能盲跑。
   */
  function claimInputs(task) {
    const list = [];
    for (const e of db.edges.filter((x) => x.to === task.id)) {
      const src = db.tasks.find((t) => t.id === e.from);
      if (!src) continue;

      const inPort = (task.inputs || []).find((i) => i.id === e.toPort);
      const outPort = (src.outputs || []).find((o) => o.id === e.fromPort);
      const key = envKey((inPort && inPort.name) || e.toPort, 'in');
      const base = {
        port: e.toPort,
        portName: inPort ? inPort.name : e.toPort,
        env: '__IN_' + key,
        spec: inPort ? inPort.spec || '' : '',
        tests: (inPort && inPort.tests) || [],
        contract: e.contract || '',
        edgeId: e.id,
        label: e.label || '',
        kind: e.kind || 'code',
        proto: e.proto || '',
        auth: e.auth || '',
        edgeSpec: e.spec || '',
        srcTaskId: src.id,
        srcTaskName: src.name,
        srcPortName: outPort ? outPort.name : e.fromPort,
        srcPortSpec: outPort ? outPort.spec || '' : ''
      };

      const lastRun = db.runs.find((r) => r.taskId === src.id && r.status === 'success');
      if (!lastRun) { list.push(Object.assign({}, base, { kind: 'missing', reason: '上游尚未成功运行' })); continue; }

      const res = (lastRun.portResults || []).find((pr) => pr.portId === e.fromPort);
      if (!res) { list.push(Object.assign({}, base, { kind: 'missing', reason: '上游未产出该端口成果' })); continue; }
      if (res.valid === false) {
        list.push(Object.assign({}, base, { kind: 'invalid', reason: '上游成果未通过产出接口测试', tests: res.tests || [] }));
        continue;
      }

      if (res.kind === 'text') {
        const txt = (lastRun.output || []).map((o) => o.d).join('').slice(0, L.inputTextLimit);
        list.push(Object.assign({}, base, { kind: 'text', value: txt, srcRunId: lastRun.id }));
      } else if (res.kind === 'file' && res.file) {
        list.push(Object.assign({}, base, {
          kind: 'file',
          srcRunId: lastRun.id,
          portId: res.portId,
          fileName: res.file.name,
          size: res.file.size,
          fileUrl: '/api/runs/' + lastRun.id + '/file?portId=' + encodeURIComponent(res.portId)
        }));
      }
    }
    return list;
  }

  // ---------- 启动 ----------
  function activeRunFor(taskId) {
    return db.runs.find((r) => r.taskId === taskId && ACTIVE.includes(r.status)) || null;
  }

  function startRun(task, trigger, triggerLabel) {
    // 没有命令的节点是「纯架构模块」，不是「跑一条空命令」
    if (!String(task.command || '').trim()) {
      return { blocked: true, run: null, reason: '该模块没有执行入口（纯架构节点，需要先在详情里接一条命令）' };
    }
    const active = activeRunFor(task.id);
    if (active) return { skipped: true, run: null, active };

    const run = {
      id: uid(),
      taskId: task.id,
      taskName: task.name,
      engine: task.engine,
      command: task.command,
      trigger,
      triggerLabel,
      status: 'pending',
      startedAt: nowIso(),
      finishedAt: null,
      exitCode: null,
      durationMs: null,
      worker: null,
      claimedAt: null,
      heartbeatAt: null,
      abortRequested: false,
      output: [],
      portResults: [],
      artifacts: {},
      traceId: null
    };
    db.runs.unshift(run);
    if (db.runs.length > L.runCap) db.runs.length = L.runCap;
    store.save();
    emitRun('run:start', run);
    emitSafe(bus, 'state:dirty', {});
    return { run, skipped: false };
  }

  function addSkippedRun(task, triggerLabel, traceId) {
    const run = {
      id: uid(), taskId: task.id, taskName: task.name, engine: task.engine, command: task.command,
      trigger: 'scheduled', triggerLabel,
      status: 'skipped', startedAt: nowIso(), finishedAt: nowIso(), exitCode: null, durationMs: 0,
      output: [{ t: Date.now(), s: 'err', d: '[跳过] 该任务已有运行正在进行，本次定时触发跳过。' }],
      portResults: [], artifacts: {}, worker: null, traceId: traceId || null
    };
    db.runs.unshift(run);
    if (db.runs.length > L.runCap) db.runs.length = L.runCap;
    store.save();
    emitRun('run:start', run);
    emitRun('run:end', run);
    return run;
  }

  // ---------- 认领 ----------
  function claim(workerId) {
    // FIFO：runs 是 newest-first，所以从尾部找最老的 pending，避免新任务饿死老任务
    let run = null;
    for (let i = db.runs.length - 1; i >= 0; i--) {
      if (db.runs[i].status === 'pending') { run = db.runs[i]; break; }
    }
    if (!run) return null;

    const task = db.tasks.find((t) => t.id === run.taskId);
    if (!task) return null;

    const engine = cfg.engines[task.engine] || cfg.engines[cfg.engineIds[0]] || { label: task.engine, cwd: cfg.projectRoot };
    run.status = 'running';
    run.worker = workerId;
    run.claimedAt = nowIso();
    run.heartbeatAt = run.claimedAt;

    const w = touchWorker(workerId);
    w.currentRunId = run.id;

    let engineCwd = '.';
    if (engine.cwd) {
      const rel = path.relative(cfg.projectRoot, engine.cwd).split(path.sep).join('/');
      engineCwd = !rel || rel === '' ? '.' : rel;
      if (engineCwd.startsWith('..')) engineCwd = '.';
    }

    store.save();
    emitRun('run:update', run);

    const graph = db.graphs.find((g) => g.id === task.graphId);
    const rt = runtimeInfo ? (runtimeInfo.runtimes || []).find((r) => r.id === task.engine) : null;
    return {
      runId: run.id,
      taskId: task.id,
      taskName: task.name,
      command: task.command,
      engine: task.engine,
      engineLabel: engine.label,
      engineCwd,
      timeoutMs: engine.timeoutMs || 1800000,
      graphId: task.graphId || null,
      graphName: graph ? graph.name : null,
      // 这个模块属于哪一层、这层的唯一入口和产物库在哪 —— 别再让 agent 自己猜
      layer: task.layer || (graph && graph.layer) || null,
      runtime: rt ? {
        id: rt.id, kind: rt.kind, entry: rt.entry, unit: rt.unit, db: rt.db, note: rt.note
      } : null,
      scope: task.scope || '',
      spec: task.spec || '',
      interfaces: claimInterfaces(task),
      inputs: claimInputs(task),
      outputs: (task.outputs || []).map((o) => ({
        id: o.id, name: o.name, capture: o.capture || 'stdout', glob: o.glob || '*',
        spec: o.spec || '', tests: o.tests || []
      }))
    };
  }

  // ---------- 心跳 ----------
  function heartbeat(runId, workerId) {
    const run = findRun(runId);
    if (!run) throw notFound('run not found');
    if (run.status !== 'running') throw conflict('run not running', { status: run.status });
    if (run.worker && workerId && run.worker !== workerId) throw conflict('claimed by another worker');
    run.heartbeatAt = nowIso();
    touchWorker(workerId || run.worker || 'unknown');
    store.save();
    return { ok: true, abort: !!run.abortRequested };
  }

  // ---------- 成果捕获 ----------
  function captureResults(run) {
    const task = db.tasks.find((t) => t.id === run.taskId);
    if (!task) return;
    const stdoutText = (run.output || []).map((o) => o.d).join('');
    const allTests = Array.isArray(run.tests) ? run.tests : [];
    const results = [];

    for (const out of task.outputs || []) {
      const cap = out.capture || 'stdout';
      // 只有 out 侧测试决定「这个端口的数据能不能用」；iface 失败在 run 层判 error
      const portTests = allTests.filter((t) => t.side === 'out' && t.portId === out.id);
      const valid = portTests.length ? portTests.every((t) => t.passed) : true;
      const tests = portTests.map((t) => ({
        id: t.testId, name: t.name, passed: !!t.passed, exitCode: t.exitCode,
        output: String(t.output || '').slice(0, 2000)
      }));

      if (cap === 'stdout') {
        results.push({
          portId: out.id, name: out.name, kind: 'text',
          preview: stdoutText.slice(0, 2000), chars: stdoutText.length,
          valid, tests, tested: portTests.length > 0
        });
      } else if (cap === 'file') {
        const names = Object.keys(run.artifacts || {}).filter((fn) => simpleGlobMatch(fn, out.glob || '*'));
        if (names.length) {
          const fn = names[0];
          const art = run.artifacts[fn];
          results.push({
            portId: out.id, name: out.name, kind: 'file',
            file: { name: fn, path: art.path, size: art.size },
            valid, tests, tested: portTests.length > 0
          });
        }
      }
    }
    run.portResults = results;
  }

  // ---------- 提交 ----------
  function submit(runId, workerId, body) {
    const run = findRun(runId);
    if (!run) throw notFound('run not found');
    if (TERMINAL.includes(run.status)) return { ok: true, already: true, run: briefRun(run) };
    if (run.worker && body.workerId && run.worker !== body.workerId) throw conflict('claimed by another worker');

    const stdout = String(body.stdout == null ? '' : body.stdout);
    const stderr = String(body.stderr == null ? '' : body.stderr);
    if (stdout) run.output.push({ t: Date.now(), s: 'out', d: stdout });
    if (stderr) run.output.push({ t: Date.now(), s: 'err', d: stderr });

    const aborted = !!body.aborted || !!run.abortRequested;
    const testResults = Array.isArray(body.tests) ? body.tests.map((t) => ({
      portId: String(t.portId || ''),
      edgeId: String(t.edgeId || ''),
      side: t.side === 'in' ? 'in' : t.side === 'iface' ? 'iface' : 'out',
      testId: String(t.testId || ''),
      name: String(t.name || ''),
      passed: !!t.passed,
      exitCode: typeof t.exitCode === 'number' ? t.exitCode : null,
      output: String(t.output || '').slice(0, 4000)
    })) : [];
    run.tests = testResults;

    const exitOk = body.exitCode === 0 || body.exitCode === '0';
    let status = aborted ? 'aborted' : (exitOk ? 'success' : 'error');

    // 门禁：in / out / iface 任一失败即 error。旧实现只看 out，iface 破了还当成功投给下游。
    const failures = testResults.filter((t) => !t.passed);
    if (status === 'success' && failures.length) status = 'error';
    run.gate = {
      ok: failures.length === 0,
      failures: failures.map((f) => ({ side: f.side, name: f.name, portId: f.portId, edgeId: f.edgeId }))
    };
    if (status === 'error' && failures.length) {
      for (const f of failures.slice(0, 6)) {
        const label = f.side === 'iface'
          ? '接口测试未通过：' + (f.name || f.edgeId || '')
          : (f.side === 'in' ? '输入接口测试未通过：' : '产出接口测试未通过：') + (f.name || f.portId || '');
        run.output.push({ t: Date.now(), s: 'err', d: '[' + label + ']\n' + String(f.output || '').slice(0, 1500) + '\n' });
      }
    }

    run.worker = run.worker || String(body.workerId || '');
    if (run.worker) touchWorker(run.worker);
    const w = db.workers.find((x) => x.id === run.worker);
    if (w) w.currentRunId = null;

    finalize(run, status, typeof body.exitCode === 'number' ? body.exitCode : null);
    return { ok: true, run: briefRun(run) };
  }

  function finalize(run, status, exitCode) {
    if (TERMINAL.includes(run.status)) return run;
    run.status = status;
    run.exitCode = exitCode;
    run.finishedAt = nowIso();
    run.durationMs = Date.now() - new Date(run.startedAt).getTime();
    captureResults(run);
    if (run.traceId) updateTrace(run.traceId);
    store.save();
    emitRun('run:end', run);
    emitSafe(bus, 'state:dirty', {});
    return run;
  }

  // ---------- 中止 ----------
  function abort(runId, reason) {
    const run = findRun(runId);
    if (!run) throw notFound('run not found');
    if (TERMINAL.includes(run.status)) return briefRun(run);

    if (run.status === 'pending') {
      run.output.push({ t: Date.now(), s: 'err', d: '[中止] ' + (reason || '手动中止') + '\n' });
      finalize(run, 'aborted', null);
      return briefRun(run);
    }
    // running：只能打标记，由 worker 心跳拿到 abort=true 后自己收尾
    run.abortRequested = true;
    store.save();
    emitRun('run:update', run);
    return briefRun(run);
  }

  // ---------- 失联回收 ----------
  function sweep(nowMs) {
    const now = nowMs || Date.now();
    let changed = false;
    for (const run of db.runs) {
      if (run.status !== 'running' || !run.heartbeatAt) continue;
      if (now - new Date(run.heartbeatAt).getTime() <= L.heartbeatTimeoutMs) continue;
      run.status = 'pending';
      run.worker = null;
      run.claimedAt = null;
      run.heartbeatAt = null;
      run.output.push({ t: Date.now(), s: 'err', d: '[失联] 节点心跳超时，运行重新入队\n' });
      changed = true;
      emitRun('run:update', run);
      emitRun('run:stale', run);
    }
    if (changed) markState();
    return changed;
  }

  // ---------- trace ----------
  function computeTraceStatus(trace) {
    let hasErr = false, hasSkip = false, hasKill = false, running = false;
    for (const rid of trace.runIds) {
      const r = db.runs.find((x) => x.id === rid);
      if (!r) continue;
      if (r.status === 'error') hasErr = true;
      else if (r.status === 'killed' || r.status === 'aborted') hasKill = true;
      else if (r.status === 'skipped') hasSkip = true;
      if (ACTIVE.includes(r.status) || r.status === 'queued') running = true;
    }
    if (running) return 'running';
    if (hasKill) return 'killed';
    if (hasErr) return 'error';
    if (hasSkip) return 'skipped';
    return 'success';
  }

  function updateTrace(traceId) {
    const t = db.traces.find((x) => x.id === traceId);
    if (!t) return null;
    let fin = null;
    for (const rid of t.runIds) {
      const r = db.runs.find((x) => x.id === rid);
      if (r && r.finishedAt && (!fin || r.finishedAt > fin)) fin = r.finishedAt;
    }
    t.finishedAt = fin;
    t.durationMs = (t.startedAt && fin) ? (new Date(fin).getTime() - new Date(t.startedAt).getTime()) : null;
    t.status = computeTraceStatus(t);
    // 任务/运行被删除后，trace 里的引用同步瘦身
    t.runIds = t.runIds.filter((rid) => db.runs.some((x) => x.id === rid));
    t.taskIds = t.taskIds.filter((tid) => db.tasks.some((x) => x.id === tid));
    return t;
  }

  function createTrace(kind, label, startedAt) {
    const t = {
      id: uid(), kind, label: label || kind,
      startedAt: startedAt || nowIso(), finishedAt: null,
      status: 'running', runIds: [], taskIds: [], durationMs: null, createdAt: nowIso()
    };
    db.traces.unshift(t);
    if (db.traces.length > L.traceCap) db.traces.length = L.traceCap;
    return t;
  }

  function ensureTrace(kind, label, run, traceId) {
    let t = traceId ? db.traces.find((x) => x.id === traceId) : null;
    if (!t) t = createTrace(kind, label, run ? run.startedAt : null);
    if (run) {
      if (!t.runIds.includes(run.id)) t.runIds.push(run.id);
      if (!t.taskIds.includes(run.taskId)) t.taskIds.push(run.taskId);
      run.traceId = t.id;
    }
    updateTrace(t.id);
    return t.id;
  }

  // ---------- 等待运行结束（事件驱动，替代 400ms 轮询） ----------
  function waitForRun(runId, timeoutMs) {
    const current = findRun(runId);
    if (current && TERMINAL.includes(current.status)) {
      return Promise.resolve({ run: current, state: 'end' });
    }
    return new Promise((resolve) => {
      let timer = null;
      const cleanup = () => {
        bus.removeListener('run:end', onEnd);
        bus.removeListener('run:stale', onStale);
        if (timer) clearTimeout(timer);
      };
      const onEnd = (payload) => {
        if (!payload || payload.runId !== runId) return;
        cleanup();
        resolve({ run: findRun(runId), state: 'end' });
      };
      const onStale = (payload) => {
        if (!payload || payload.runId !== runId) return;
        cleanup();
        resolve({ run: findRun(runId), state: 'stale' });
      };
      bus.on('run:end', onEnd);
      bus.on('run:stale', onStale);
      if (timeoutMs && timeoutMs > 0) {
        timer = setTimeout(() => { cleanup(); resolve({ run: findRun(runId), state: 'timeout' }); }, timeoutMs);
        if (timer.unref) timer.unref();
      }
    });
  }

  function clearRuns() {
    db.runs = [];
    markState();
  }

  function listTraces(limit) {
    return db.traces.slice(0, limit || 200).map(briefTrace);
  }

  function getTrace(id) {
    const t = db.traces.find((x) => x.id === id);
    if (!t) return null;
    const runs = t.runIds.map(findRun).filter(Boolean).map(briefRun);
    return Object.assign(briefTrace(t), { runs });
  }

  return {
    ACTIVE, TERMINAL,
    touchWorker,
    claimInterfaces, claimInputs,
    activeRunFor, startRun, addSkippedRun,
    claim, heartbeat, submit, abort, sweep,
    captureResults, finalize,
    ensureTrace, updateTrace, createTrace, listTraces, getTrace,
    waitForRun, clearRuns,
    findRun
  };
}

module.exports = { createRunService, ACTIVE_RUN_STATUS: ACTIVE, TERMINAL_RUN_STATUS: TERMINAL };
