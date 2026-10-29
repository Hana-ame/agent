'use strict';

const fs = require('fs');
const path = require('path');
const { json, readBody, readRaw } = require('../router');
const { isInside } = require('../static');
const { notFound, forbidden } = require('../../core/errors');
const { briefRun } = require('../../core/views');

/** 运行 / Worker / Trace 的 HTTP 入口 */
function registerRuns(router, ctx) {
  const { cfg, db, store, runService } = ctx;

  router.get('/api/runs', ({ res }) => {
    json(res, 200, { runs: db.runs.slice(0, 200).map(briefRun) });
  });

  router.get('/api/runs/:id', ({ res, params }) => {
    const run = store.findRun(params.id);
    if (!run) throw notFound('run not found');
    json(res, 200, { run });
  });

  // 成果文件下载：只允许 artifact 目录或该引擎工作目录内的文件
  router.get('/api/runs/:id/file', ({ res, params, query }) => {
    const run = store.findRun(params.id);
    if (!run) throw notFound('run not found');
    const portId = query.get('portId');
    const task = store.findTask(run.taskId);
    const engine = cfg.engines[task ? task.engine : ''] || cfg.engines[cfg.engineIds[0]];
    const result = (run.portResults || []).find((pr) => pr.portId === portId && pr.kind === 'file' && pr.file);
    if (!result) throw notFound('result not found');

    const fp = path.resolve(result.file.path);
    const allowed = isInside(cfg.artifactDir, fp) || (engine && engine.cwd && isInside(engine.cwd, fp));
    if (!allowed) throw forbidden('forbidden');

    fs.readFile(fp, (err, data) => {
      if (err) return json(res, 404, { error: 'file missing' });
      res.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': 'attachment; filename="' + encodeURIComponent(result.file.name) + '"'
      });
      res.end(data);
    });
    return undefined;
  });

  router.post('/api/runs/clear', ({ res }) => {
    runService.clearRuns();
    json(res, 200, { ok: true });
  });

  router.post('/api/runs/claim', async ({ req, res }) => {
    const body = await readBody(req);
    const workerId = String(body.workerId || '').trim();
    if (!workerId) return json(res, 400, { error: 'workerId required' });
    runService.touchWorker(workerId);
    const claimed = runService.claim(workerId);
    if (!claimed) return json(res, 200, { run: null });
    return json(res, 200, { run: claimed });
  });

  router.post('/api/runs/:id/heartbeat', async ({ req, res, params }) => {
    const body = await readBody(req);
    const out = runService.heartbeat(params.id, String(body.workerId || '').trim());
    json(res, 200, out);
  });

  router.post('/api/runs/:id/submit', async ({ req, res, params }) => {
    const body = await readBody(req);
    json(res, 200, runService.submit(params.id, String(body.workerId || '').trim(), body));
  });

  router.post('/api/runs/:id/files/:name', async ({ req, res, params }) => {
    const run = store.findRun(params.id);
    if (!run) throw notFound('run not found');

    let name = String(params.name || '').replace(/[\\/:*?"<>|]/g, '_').slice(0, 120);
    if (!name || name === '.' || name === '..') return json(res, 400, { error: 'bad file name' });

    const data = await readRaw(req, cfg.limits.fileBodyLimit);
    const dir = path.join(cfg.artifactDir, run.id);
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, name);
    if (!isInside(dir, fp)) return json(res, 400, { error: 'bad file name' });
    fs.writeFileSync(fp, data);

    run.artifacts = run.artifacts || {};
    run.artifacts[name] = { path: fp, size: data.length, at: new Date().toISOString() };
    store.save();
    return json(res, 200, { ok: true, name, size: data.length });
  });

  router.post('/api/runs/:id/kill', ({ res, params }) => {
    const run = store.findRun(params.id);
    if (!run) throw notFound('run not found');
    json(res, 200, { ok: true, runId: params.id, run: runService.abort(params.id, '手动中止') });
  });

  // ---------- Trace ----------
  router.get('/api/traces', ({ res }) => {
    json(res, 200, { traces: runService.listTraces(200) });
  });

  router.get('/api/traces/:id', ({ res, params }) => {
    const t = runService.getTrace(params.id);
    if (!t) throw notFound('trace not found');
    json(res, 200, { trace: t });
  });
}

module.exports = { registerRuns };
