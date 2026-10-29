'use strict';

const { json, readBody } = require('../router');
const { notFound } = require('../../core/errors');

/** 图 / 模块 / 接口 的 HTTP 入口。只做三件事：读参数 → 调领域服务 → 返回 JSON。 */
function registerGraphs(router, ctx) {
  const { graphService, runService, store } = ctx;

  // ---------- 图 ----------
  router.post('/api/graphs', async ({ req, res }) => {
    const body = await readBody(req);
    json(res, 200, { graph: graphService.createGraph(body) });
  });

  router.put('/api/graphs/:id', async ({ req, res, params }) => {
    json(res, 200, { graph: graphService.updateGraph(params.id, await readBody(req)) });
  });

  router.post('/api/graphs/:id', async ({ req, res, params }) => {
    json(res, 200, { graph: graphService.updateGraph(params.id, await readBody(req)) });
  });

  router.delete('/api/graphs/:id', ({ res, params }) => {
    json(res, 200, graphService.deleteGraphCascade(params.id));
  });

  // ---------- 模块 ----------
  router.post('/api/tasks', async ({ req, res }) => {
    json(res, 200, { task: graphService.createTask(await readBody(req)) });
  });

  router.put('/api/tasks/:id', async ({ req, res, params }) => {
    json(res, 200, { task: graphService.updateTask(params.id, await readBody(req)) });
  });

  router.post('/api/tasks/:id', async ({ req, res, params }) => {
    json(res, 200, { task: graphService.updateTask(params.id, await readBody(req)) });
  });

  router.delete('/api/tasks/:id', ({ res, params }) => {
    json(res, 200, graphService.deleteTask(params.id));
  });

  router.post('/api/tasks/:id/run', ({ res, params }) => {
    const task = store.findTask(params.id);
    if (!task) throw notFound('task not found');
    const r = runService.startRun(task, 'manual', '手动');
    if (r.skipped) return json(res, 409, { error: '该任务已在运行中' });
    runService.ensureTrace('manual', '手动', r.run);
    store.save();
    return json(res, 200, { runId: r.run.id });
  });

  // ---------- 接口 ----------
  router.post('/api/edges', async ({ req, res }) => {
    json(res, 200, { edge: graphService.createEdge(await readBody(req)) });
  });

  router.put('/api/edges/:id', async ({ req, res, params }) => {
    json(res, 200, { edge: graphService.updateEdge(params.id, await readBody(req)) });
  });

  router.post('/api/edges/:id', async ({ req, res, params }) => {
    json(res, 200, { edge: graphService.updateEdge(params.id, await readBody(req)) });
  });

  router.delete('/api/edges/:id', ({ res, params }) => {
    json(res, 200, graphService.deleteEdge(params.id));
  });

  // ---------- 架构图导入 ----------
  router.post('/api/import/architecture', async ({ req, res }) => {
    json(res, 200, graphService.importArchitecture(await readBody(req)));
  });
}

module.exports = { registerGraphs };
