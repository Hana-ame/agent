'use strict';

const { json, readBody } = require('../router');
const { notFound } = require('../../core/errors');
const { uid, nowIso } = require('../../core/ids');
const { labelOf } = require('../../core/cron');
const { emitSafe } = require('../../core/events');

/** 定时任务 + 整图运行 */
function registerSchedules(router, ctx) {
  const { db, store, bus, scheduler } = ctx;

  function touch() {
    store.save();
    emitSafe(bus, 'state:dirty', {});
  }

  function applyPatch(s, body) {
    if (['interval', 'daily', 'cron'].includes(body.type)) s.type = body.type;
    if (typeof body.enabled === 'boolean') s.enabled = body.enabled;
    if (body.minutes) s.minutes = parseInt(body.minutes, 10) || 5;
    if (body.time) s.time = String(body.time);
    if (body.cron) s.cron = String(body.cron);
    scheduler.refreshNextAt(s, Date.now());
    return s;
  }

  router.post('/api/schedules', async ({ req, res }) => {
    const body = await readBody(req);
    const task = store.findTask(body.taskId);
    if (!task) return json(res, 400, { error: 'task not found' });

    const s = applyPatch({
      id: uid(),
      taskId: task.id,
      type: 'interval',
      enabled: body.enabled !== false,
      minutes: 5,
      time: '08:00',
      cron: '0 * * * *',
      nextAt: null,
      createdAt: nowIso()
    }, body);
    db.schedules.push(s);
    touch();
    return json(res, 200, { schedule: Object.assign({ label: labelOf(s) }, s) });
  });

  async function updateHandler({ req, res, params }) {
    const s = store.findSchedule(params.id);
    if (!s) throw notFound('schedule not found');
    applyPatch(s, await readBody(req));
    touch();
    return json(res, 200, { schedule: Object.assign({ label: labelOf(s) }, s) });
  }

  router.put('/api/schedules/:id', updateHandler);
  router.post('/api/schedules/:id', updateHandler);

  router.delete('/api/schedules/:id', ({ res, params }) => {
    const i = db.schedules.findIndex((x) => x.id === params.id);
    if (i < 0) throw notFound('schedule not found');
    db.schedules.splice(i, 1);
    touch();
    json(res, 200, { ok: true });
  });

  // 整图运行 / 停止
  router.post('/api/graph/run', async ({ req, res }) => {
    const body = await readBody(req).catch(() => ({}));
    json(res, 200, scheduler.runGraph(body));
  });

  router.post('/api/graph/stop', ({ res }) => json(res, 200, scheduler.stopGraph()));
}

module.exports = { registerSchedules };
