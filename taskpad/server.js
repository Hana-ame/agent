'use strict';

/**
 * TaskPad — 任务白板（零依赖 Node.js）
 *
 * 这个文件只做「装配」：读配置 → 建 store → 建事件总线 → 建领域服务 → 建 HTTP 路由 → 监听。
 * 业务逻辑一律不写在这里。分层与依赖方向见 README「服务端结构」一节：
 *
 *   http/routes/*  →  domain/*  →  core/*
 *
 * 反向依赖是不允许的：core 不知道 domain 存在，domain 不知道 HTTP 存在。
 */

const http = require('http');

const { loadConfig } = require('./src/core/config');
const { createStore } = require('./src/core/store');
const { createBus } = require('./src/core/events');
const { statePayload } = require('./src/core/views');

const { createGraphService } = require('./src/domain/graph-service');
const { createRunService } = require('./src/domain/run-service');
const { createScheduler } = require('./src/domain/scheduler');

const { createRouter } = require('./src/http/router');
const { createSseHub } = require('./src/http/sse');
const { createStatic } = require('./src/http/static');
const { registerMeta } = require('./src/http/routes/meta');
const { registerGraphs } = require('./src/http/routes/graphs');
const { registerRuns } = require('./src/http/routes/runs');
const { registerSchedules } = require('./src/http/routes/schedules');

function createApp(rootDir) {
  const cfg = loadConfig(rootDir);

  const store = createStore(cfg);
  const db = store.init();

  const bus = createBus();
  const graphService = createGraphService({ db, store, bus, cfg });
  const runService = createRunService({ db, store, bus, cfg });
  const scheduler = createScheduler({ db, store, bus, cfg, runService, graphService });

  const ctx = {
    cfg, db, store, bus,
    graphService, runService, scheduler,
    statePayload: () => statePayload(db, { engines: cfg.enginesView, queue: scheduler.queueSnapshot() })
  };

  const sse = createSseHub({
    bus, cfg,
    buildState: ctx.statePayload,
    queueSnapshot: scheduler.queueSnapshot
  });
  ctx.sse = sse;

  const router = createRouter();
  registerMeta(router, ctx);
  registerGraphs(router, ctx);
  registerRuns(router, ctx);
  registerSchedules(router, ctx);
  ctx.router = router;

  const staticFiles = createStatic({ cfg });

  const server = http.createServer((req, res) => {
    router.handle(req, res, ctx)
      .then((matched) => {
        if (matched) return;
        const pathname = new URL(req.url, 'http://localhost').pathname.replace(/\/+$/, '') || '/';
        staticFiles.serve(req, res, pathname);
      })
      .catch((err) => {
        try { res.writeHead(500); res.end(String(err && err.message)); } catch (e) { /* ignore */ }
      });
  });

  function shutdown(reason) {
    scheduler.stop();
    sse.closeAll();
    store.flush().finally(() => {
      server.close(() => process.exit(0));
      setTimeout(() => process.exit(0), 1500).unref();
    });
    if (reason) console.log('[taskpad] 关闭：' + reason);
  }

  return { cfg, db, store, bus, ctx, server, scheduler, shutdown };
}

if (require.main === module) {
  const app = createApp(__dirname);
  const { cfg } = app;
  app.scheduler.start();

  app.server.listen(cfg.port, () => {
    const addr = app.server.address();
    console.log('[taskpad] listening on http://127.0.0.1:' + addr.port);
    console.log('  UI:     http://127.0.0.1:' + addr.port);
    console.log('  数据:   ' + cfg.storeFile);
    console.log('  引擎:   ' + cfg.engineIds.map((k) => k + '→' + cfg.engines[k].cwd).join(' | '));
  });

  process.on('SIGINT', () => app.shutdown('SIGINT'));
  process.on('SIGTERM', () => app.shutdown('SIGTERM'));
}

module.exports = { createApp };
