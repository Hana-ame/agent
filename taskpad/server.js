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

/**
 * 命令行参数（优先级高于 config.json / 环境变量）：
 *   --port 8335          监听端口
 *   --data-dir <dir>     数据目录（默认 taskpad/data）
 *   --host 0.0.0.0       绑定地址（默认监听全部网卡）
 *   --log <file>         把 stdout/stderr 追加写入文件（后台启动时用）
 */
function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--port' && argv[i + 1]) out.port = argv[++i];
    else if (a === '--data-dir' && argv[i + 1]) out.dataDir = argv[++i];
    else if (a === '--host' && argv[i + 1]) out.host = argv[++i];
    else if (a === '--log' && argv[i + 1]) out.log = argv[++i];
  }
  return out;
}

/** 后台运行时没有控制台，把 console 同时写进日志文件 */
function teeTo(file) {
  const fs = require('fs');
  const util = require('util');
  const stream = fs.createWriteStream(file, { flags: 'a' });
  const wrap = (fn) => function () {
    fn.apply(console, arguments);
    try {
      stream.write(Array.prototype.map.call(arguments, (x) => (typeof x === 'string' ? x : util.inspect(x))).join(' ') + '\n');
    } catch (e) { /* 日志写不进不影响服务 */ }
  };
  console.log = wrap(console.log);
  console.error = wrap(console.error);
  console.warn = wrap(console.warn);
}

if (require.main === module) {
  const args = parseArgs(process.argv.slice(2));
  if (args.port) process.env.TASKPAD_PORT = String(args.port);
  if (args.dataDir) process.env.TASKPAD_DATA_DIR = args.dataDir;
  if (args.log) teeTo(args.log);

  const app = createApp(__dirname);
  const { cfg } = app;
  app.scheduler.start();

  app.server.listen(cfg.port, args.host, () => {
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
