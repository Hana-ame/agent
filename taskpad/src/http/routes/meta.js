'use strict';

const { json, text } = require('../router');
const { TEMPLATES } = require('../../core/seed');
const { labelOf } = require('../../core/cron');

/** 元信息 / 状态 / Agent 发现入口 / SSE */
function registerMeta(router, ctx) {
  const { cfg, db, sse, scheduler } = ctx;

  router.get('/api/health', ({ res }) => {
    json(res, 200, {
      ok: true,
      time: new Date().toISOString(),
      uptimeSec: Math.round(process.uptime()),
      counts: {
        graphs: db.graphs.length,
        tasks: db.tasks.length,
        edges: db.edges.length,
        runs: db.runs.length,
        traces: db.traces.length,
        schedules: db.schedules.length,
        sseClients: sse.clientCount()
      }
    });
  });

  router.get('/api/state', ({ res }) => json(res, 200, ctx.statePayload()));

  router.get('/api/templates', ({ res }) => json(res, 200, { templates: TEMPLATES }));

  router.get('/api/events', ({ req, res }) => sse.attach(req, res));

  router.get('/api/agent-guide', ({ req, res }) => {
    const host = req.headers.host || ('127.0.0.1:' + cfg.port);
    const base = 'http://' + host;
    json(res, 200, {
      server: base,
      protocol: 'taskpad-pull-v1',
      summary: '项目 → 模块（节点）→ 接口（边）→ 流程图（节点内部子图）。agent 用 HTTP/curl 认领任务、执行、上传结果。',
      steps: [
        '1) curl ' + base + '/api/state    # 读取图/模块/接口定义（nodes、edges、inputs/outputs、spec、tests）',
        '2) curl -X POST ' + base + '/api/runs/claim -H "Content-Type: application/json" -d \'{"workerId":"<你的节点名>"}\'   # 认领（无任务返回 {"run":null}）',
        '3) 按 run.command 本地执行；run.inputs 里 kind=text 的项按 env 字段注入（__IN_<输入端口名>），kind=file 的下载 fileUrl 后用 __IN_<名>_FILE 指路',
        '4) 先跑 run.inputs[].tests（输入门禁），再跑 run.interfaces[].tests（接口契约）与 run.outputs[].tests（产出契约）',
        '5) 执行中每 5s：curl -X POST ' + base + '/api/runs/<runId>/heartbeat -d \'{"workerId":"<你的节点名>"}\'；响应 abort=true 时终止',
        '6) 文件成果：curl -X POST ' + base + '/api/runs/<runId>/files/<文件名> --data-binary @<本地文件>',
        '7) 提交：curl -X POST ' + base + '/api/runs/<runId>/submit -H "Content-Type: application/json" -d \'{"workerId":"<你的节点名>","exitCode":0,"stdout":"...","stderr":"","tests":[{"portId":"out","side":"out","testId":"t1","name":"接口测试","passed":true,"exitCode":0,"output":""}]}\''
      ],
      endpoints: {
        state: 'GET /api/state',
        claim: 'POST /api/runs/claim',
        heartbeat: 'POST /api/runs/:id/heartbeat',
        upload: 'POST /api/runs/:id/files/:name',
        submit: 'POST /api/runs/:id/submit',
        discovery: 'GET /api/agent-guide',
        human_guide: 'GET /agent.md'
      },
      worker_script: 'taskpad/worker.py（纯 stdlib，可直接 --server ' + base + ' 运行）',
      gate: 'side=in/out/iface 任一测试未通过 → 本次运行判 error，成果不投递给下游。'
    });
  });

  router.get('/agent.md', ({ req, res }) => {
    const host = req.headers.host || ('127.0.0.1:' + cfg.port);
    const lines = [
      '# TaskPad Agent 接入指南（HTTP / curl）',
      '',
      'Server: http://' + host,
      '',
      '## 模型',
      '项目(Project) → 模块(Node) → 接口(Edge, code/data/net + proto/auth/spec) → 流程图(Node 内部子图)。',
      '',
      '## 最小接入（curl）',
      '```bash',
      '# 1. 看状态',
      'curl -s http://' + host + '/api/state',
      '# 2. 认领（返回 scope/spec/interfaces/inputs/outputs）',
      'curl -s -X POST http://' + host + '/api/runs/claim -H "Content-Type: application/json" -d \'{"workerId":"my-agent"}\'',
      '# 3. 执行：env 里带 __IN_* / __IN_*_FILE；先跑 inputs[].tests 再跑主命令',
      '# 4. 心跳',
      'curl -s -X POST http://' + host + '/api/runs/<runId>/heartbeat -H "Content-Type: application/json" -d \'{"workerId":"my-agent"}\'',
      '# 5. 提交',
      'curl -s -X POST http://' + host + '/api/runs/<runId>/submit -H "Content-Type: application/json" -d \'{"workerId":"my-agent","exitCode":0,"stdout":"ok","tests":[]}\'',
      '```',
      '',
      '## 或直接跑内置 worker',
      '```bash',
      'python3 worker.py --server http://' + host + ' --worker-id my-agent --root /path/to/project',
      '```',
      '',
      '## 定时任务接口',
      '- GET  /api/schedules          列出全部定时（含 label / nextAt）',
      '- POST /api/schedules          {taskId,type:interval|daily|cron,minutes,time,cron,enabled}',
      '- POST /api/graph/run          {graphId} 或 {taskIds}，按拓扑序逐个执行'
    ];
    text(res, 200, lines.join('\n'), 'text/markdown; charset=utf-8');
  });
}

module.exports = { registerMeta, labelOf };
