#!/usr/bin/env node
'use strict';

/**
 * TaskPad 服务端冒烟测试（零依赖，Node 18+）。
 *
 *   node test/smoke.mjs
 *
 * 覆盖的是「架构层面容易回归」的几条线，而不是所有 CRUD：
 *   1. 冷启动迁移 + 种子数据
 *   2. 模块/接口创建与引用完整性
 *   3. Worker 认领 → 接口测试失败 → 运行判 error（门禁不能漏）
 *   4. 成功运行 → 上游成果注入下游 claim（inputs 不能断）
 *   5. 图运行按拓扑序排队
 *   6. 删模块级联清边
 *   7. 未知路由 / 静态目录穿越
 */

const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const assert = require('assert');

const PORT = Number(process.env.SMOKE_PORT || 8411);
const BASE = 'http://127.0.0.1:' + PORT;
const ROOT = path.resolve(__dirname, '..');
const DATA_DIR = path.join(os.tmpdir(), 'taskpad-smoke-' + Date.now());

let pass = 0;
const fails = [];

async function check(name, fn) {
  try {
    await fn();
    pass++;
    console.log('  ok   ' + name);
  } catch (e) {
    fails.push({ name, error: e });
    console.log('  FAIL ' + name + ' :: ' + (e && e.message));
  }
}

async function req(method, p, body) {
  const opts = { method };
  if (body !== undefined) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(BASE + p, opts);
  const t = await r.text();
  let json = null;
  try { json = JSON.parse(t); } catch (e) { /* 非 JSON */ }
  return { status: r.status, body: json, text: t };
}

function waitForListen(child) {
  return new Promise((resolve, reject) => {
    let buf = '';
    const timer = setTimeout(() => reject(new Error('启动超时')),
      10000);
    child.stdout.on('data', (d) => {
      buf += d.toString();
      if (buf.includes('listening on')) { clearTimeout(timer); resolve(); }
    });
    child.stderr.on('data', (d) => process.stderr.write('[server] ' + d.toString()));
    child.on('exit', (code) => reject(new Error('server 提前退出 code=' + code)));
  });
}

function startServer() {
  fs.rmSync(DATA_DIR, { recursive: true, force: true });
  const child = spawn(process.execPath, [path.join(ROOT, 'server.js')], {
    cwd: ROOT,
    env: Object.assign({}, process.env, { TASKPAD_PORT: String(PORT), TASKPAD_DATA_DIR: DATA_DIR }),
    stdio: ['ignore', 'pipe', 'pipe']
  });
  return child;
}

async function main() {
  const child = startServer();
  const stop = () => { try { child.kill(); } catch (e) { /* ignore */ } };

  try {
    await waitForListen(child);
    console.log('\n# 1. 冷启动');
    await check('state 返回 6 个种子模块 + 1 条示例接口', async () => {
      const r = await req('GET', '/api/state');
      assert.strictEqual(r.status, 200);
      assert.strictEqual(r.body.tasks.length, 6);
      assert.strictEqual(r.body.edges.length, 1);
      assert.strictEqual(r.body.graphs.length, 1);
    });

    await check('健康检查带计数', async () => {
      const r = await req('GET', '/api/health');
      assert.strictEqual(r.body.ok, true);
      assert.strictEqual(r.body.counts.tasks, 6);
    });

    console.log('\n# 2. 建模块 + 建接口');
    const up = (await req('POST', '/api/tasks', {
      name: '上游产出', command: 'echo UPSTREAM_OK', engine: 'v8'
    })).body.task;
    const down = (await req('POST', '/api/tasks', {
      name: '下游消费', command: 'echo DOWN', engine: 'v8'
    })).body.task;
    assert.ok(up.id && down.id);

    const edge = (await req('POST', '/api/edges', {
      from: up.id, to: down.id, name: '成果接口', kind: 'data',
      proto: 'file', auth: 'no', spec: 'JSON',
      tests: [{ name: '接口存活', command: 'true' }]
    })).body.edge;
    await check('接口带契约测试', () => {
      assert.strictEqual(edge.tests.length, 1);
      assert.strictEqual(edge.kind, 'data');
    });

    console.log('\n# 3. 门禁：接口测试失败必须判 error');
    await check('iface 测试失败 → status=error，且不向下游投递', async () => {
      const run = (await req('POST', '/api/tasks/' + up.id + '/run')).body;
      assert.ok(run.runId);

      const claim = (await req('POST', '/api/runs/claim', { workerId: 'smoke-1' })).body.run;
      assert.ok(claim, '应当认领到运行');
      assert.strictEqual(claim.command, 'echo UPSTREAM_OK');
      assert.strictEqual(claim.interfaces.length, 1);
      assert.strictEqual(claim.interfaces[0].direction, 'provide');

      const sub = (await req('POST', '/api/runs/' + claim.runId + '/submit', {
        workerId: 'smoke-1', exitCode: 0, stdout: 'UPSTREAM_OK',
        tests: [{ edgeId: claim.interfaces[0].edgeId, side: 'iface', name: '接口存活', passed: false, exitCode: 7, output: 'boom' }]
      })).body;
      assert.strictEqual(sub.run.status, 'error');
      assert.strictEqual(sub.run.gate.ok, false);

      const detail = (await req('GET', '/api/runs/' + claim.runId)).body.run;
      assert.ok(detail.output.some((o) => o.d.includes('接口测试未通过')));
    });

    console.log('\n# 4. 成功运行 → 上游成果注入下游');
    await check('提交全部通过 → success，portResults 有效', async () => {
      await req('POST', '/api/tasks/' + up.id + '/run');
      const claim = (await req('POST', '/api/runs/claim', { workerId: 'smoke-1' })).body.run;
      assert.ok(claim);
      const sub = (await req('POST', '/api/runs/' + claim.runId + '/submit', {
        workerId: 'smoke-1', exitCode: 0, stdout: 'UPSTREAM_OK_PAYLOAD',
        tests: [{ edgeId: claim.interfaces[0].edgeId, side: 'iface', name: '接口存活', passed: true, exitCode: 0 }]
      })).body;
      assert.strictEqual(sub.run.status, 'success');
      assert.strictEqual(sub.run.portResults.length, 1);
      assert.strictEqual(sub.run.portResults[0].valid, true);
    });

    await check('下游认领能拿到 __IN_* 上游文本', async () => {
      await req('POST', '/api/tasks/' + down.id + '/run');
      const claim = (await req('POST', '/api/runs/claim', { workerId: 'smoke-1' })).body.run;
      assert.ok(claim);
      assert.ok(Array.isArray(claim.inputs), 'claim 必须带 inputs');
      assert.strictEqual(claim.inputs.length, 1);
      const inp = claim.inputs[0];
      assert.strictEqual(inp.kind, 'text');
      assert.strictEqual(inp.env, '__IN_dep');
      assert.ok(inp.value.includes('UPSTREAM_OK_PAYLOAD'));
      assert.strictEqual(claim.interfaces[0].direction, 'use');
      await req('POST', '/api/runs/' + claim.runId + '/submit', {
        workerId: 'smoke-1', exitCode: 0, stdout: 'DOWN',
        tests: [{ portId: 'in', side: 'in', name: '输入契约', passed: true, exitCode: 0 }]
      });
    });

    console.log('\n# 5. 图运行拓扑');
    await check('按 from→to 排序，无环', async () => {
      const r = (await req('POST', '/api/graph/run', { taskIds: [down.id, up.id] })).body;
      assert.strictEqual(r.order, 2);
      assert.strictEqual(r.cycle, false);
      await req('POST', '/api/graph/stop');
    });

    await check('队列状态可查', async () => {
      const s = (await req('GET', '/api/state')).body;
      assert.strictEqual(typeof s.queue.running, 'boolean');
    });

    console.log('\n# 6. 引用完整性');
    await check('删模块级联删边', async () => {
      await req('POST', '/api/runs/clear');
      const del = await req('DELETE', '/api/tasks/' + up.id);
      assert.strictEqual(del.status, 200);
      const s = (await req('GET', '/api/state')).body;
      assert.ok(!s.edges.some((e) => e.from === up.id || e.to === up.id));
    });

    console.log('\n# 7. 边界');
    await check('未知 API 404', async () => {
      const r = await req('GET', '/api/nope');
      assert.strictEqual(r.status, 404);
    });
    await check('静态目录穿越被拒', async () => {
      // 用原生 http 发未规范化的路径：fetch 会先在客户端把 %2e%2e 折叠掉
      const status = await new Promise((resolve, reject) => {
        const r = require('http').request(
          { host: '127.0.0.1', port: PORT, path: '/%2e%2e%2fpackage.json', method: 'GET' },
          (res) => { res.resume(); resolve(res.statusCode); }
        );
        r.on('error', reject);
        r.end();
      });
      assert.strictEqual(status, 403, '实际 ' + status);
    });
    await check('workerId 缺失 400', async () => {
      const r = await req('POST', '/api/runs/claim', {});
      assert.strictEqual(r.status, 400);
    });

    console.log('\n' + '-'.repeat(52));
    console.log('通过 ' + pass + ' 项，失败 ' + fails.length + ' 项');
    if (fails.length) {
      for (const f of fails) console.log('  ✗ ' + f.name + '\n    ' + (f.error && f.error.stack));
      process.exitCode = 1;
    }
  } finally {
    stop();
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
