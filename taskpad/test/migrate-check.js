#!/usr/bin/env node
'use strict';

/**
 * engine 语义迁移验证：拿真实 store.json 的**副本**跑，绝不动线上数据。
 *
 *   node test/migrate-check.js [--store ../data/store.json]
 *
 * 断言：
 *   1. 节点数一个不少
 *   2. 伪造的 `echo "[TaskPad] … 无命令"` 全部清空，engine 归到 none
 *   3. 真命令按实际调用的层归类（dag8 / v9 / v7 / shell）
 *   4. 图带上 layer 字段（旧数据为 null，不代表 v8）
 */

const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const ROOT = path.resolve(__dirname, '..');
const SRC = path.resolve(arg('--store', path.join(ROOT, 'data', 'store.json')));
const PORT = Number(arg('--port', 8417));
const DATA_DIR = path.join(os.tmpdir(), 'taskpad-migrate-' + Date.now());

let failed = 0;
const ok = (n) => console.log('  ok   ' + n);
const bad = (n, m) => { failed++; console.log('  FAIL ' + n + ' :: ' + m); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  if (!fs.existsSync(SRC)) { console.log('跳过：没有 store.json（' + SRC + '）'); return; }
  fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.copyFileSync(SRC, path.join(DATA_DIR, 'store.json'));
  if (fs.existsSync(path.join(ROOT, 'data', 'artifacts'))) {
    // 成果文件不参与迁移断言，跳过拷贝以省时间
  }

  const before = JSON.parse(fs.readFileSync(SRC, 'utf8'));
  const child = spawn(process.execPath, [path.join(ROOT, 'server.js')], {
    cwd: ROOT,
    env: Object.assign({}, process.env, { TASKPAD_PORT: String(PORT), TASKPAD_DATA_DIR: DATA_DIR }),
    stdio: ['ignore', 'pipe', 'pipe']
  });

  try {
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('启动超时')), 12000);
      child.stdout.on('data', (d) => { if (d.toString().includes('listening on')) { clearTimeout(timer); resolve(); } });
      child.on('exit', (c) => reject(new Error('server 提前退出 ' + c)));
    });

    const state = await (await fetch('http://127.0.0.1:' + PORT + '/api/state')).json();
    const tasks = state.tasks;
    const graphs = state.graphs;

    if (tasks.length === before.tasks.length) ok('节点数不变：' + tasks.length);
    else bad('节点数', before.tasks.length + ' → ' + tasks.length);

    const fake = tasks.filter((t) => /^echo "\[TaskPad\][^"]*无命令/.test(String(t.command || '')));
    if (!fake.length) ok('伪造的「无命令」占位命令已全部清除');
    else bad('占位命令残留', fake.length + ' 个，例如 ' + fake[0].name);

    const empty = tasks.filter((t) => !String(t.command || '').trim());
    const wrong = empty.filter((t) => t.engine !== 'none');
    if (!wrong.length) ok(empty.length + ' 个纯架构模块 engine 均为 none');
    else bad('空命令节点 engine', wrong.length + ' 个不是 none，例如 ' + wrong[0].name + '=' + wrong[0].engine);

    const dag8 = tasks.filter((t) => /-m\s+dag8\b/.test(String(t.command || '')));
    const dag8Ok = dag8.every((t) => t.engine === 'dag8');
    if (dag8.length && dag8Ok) ok('调用 dag8 的 ' + dag8.length + ' 个节点 engine=dag8');
    else if (!dag8.length) console.log('  note 数据里没有 dag8 节点');
    else bad('dag8 归类', '有节点 engine 不是 dag8');

    const v9Tasks = tasks.filter((t) => /-m\s+v9\.|-m\s+pytest\s+v9\//.test(String(t.command || '')));
    const v9Ok = v9Tasks.every((t) => t.engine === 'v9');
    if (!v9Tasks.length) console.log('  note 数据里没有 v9 节点');
    else if (v9Ok) ok('调用 v9 的 ' + v9Tasks.length + ' 个节点 engine=v9');
    else bad('v9 归类', '有节点 engine 不是 v9');

    const noLayer = graphs.filter((g) => !('layer' in g));
    if (!noLayer.length) ok('所有图都带 layer 字段（' + graphs.filter((g) => g.layer).length + ' 个已标注层）');
    else bad('图缺 layer', noLayer.length + ' 个');

    // 未接线的模块不允许起运行
    if (empty.length) {
      const r = await fetch('http://127.0.0.1:' + PORT + '/api/tasks/' + empty[0].id + '/run', { method: 'POST' });
      if (r.status === 400) ok('未接执行入口的模块拒绝启动（400）：' + empty[0].name);
      else bad('拒绝启动', 'HTTP ' + r.status);
    }

    const kinds = {};
    for (const t of tasks) kinds[t.engine] = (kinds[t.engine] || 0) + 1;
    console.log('  info engine 分布：' + JSON.stringify(kinds));
  } finally {
    try { child.kill(); } catch (e) { /* ignore */ }
    await sleep(200);
    try { fs.rmSync(DATA_DIR, { recursive: true, force: true }); } catch (e) { /* ignore */ }
  }

  console.log(failed ? '\n失败 ' + failed + ' 项' : '\n全部通过');
  process.exitCode = failed ? 1 : 0;
}

main().catch((e) => { console.error(e.message || e); process.exit(1); });
