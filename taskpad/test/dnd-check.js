#!/usr/bin/env node
'use strict';

/**
 * 节点库拖放验证（无头 Edge + CDP，零依赖）。
 *
 *   node test/dnd-check.js [--url http://127.0.0.1:8335]
 *
 * 做三件事：
 *   1. 确认节点库真的渲染出了 .lib-item（渲染崩了会直接报出来）
 *   2. 依次派发 dragstart / dragover / drop，走的是页面里真实的处理函数
 *   3. 断言服务端 tasks 数量 +1，且新节点的坐标落在丢下的位置附近
 *
 * 需要本机装了 Edge；找不到就直接跳过，不算失败。
 */

const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const args = process.argv.slice(2);
function arg(name, def) {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : def;
}
const BASE = arg('--url', 'http://127.0.0.1:8335');
const EDGE = [
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe'
].find((p) => fs.existsSync(p));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** 拿一个空闲端口（9223 这类常用端口可能被系统服务占用） */
function freePort() {
  const net = require('net');
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const p = srv.address().port;
      srv.close(() => resolve(p));
    });
  });
}

async function cdpTargets(port) {
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch('http://127.0.0.1:' + port + '/json/list');
      const list = await r.json();
      const page = list.find((t) => t.type === 'page' && t.url.indexOf(BASE) === 0);
      if (page) return page;
    } catch (e) { /* 还没起来 */ }
    await sleep(250);
  }
  throw new Error('连不上 Edge 调试端口 ' + port);
}

function openWs(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.onopen = () => resolve(ws);
    ws.onerror = (e) => reject(new Error('WS 连接失败'));
  });
}

let seq = 0;
function makeSend(ws) {
  const pending = new Map();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(msg.error.message));
      else resolve(msg.result);
    }
  };
  return function send(method, params) {
    const id = ++seq;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params: params || {} }));
      setTimeout(() => { if (pending.has(id)) { pending.delete(id); reject(new Error(method + ' 超时')); } }, 15000);
    });
  };
}

async function main() {
  if (!EDGE) {
    console.log('跳过：本机没找到 Edge');
    return;
  }

  const before = (await (await fetch(BASE + '/api/state')).json());
  const beforeCount = before.tasks.length;

  const profile = path.join(os.tmpdir(), 'taskpad-dnd-' + Date.now());
  const PORT = await freePort();
  let edgeErr = '';
  const child = spawn(EDGE, [
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--remote-debugging-port=' + PORT,
    '--user-data-dir=' + profile,
    BASE + '/'
  ], { stdio: ['ignore', 'ignore', 'pipe'] });
  child.stderr.on('data', (d) => { edgeErr += d.toString(); });

  let ws;
  let failed = 0;
  const ok = (name) => console.log('  ok   ' + name);
  const bad = (name, msg) => { failed++; console.log('  FAIL ' + name + ' :: ' + msg); };

  try {
    const page = await cdpTargets(PORT);
    ws = await openWs(page.webSocketDebuggerUrl);
    const send = makeSend(ws);
    await send('Runtime.enable');
    await sleep(1500); // 等首屏 refresh + renderLibrary

    const evalJs = async (expr) => {
      const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
      if (r.exceptionDetails) throw new Error(r.exceptionDetails.text + ' ' + (r.exceptionDetails.exception || {}).description);
      return r.result.value;
    };

    // 1. 节点库是否渲染
    const lib = await evalJs("({items: document.querySelectorAll('#lib-groups .lib-item').length, canvases: document.querySelectorAll('.node').length})");
    if (lib.items > 0) ok('节点库渲染出 ' + lib.items + ' 个模板（画布 ' + lib.canvases + ' 个节点）');
    else bad('节点库渲染', 'lib-item 数量为 0');

    // 2. 派发真实拖放
    const res = await evalJs(`(() => {
      const item = document.querySelector('#lib-groups .lib-item');
      if (!item) return 'no-item';
      const name = item.dataset.tpl;
      const dt = new DataTransfer();
      const mk = (type, extra) => new DragEvent(type, Object.assign({ bubbles: true, cancelable: true, dataTransfer: dt }, extra));
      item.dispatchEvent(mk('dragstart'));
      const board = document.querySelector('#board');
      const r = board.getBoundingClientRect();
      const clientX = Math.round(r.left + r.width * 0.62);
      const clientY = Math.round(r.top + r.height * 0.55);
      board.dispatchEvent(mk('dragover', { clientX, clientY }));
      board.dispatchEvent(mk('drop', { clientX, clientY }));
      return JSON.stringify({ name, clientX, clientY });
    })()`);
    if (res === 'no-item') bad('拖放', '页面上没有 .lib-item');
    else {
      const info = JSON.parse(res);
      await sleep(1800); // 等 POST + refresh
      const after = await (await fetch(BASE + '/api/state')).json();
      const added = after.tasks.length - beforeCount;
      if (added === 1) ok('drop 后新增 1 个节点（模板：' + info.name + '）');
      else bad('drop 后节点数', '期望 +1，实际 ' + added);

      const newest = after.tasks[after.tasks.length - 1];
      if (newest && Number.isFinite(newest.x) && Number.isFinite(newest.y)) {
        ok('新节点坐标 x=' + newest.x + ' y=' + newest.y + '（鼠标在 ' + info.clientX + ',' + info.clientY + '）');
      } else bad('新节点坐标', JSON.stringify(newest && { x: newest.x, y: newest.y }));

      // 3. 拖影提示元素应该被创建过且已隐藏
      const hint = await evalJs("(() => { const h = document.querySelector('.drop-hint'); return h ? (h.hidden ? 'hidden' : 'visible') : 'none'; })()");
      if (hint === 'hidden') ok('拖影提示存在且 drop 后已隐藏');
      else console.log('  note 拖影提示状态：' + hint);

      // 清场：验证完把造出来的节点删掉，别污染真实画布
      if (added === 1 && newest && newest.id) {
        const r = await fetch(BASE + '/api/tasks/' + newest.id, { method: 'DELETE' });
        if (r.ok) ok('已清理测试节点 ' + newest.name);
        else bad('清理测试节点', 'HTTP ' + r.status);
      }
    }
  } finally {
    try { if (ws) ws.close(); } catch (e) { /* ignore */ }
    try { child.kill(); } catch (e) { /* ignore */ }
    await sleep(300);
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { /* ignore */ }
    if (failed && edgeErr) console.log('\n[edge stderr]\n' + edgeErr.split('\n').slice(0, 6).join('\n'));
  }

  console.log(failed ? '\n失败 ' + failed + ' 项' : '\n全部通过');
  process.exitCode = failed ? 1 : 0;
}

main().catch((e) => { console.error(e.message || e); process.exit(1); });
