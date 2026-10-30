'use strict';

/**
 * 无头 Edge + CDP 的最小驱动（零依赖，Node 18+ 自带 fetch/WebSocket）。
 *
 *   const cdp = await launch(BASE_URL);
 *   const v = await cdp.evalJs('document.title');
 *   await cdp.close();
 *
 * 端口动态分配——9222/9223 之类常被系统服务占用，写死会直接起不来。
 */

const { spawn } = require('child_process');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');

const EDGE_CANDIDATES = [
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe'
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function findEdge() {
  return EDGE_CANDIDATES.find((p) => fs.existsSync(p)) || null;
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const p = srv.address().port;
      srv.close(() => resolve(p));
    });
  });
}

async function findPage(port, base, tries) {
  for (let i = 0; i < (tries || 60); i++) {
    try {
      const r = await fetch('http://127.0.0.1:' + port + '/json/list');
      const list = await r.json();
      const page = list.find((t) => t.type === 'page' && t.url.indexOf(base) === 0);
      if (page) return page;
    } catch (e) { /* 浏览器还没起来 */ }
    await sleep(250);
  }
  throw new Error('连不上 Edge 调试端口 ' + port);
}

/**
 * @param {string} base 页面地址，例如 http://127.0.0.1:8335
 * @returns {Promise<{evalJs:(expr:string)=>Promise<any>, send:Function, close:()=>Promise<void>, edgeMissing?:boolean}>}
 */
async function launch(base) {
  const edge = findEdge();
  if (!edge) return { edgeMissing: true, evalJs: async () => null, send: async () => null, close: async () => {} };

  const port = await freePort();
  const profile = path.join(os.tmpdir(), 'taskpad-cdp-' + Date.now());
  let stderr = '';
  const child = spawn(edge, [
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--remote-debugging-port=' + port,
    '--user-data-dir=' + profile,
    base + '/'
  ], { stdio: ['ignore', 'ignore', 'pipe'] });
  child.stderr.on('data', (d) => { stderr += d.toString(); });

  let ws = null;
  let seq = 0;
  const pending = new Map();

  async function connect() {
    const page = await findPage(port, base);
    ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = () => reject(new Error('CDP WebSocket 连接失败'));
    });
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && pending.has(msg.id)) {
        const { resolve, reject } = pending.get(msg.id);
        pending.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      }
    };
  }

  function send(method, params) {
    const id = ++seq;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params: params || {} }));
      setTimeout(() => {
        if (pending.has(id)) { pending.delete(id); reject(new Error(method + ' 超时')); }
      }, 15000);
    });
  }

  async function evalJs(expr) {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) {
      throw new Error(r.exceptionDetails.text + ' ' + ((r.exceptionDetails.exception || {}).description || ''));
    }
    return r.result.value;
  }

  /** 截当前页面（不需要桌面会话，headless 自己渲染） */
  async function screenshot(file, width, height) {
    if (width && height) {
      await send('Emulation.setDeviceMetricsOverride', {
        width, height, deviceScaleFactor: 1, mobile: false
      });
      await sleep(300);
    }
    const r = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(file, Buffer.from(r.data, 'base64'));
    return file;
  }

  async function close() {
    try { if (ws) ws.close(); } catch (e) { /* ignore */ }
    try { child.kill(); } catch (e) { /* ignore */ }
    await sleep(300);
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { /* ignore */ }
    if (stderr) return stderr;
    return '';
  }

  await connect();
  await send('Runtime.enable');
  await send('Page.enable');
  await sleep(1200); // 等首屏 refresh 完成
  return { evalJs, send, close, screenshot, edgeMissing: false };
}

module.exports = { launch, findEdge, sleep, freePort };
