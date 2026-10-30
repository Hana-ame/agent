'use strict';

/**
 * 测试用临时服务：独立端口 + 临时数据目录，跑完就删。
 *
 * 之前 dnd-check / nav-check 默认连 127.0.0.1:8335 —— 那是用户正在看的实例，
 * 测试建节点、建子图，一旦清理失败就会在人家画布上留垃圾。
 * 现在默认自起一个隔离实例，只有显式传 --url 才连已有服务。
 */

const { spawn } = require('child_process');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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

/**
 * @param {object} [opts]
 * @param {string} [opts.storeSeed]  把某个 store.json 复制进来当初始数据
 * @returns {Promise<{base:string, port:number, dataDir:string, stop:()=>Promise<void>}>}
 */
async function startServer(opts) {
  const o = opts || {};
  const port = await freePort();
  const dataDir = path.join(os.tmpdir(), 'taskpad-test-' + Date.now() + '-' + port);
  fs.mkdirSync(dataDir, { recursive: true });
  if (o.storeSeed && fs.existsSync(o.storeSeed)) {
    fs.copyFileSync(o.storeSeed, path.join(dataDir, 'store.json'));
  }

  const child = spawn(process.execPath, [path.join(ROOT, 'server.js')], {
    cwd: ROOT,
    env: Object.assign({}, process.env, { TASKPAD_PORT: String(port), TASKPAD_DATA_DIR: dataDir }),
    stdio: ['ignore', 'pipe', 'pipe']
  });

  let log = '';
  child.stdout.on('data', (d) => { log += d.toString(); });
  child.stderr.on('data', (d) => { log += d.toString(); });

  const deadline = Date.now() + 12000;
  while (Date.now() < deadline) {
    if (log.includes('listening on')) break;
    if (child.exitCode !== null) throw new Error('server 提前退出：\n' + log);
    await sleep(120);
  }
  if (!log.includes('listening on')) throw new Error('server 启动超时：\n' + log);

  async function stop() {
    try { child.kill(); } catch (e) { /* ignore */ }
    await sleep(200);
    try { fs.rmSync(dataDir, { recursive: true, force: true }); } catch (e) { /* ignore */ }
  }

  return { base: 'http://127.0.0.1:' + port, port, dataDir, stop, log: () => log };
}

module.exports = { startServer, freePort, sleep };
