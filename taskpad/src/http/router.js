'use strict';

const { AppError } = require('../core/errors');

/**
 * 极简声明式路由。
 *
 * 旧实现是一个 520 行的 if/else 链，匹配靠 `p === '/api/xxx'` 与逐个正则，
 * 新增接口必须翻完整段才能确认没冲突。这里换成路由表：
 *   router.get('/api/runs/:id', handler)
 * 匹配失败 → 404；handler 抛 AppError → 对应状态码；其他异常 → 500。
 *
 * handler 签名：({ req, res, params, query, ctx }) => void | Promise<void>
 */
function compile(pattern) {
  if (pattern instanceof RegExp) return { rx: pattern, names: [] };
  const names = [];
  const src = String(pattern).split('/').map((seg) => {
    if (seg.startsWith(':')) { names.push(seg.slice(1)); return '([^/]+)'; }
    if (seg === '*') { names.push('rest'); return '(.*)'; }
    return seg.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }).join('/');
  return { rx: new RegExp('^' + src + '$'), names };
}

function createRouter({ log } = {}) {
  const logger = log || console;
  const routes = [];

  function add(method, pattern, handler) {
    const { rx, names } = compile(pattern);
    routes.push({ method, rx, names, handler, pattern: String(pattern) });
    return api;
  }

  const api = {
    get: (p, h) => add('GET', p, h),
    post: (p, h) => add('POST', p, h),
    put: (p, h) => add('PUT', p, h),
    delete: (p, h) => add('DELETE', p, h),
    any: (p, h) => add('*', p, h),
    routes
  };

  async function handle(req, res, ctx) {
    const url = new URL(req.url, 'http://localhost');
    const p = url.pathname.replace(/\/+$/, '') || '/';
    const method = req.method;

    for (const r of routes) {
      if (r.method !== '*' && r.method !== method) continue;
      const m = p.match(r.rx);
      if (!m) continue;
      const params = {};
      r.names.forEach((n, i) => { params[n] = decodeURIComponent(m[i + 1] || ''); });
      try {
        await r.handler({ req, res, params, query: url.searchParams, url, path: p, ctx });
      } catch (err) {
        sendError(res, err);
      }
      return true;
    }
    return false;
  }

  function sendError(res, err) {
    if (res.headersSent) {
      try { res.end(); } catch (e) { /* ignore */ }
      return;
    }
    if (err instanceof AppError) {
      json(res, err.status, { error: err.message, detail: err.detail });
      return;
    }
    logger.error('[http] unhandled:', (err && err.stack) || err);
    json(res, 500, { error: String((err && err.message) || err) });
  }

  api.handle = handle;
  return api;
}

// ---------- 响应 / 请求辅助 ----------
function json(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(body);
}

function text(res, code, body, type) {
  res.writeHead(code, {
    'Content-Type': type || 'text/plain; charset=utf-8',
    'Cache-Control': 'no-store'
  });
  res.end(body);
}

function readBody(req, limit) {
  const cap = limit || 2 * 1024 * 1024;
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > cap) { reject(new AppError(413, 'body too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => {
      try { resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {}); }
      catch (e) { reject(new AppError(400, 'invalid JSON body')); }
    });
    req.on('error', reject);
  });
}

function readRaw(req, limit) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > limit) { reject(new AppError(413, 'file too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

module.exports = { createRouter, json, text, readBody, readRaw };
