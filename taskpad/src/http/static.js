'use strict';

const fs = require('fs');
const path = require('path');
const { text } = require('./router');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8'
};

/** 判断 child 是否真的在 base 内（用 relative，避免 /data/artifactsX 绕过 /data/artifacts 前缀检查） */
function isInside(base, child) {
  const rel = path.relative(base, child);
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
}

function createStatic({ cfg }) {
  function serve(req, res, urlPath) {
    if (urlPath === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    let rel;
    try { rel = decodeURIComponent(urlPath); } catch (e) { text(res, 400, 'bad path'); return; }
    if (rel === '/') rel = '/index.html';

    const file = path.normalize(path.join(cfg.publicDir, rel));
    if (!isInside(cfg.publicDir, file)) { text(res, 403, 'forbidden'); return; }

    fs.readFile(file, (err, data) => {
      if (err) { text(res, 404, '404 Not Found'); return; }
      const ext = path.extname(file).toLowerCase();
      res.writeHead(200, {
        'Content-Type': MIME[ext] || 'application/octet-stream',
        'Cache-Control': 'no-store'
      });
      res.end(data);
    });
  }

  return { serve, isInside, MIME };
}

module.exports = { createStatic, isInside, MIME };
