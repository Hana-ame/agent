#!/usr/bin/env node
/*
 * test-upstream.js — a minimal stand-in for github.com's git smart-HTTP
 * endpoints, backed by local bare repos.
 *
 *   node test-upstream.js <root> <port>
 *
 *   <root>/<owner>/<repo>.git must be a bare repository.
 *
 * Speaks exactly the four endpoints git-relay proxies, using the real
 * `git upload-pack` / `git receive-pack` binaries, so a push through the relay
 * exercises the same pkt-line + packfile path a real GitHub push does.
 */

"use strict";

const http = require("node:http");
const path = require("node:path");
const fs = require("node:fs");
const { spawn } = require("node:child_process");
const { Transform } = require("node:stream");

const ROOT = path.resolve(process.argv[2] || "./upstream-root");
const PORT = parseInt(process.argv[3] || "9900", 10);
// optional slow link, to reproduce proxy read timeouts on a big clone
const THROTTLE_KBPS = parseInt(process.argv[4] || "0", 10);

const SERVICES = new Set(["git-upload-pack", "git-receive-pack"]);

function throttle(kbps) {
  if (!kbps) return null;
  const bps = kbps * 1024;
  let sent = 0;
  const t0 = Date.now();
  return new Transform({
    transform(chunk, _enc, cb) {
      sent += chunk.length;
      const due = (sent / bps) * 1000;
      const wait = due - (Date.now() - t0);
      if (wait > 0) setTimeout(() => cb(null, chunk), wait);
      else cb(null, chunk);
    },
  });
}

function pkt(s) {
  return (Buffer.byteLength(s) + 4).toString(16).padStart(4, "0") + s;
}

const server = http.createServer((req, res) => {
  const u = new URL(req.url, "http://localhost");
  const m = u.pathname.match(/^\/([^/]+)\/([^/]+)\.git(\/.*)$/);
  if (!m) {
    res.writeHead(404).end("no such repo");
    return;
  }
  const repo = path.join(ROOT, m[1], m[2] + ".git");
  const endpoint = m[3].replace(/\/$/, "");

  if (!fs.existsSync(path.join(repo, "HEAD"))) {
    res.writeHead(404).end("no such repo");
    return;
  }

  if (endpoint === "/info/refs") {
    const svc = u.searchParams.get("service");
    if (!SERVICES.has(svc)) {
      res.writeHead(403).end("bad service");
      return;
    }
    res.writeHead(200, {
      "content-type": `application/x-${svc}-advertisement`,
      "cache-control": "no-cache",
    });
    res.write(pkt(`# service=${svc}\n`) + "0000");
    const p = spawn("git", [svc.replace("git-", ""), "--stateless-rpc", "--advertise-refs", repo]);
    p.stdout.pipe(res, { end: false });
    p.stderr.pipe(process.stderr);
    // end on stdout 'end', not on process 'close': close fires before any
    // downstream transform has drained its queue
    p.stdout.on("end", () => res.end());
    return;
  }

  if (endpoint === "/git-upload-pack" || endpoint === "/git-receive-pack") {
    const svc = endpoint.slice(1).replace("git-", "");
    res.writeHead(200, { "content-type": `application/x-${endpoint.slice(1)}-result`, "cache-control": "no-cache" });
    const p = spawn("git", [svc, "--stateless-rpc", repo]);
    req.pipe(p.stdin);
    p.stderr.pipe(process.stderr);
    // pipe() ends `res` after the throttling transform drains; ending it on
    // process 'close' truncates the packfile and the client reports early EOF
    const t = throttle(THROTTLE_KBPS);
    if (t) p.stdout.pipe(t).pipe(res);
    else p.stdout.pipe(res);
    return;
  }

  res.writeHead(404).end("unknown endpoint");
});

server.listen(PORT, "127.0.0.1", () => {
  process.stdout.write(`test-upstream on http://127.0.0.1:${PORT} serving ${ROOT}\n`);
});
