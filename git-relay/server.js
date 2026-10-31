#!/usr/bin/env node
/*
 * git-relay — anonymous git smart-HTTP in front of GitHub.
 *
 *   git clone http://127.0.0.1:8848/owner/repo.git
 *   git push  http://127.0.0.1:8848/owner/repo.git main
 *
 * A *streaming reverse proxy* for the three endpoints of the git smart-HTTP
 * protocol:
 *
 *   GET  /<owner>/<repo>.git/info/refs?service=git-upload-pack   (clone/fetch)
 *   POST /<owner>/<repo>.git/git-upload-pack
 *   GET  /<owner>/<repo>.git/info/refs?service=git-receive-pack  (push)
 *   POST /<owner>/<repo>.git/git-receive-pack
 *
 * Nothing is stored locally: the only copy of the code lives on GitHub. The
 * relay injects a server-side credential, so clients never see or need one,
 * and enforces a repo whitelist before a single byte reaches the upstream.
 *
 * Why a plain proxy works: git's smart-HTTP transport is stateless. Each
 * clone/fetch/push is one GET (capability advertisement) plus one POST
 * (pkt-line request -> pkt-line + packfile response). There is no session
 * server-side, so every byte can be piped straight through.
 *
 * Zero npm dependencies. Node 18+.
 */

"use strict";

const http = require("node:http");
const https = require("node:https");
const tlsMod = require("node:tls");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const crypto = require("node:crypto");
const { spawnSync } = require("node:child_process");

const VERSION = "1.0.0";

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const cfg = {
    host: process.env.GIT_RELAY_HOST || "127.0.0.1",
    port: parseInt(process.env.GIT_RELAY_PORT || "8848", 10),
    upstream: (process.env.GIT_RELAY_UPSTREAM || "https://github.com").replace(/\/+$/, ""),
    allow: [],
    allowFiles: [],
    token: process.env.GIT_RELAY_TOKEN || "",
    tokenFile: process.env.GIT_RELAY_TOKEN_FILE || "",
    basic: process.env.GIT_RELAY_BASIC || "", // "user:pass" required from clients
    readOnly: false,
    hideUnlisted: true, // 404 instead of 403 for non-whitelisted repos
    ledger: process.env.GIT_RELAY_LEDGER || path.join(__dirname, "relay.ledger.jsonl"),
    quiet: false,
    tlsCert: "",
    tlsKey: "",
    tlsGenerate: "",
    trustClientAuth: false,
    maxPushBytes: 0, // 0 = unlimited
    proxy: process.env.GIT_RELAY_PROXY || process.env.HTTPS_PROXY || process.env.https_proxy || "",
    help: false,
  };

  const need = (i, name) => {
    if (i + 1 >= argv.length) die(`missing value for ${name}`);
    return argv[i + 1];
  };

  for (let i = 0; i < argv.length; i++) {
    let a = argv[i];
    let inline = null;
    const eq = a.indexOf("=");
    if (a.startsWith("--") && eq > 0) {
      inline = a.slice(eq + 1);
      a = a.slice(0, eq);
    }
    const val = () => (inline !== null ? inline : need(i, a) && (i++, argv[i]));
    switch (a) {
      case "--host": cfg.host = val(); break;
      case "--port": cfg.port = parseInt(val(), 10); break;
      case "--upstream": cfg.upstream = val().replace(/\/+$/, ""); break;
      case "--allow": cfg.allow.push(val()); break;
      case "--allow-file": cfg.allowFiles.push(val()); break;
      case "--token": cfg.token = val(); break;
      case "--token-file": cfg.tokenFile = val(); break;
      case "--basic": cfg.basic = val(); break;
      case "--read-only": cfg.readOnly = true; break;
      case "--show-unlisted": cfg.hideUnlisted = false; break;
      case "--ledger": cfg.ledger = val(); break;
      case "--quiet": cfg.quiet = true; break;
      case "--tls-cert": cfg.tlsCert = val(); break;
      case "--tls-key": cfg.tlsKey = val(); break;
      case "--tls-generate": cfg.tlsGenerate = val(); break;
      case "--trust-client-auth": cfg.trustClientAuth = true; break;
      case "--max-push-bytes": cfg.maxPushBytes = parseInt(val(), 10); break;
      case "--proxy": cfg.proxy = val(); break;
      case "--no-proxy": cfg.proxy = ""; break;
      case "-h":
      case "--help": cfg.help = true; break;
      default:
        die(`unknown option: ${a}`);
    }
  }
  return cfg;
}

function die(msg) {
  process.stderr.write(`git-relay: ${msg}\n`);
  process.exit(2);
}

const USAGE = `git-relay ${VERSION} — anonymous git smart-HTTP in front of GitHub

usage: node server.js [options]

  --host <addr>          listen address            (default 127.0.0.1)
  --port <n>             listen port               (default 8848)
  --upstream <url>       git host to proxy to      (default https://github.com)
  --allow <rule>         whitelist entry, repeatable:
                           owner/repo        read+write
                           owner/repo:ro     read only
                           owner/*           every repo of that owner
                           *                 everything (dangerous)
  --allow-file <path>    file of whitelist rules, one per line (# comments)
  --token <pat>          upstream credential (GitHub PAT)
  --token-file <path>    read the PAT from a file
  --basic <user:pass>    require HTTP Basic auth from clients (default: anonymous)
  --read-only            reject every push
  --show-unlisted        answer 403 instead of 404 for unlisted repos
  --tls-cert <crt>       serve HTTPS with this certificate
  --tls-key <pem>        matching private key
  --tls-generate <dir>   generate a self-signed CA+leaf into <dir>, then use it
  --trust-client-auth    forward the client's Authorization instead of --token
  --max-push-bytes <n>   reject larger push bodies (0 = unlimited)
  --proxy <url>          reach the upstream through an HTTP proxy
                           e.g. --proxy http://127.0.0.1:10809   (v2rayN http in)
  --no-proxy             never use a proxy for the upstream
  --ledger <path>        JSONL audit log (default ./relay.ledger.jsonl)
  --quiet                no per-request stdout lines
  -h, --help             this text

env: GIT_RELAY_HOST GIT_RELAY_PORT GIT_RELAY_UPSTREAM GIT_RELAY_TOKEN
     GIT_RELAY_TOKEN_FILE GIT_RELAY_BASIC GIT_RELAY_LEDGER
`;

// ---------------------------------------------------------------------------
// whitelist
// ---------------------------------------------------------------------------

function loadWhitelist(cfg) {
  const raw = [];
  for (const f of cfg.allowFiles) {
    let text;
    try {
      text = fs.readFileSync(f, "utf8");
    } catch (e) {
      die(`cannot read allow-file ${f}: ${e.message}`);
    }
    for (const line of text.split(/\r?\n/)) {
      const s = line.replace(/#.*$/, "").trim();
      if (s) raw.push(s);
    }
  }
  for (const r of cfg.allow) raw.push(String(r).trim());

  const rules = [];
  for (const r of raw) {
    if (!r) continue;
    let body = r;
    let mode = "rw";
    const colon = body.lastIndexOf(":");
    if (colon > 0 && body.slice(colon + 1).toLowerCase() === "ro") {
      mode = "ro";
      body = body.slice(0, colon);
    }
    body = body.toLowerCase().replace(/\.git$/, "").replace(/^\/+/, "");
    // normalize "owner/repo" | "owner/*" | "*"
    if (body !== "*" && !/^[a-z0-9_.-]+\/(\*|[a-z0-9_.-]+)$/.test(body)) {
      die(`bad whitelist rule: ${r}  (expected owner/repo, owner/repo:ro, owner/*, * )`);
    }
    rules.push({ body, mode });
  }
  if (rules.length === 0) {
    process.stderr.write(
      "git-relay: WARNING no whitelist configured — every repo will be refused.\n" +
        "           add --allow owner/repo (repeatable) or --allow-file rules.txt\n"
    );
  }
  return rules;
}

/** Most specific rule wins: exact > owner/* > *. Returns "rw" | "ro" | null. */
function aclDecision(rules, slug) {
  const s = slug.toLowerCase().replace(/\.git$/, "");
  const owner = s.split("/")[0];
  let best = null;
  let bestRank = -1;
  for (const r of rules) {
    let rank = -1;
    if (r.body === "*") rank = 0;
    else if (r.body === `${owner}/*`) rank = 1;
    else if (r.body === s) rank = 2;
    if (rank > bestRank) {
      bestRank = rank;
      best = r.mode;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// routing — only the smart-HTTP surface, nothing else reaches upstream
// ---------------------------------------------------------------------------

const ENDPOINTS = new Set(["/info/refs", "/git-upload-pack", "/git-receive-pack"]);

function splitUrl(rawUrl) {
  const q = rawUrl.indexOf("?");
  return {
    pathname: q >= 0 ? rawUrl.slice(0, q) : rawUrl,
    search: q >= 0 ? rawUrl.slice(q) : "",
  };
}

function safeDecode(s) {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

/**
 * /<owner>/<repo>(.git)?/<endpoint>
 * returns { owner, repo, slug, endpoint } or null
 */
function route(pathname) {
  const p = safeDecode(pathname).replace(/^\/+/, "");
  const parts = p.split("/");
  if (parts.length < 3) return null;
  const owner = parts[0];
  let repo = parts[1];
  if (repo.toLowerCase().endsWith(".git")) repo = repo.slice(0, -4);
  if (!/^[A-Za-z0-9._-]+$/.test(owner) || !/^[A-Za-z0-9._-]+$/.test(repo)) return null;
  if (owner === "." || owner === ".." || repo === "." || repo === "..") return null;
  const endpoint = "/" + parts.slice(2).join("/");
  if (!ENDPOINTS.has(endpoint)) return null;
  return { owner, repo, slug: `${owner}/${repo}`, endpoint };
}

// ---------------------------------------------------------------------------
// pkt-line parsing (for push auditing only — the stream itself is untouched)
// ---------------------------------------------------------------------------

function parsePktLines(buf) {
  const out = [];
  let i = 0;
  while (i + 4 <= buf.length) {
    const hex = buf.toString("ascii", i, i + 4);
    if (!/^[0-9a-fA-F]{4}$/.test(hex)) break;
    const len = parseInt(hex, 16);
    if (len === 0) break; // flush-pkt; the packfile may follow
    if (len < 4 || i + len > buf.length) break;
    out.push(buf.slice(i + 4, i + len).toString("utf8"));
    i += len;
  }
  return out;
}

function parseReceivePack(buf) {
  const refs = [];
  for (const line of parsePktLines(buf)) {
    const head = line.split("\0")[0].replace(/\n$/, "");
    const m = head.match(/^([0-9a-f]{40}|[0-9a-f]{64}) ([0-9a-f]{40}|[0-9a-f]{64}) (\S+)$/);
    if (!m) continue;
    refs.push({
      old: m[1],
      new: m[2],
      ref: m[3],
      delete: /^0+$/.test(m[2]),
    });
  }
  return refs;
}

// ---------------------------------------------------------------------------
// misc helpers
// ---------------------------------------------------------------------------

function nowISO() {
  return new Date().toISOString();
}

function basicHeader(user, pass) {
  return "Basic " + Buffer.from(`${user}:${pass}`).toString("base64");
}

function safeEqual(a, b) {
  const ab = Buffer.from(String(a));
  const bb = Buffer.from(String(b));
  if (ab.length !== bb.length) return false;
  return crypto.timingSafeEqual(ab, bb);
}

function resolveToken(cfg) {
  if (cfg.token) return cfg.token.trim();
  if (cfg.tokenFile) {
    try {
      const t = fs.readFileSync(cfg.tokenFile, "utf8").trim();
      if (t) return t;
    } catch (e) {
      process.stderr.write(`git-relay: cannot read token-file: ${e.message}\n`);
    }
  }
  return "";
}

function appendLedger(p, rec) {
  if (!p) return;
  try {
    fs.appendFileSync(p, JSON.stringify(rec) + "\n");
  } catch {
    /* audit log is best-effort, never break the transfer */
  }
}

/**
 * Real client address when sitting behind nginx / Cloudflare.
 * CF-Connecting-IP is set by Cloudflare itself and cannot be spoofed by the
 * client; X-Forwarded-For's left-most entry is only as trustworthy as the
 * proxy in front, which is fine for an audit log.
 */
function clientIP(req) {
  const cf = req.headers["cf-connecting-ip"];
  if (cf) return String(cf).split(",")[0].trim();
  const xff = req.headers["x-forwarded-for"];
  if (xff) return String(xff).split(",")[0].trim();
  return req.socket.remoteAddress || "";
}

function human(n) {
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}K`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1048576).toFixed(1)}M`;
  return `${(n / 1073741824).toFixed(2)}G`;
}

// ---------------------------------------------------------------------------
// upstream agent (direct, or tunnelled through an HTTP CONNECT proxy)
// ---------------------------------------------------------------------------

/**
 * Node has no built-in proxy support, and a SOCKS proxy would need a library.
 * HTTP CONNECT is trivial to implement, which covers v2rayN / Clash / mitmproxy
 * style local proxies — the common case on a mainland-CN box where a direct
 * connection to github.com is unreliable for large fetches.
 */
function makeUpstreamAgent(proxyUrl, proto) {
  const isSecure = proto === "https:";
  const Base = isSecure ? https.Agent : http.Agent;
  if (!proxyUrl) return new Base({ keepAlive: true, maxSockets: 64, maxFreeSockets: 16 });

  let p;
  try {
    p = new URL(proxyUrl);
  } catch {
    die(`bad --proxy url: ${proxyUrl}`);
  }
  if (!/^https?:$/.test(p.protocol)) die(`unsupported proxy scheme: ${p.protocol} (http:// only)`);
  const proxyMod = p.protocol === "https:" ? https : http;
  const proxyPort = p.port || (p.protocol === "https:" ? 443 : 80);
  const proxyAuth = p.username
    ? "Basic " +
      Buffer.from(`${decodeURIComponent(p.username)}:${decodeURIComponent(p.password)}`).toString("base64")
    : "";

  class TunnelAgent extends Base {
    createConnection(options, cb) {
      const host = options.host;
      const port = options.port;
      const headers = { host: `${host}:${port}`, "proxy-connection": "keep-alive" };
      if (proxyAuth) headers["proxy-authorization"] = proxyAuth;

      const creq = proxyMod.request({
        host: p.hostname,
        port: proxyPort,
        method: "CONNECT",
        path: `${host}:${port}`,
        headers,
        setHost: false,
      });
      creq.once("connect", (res, socket, head) => {
        if (res.statusCode !== 200) {
          socket.destroy();
          return cb(new Error(`proxy CONNECT ${host}:${port} -> ${res.statusCode} ${res.statusMessage}`));
        }
        if (head && head.length) socket.unshift(head);
        // NOTE: https.Agent does NOT pass secureEndpoint down to a custom
        // createConnection, so the TLS decision must come from `proto`.
        if (isSecure) {
          cb(null, tlsMod.connect({ socket, servername: host }));
        } else {
          cb(null, socket);
        }
      });
      creq.once("error", cb);
      creq.end();
    }
  }
  return new TunnelAgent({ keepAlive: true, maxSockets: 64, maxFreeSockets: 16 });
}

// ---------------------------------------------------------------------------
// self-signed TLS
// ---------------------------------------------------------------------------

function ensureTLS(cfg) {
  if (cfg.tlsCert && cfg.tlsKey) return { cert: cfg.tlsCert, key: cfg.tlsKey };
  if (!cfg.tlsGenerate) return null;

  const dir = path.resolve(cfg.tlsGenerate);
  fs.mkdirSync(dir, { recursive: true });
  const cert = path.join(dir, "relay-cert.pem");
  const key = path.join(dir, "relay-key.pem");

  if (fs.existsSync(cert) && fs.existsSync(key)) return { cert, key };

  const r = spawnSync(
    "openssl",
    [
      "req", "-x509", "-newkey", "rsa:2048", "-nodes",
      "-keyout", key, "-out", cert,
      "-days", "825",
      "-subj", "/CN=localhost",
      "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1",
      "-addext", "basicConstraints=critical,CA:TRUE",
      "-addext", "keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign",
    ],
    { encoding: "utf8" }
  );
  if (r.status !== 0) {
    die(`openssl failed: ${r.stderr || r.error?.message || "unknown"}`);
  }
  process.stdout.write(`git-relay: generated self-signed cert in ${dir}\n`);
  return { cert, key };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

function main() {
  const cfg = parseArgs(process.argv.slice(2));
  if (cfg.help) {
    process.stdout.write(USAGE);
    return;
  }

  const rules = loadWhitelist(cfg);
  const token = resolveToken(cfg);
  const tls = ensureTLS(cfg);

  const upstream = new URL(cfg.upstream);
  if (!/^https?:$/.test(upstream.protocol)) die(`bad upstream protocol: ${upstream.protocol}`);
  const upstreamMod = upstream.protocol === "https:" ? https : http;
  const upstreamPort = upstream.port || (upstream.protocol === "https:" ? 443 : 80);
  const upstreamBase = upstream.pathname.replace(/\/+$/, "");
  const agent = makeUpstreamAgent(cfg.proxy, upstream.protocol);

  const stats = {
    startedAt: nowISO(),
    requests: 0,
    clones: 0,
    fetches: 0,
    pushes: 0,
    denied: 0,
    errors: 0,
    bytesIn: 0,
    bytesOut: 0,
  };

  function logLine(rec) {
    if (cfg.quiet) return;
    const tag =
      rec.status >= 400 ? "\x1b[31m" : rec.kind === "push" ? "\x1b[35m" : "\x1b[32m";
    const refs = rec.refs && rec.refs.length ? ` refs=${rec.refs.map((r) => r.ref.replace("refs/heads/", "")).join(",")}` : "";
    process.stdout.write(
      `${tag}${String(rec.status).padEnd(3)}\x1b[0m ${rec.method.padEnd(4)} ` +
        `${rec.slug.padEnd(34)} ${rec.endpoint.padEnd(18)} ` +
        `${String(rec.ms).padStart(5)}ms ${human(rec.bytesIn)}${refs}\n`
    );
  }

  function deny(res, code, msg) {
    stats.denied++;
    const body = msg + "\n";
    res.writeHead(code, {
      "content-type": "text/plain; charset=utf-8",
      "content-length": Buffer.byteLength(body),
      "cache-control": "no-store",
    });
    res.end(body);
  }

  function handle(req, res) {
    const t0 = Date.now();
    stats.requests++;
    const { pathname, search } = splitUrl(req.url);

    // --- local control surface (never proxied) ---------------------------
    if (pathname === "/" || pathname === "/_relay") {
      if (req.method !== "GET") return deny(res, 405, "method not allowed");
      const body = JSON.stringify(
        {
          service: "git-relay",
          version: VERSION,
          upstream: cfg.upstream,
          whitelist: rules.map((r) => `${r.body}${r.mode === "ro" ? ":ro" : ""}`),
          rules: { readOnly: cfg.readOnly, anonymousClients: !cfg.basic },
          usage: {
            clone: `git clone ${tls ? "https" : "http"}://<host>:${cfg.port}/<owner>/<repo>.git`,
            push: `git push  ${tls ? "https" : "http"}://<host>:${cfg.port}/<owner>/<repo>.git <branch>`,
          },
        },
        null,
        2
      );
      res.writeHead(200, { "content-type": "application/json; charset=utf-8" });
      return res.end(body);
    }

    if (pathname === "/_relay/status") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8" });
      return res.end(JSON.stringify({ ...stats, whitelist: rules.length, uptimeS: process.uptime() }, null, 2));
    }

    // --- client auth (optional) ------------------------------------------
    if (cfg.basic) {
      const got = req.headers.authorization || "";
      const want = "Basic " + Buffer.from(cfg.basic).toString("base64");
      if (!safeEqual(got, want)) {
        stats.denied++;
        const body = "authentication required\n";
        res.writeHead(401, {
          "WWW-Authenticate": 'Basic realm="git-relay"',
          "content-type": "text/plain",
          "content-length": Buffer.byteLength(body),
        });
        return res.end(body);
      }
    }

    const r = route(pathname);
    if (!r) {
      const body = "git-relay: not a git smart-HTTP endpoint\n";
      res.writeHead(404, { "content-type": "text/plain", "content-length": Buffer.byteLength(body) });
      return res.end(body);
    }

    const isWrite = r.endpoint === "/git-receive-pack";
    if (isWrite && cfg.readOnly) return deny(res, 403, "git-relay: read-only mode, push refused");

    const mode = aclDecision(rules, r.slug);
    if (!mode) {
      // 404 by default: an unlisted repo should look nonexistent, not forbidden
      return deny(res, cfg.hideUnlisted ? 404 : 403, "git-relay: repository not available on this relay");
    }
    if (isWrite && mode === "ro") return deny(res, 403, "git-relay: repository is read-only on this relay");

    // a push to a ref we did not advertise is still a push — count it as one
    const kind = isWrite ? "push" : r.endpoint === "/info/refs" ? (search.includes("receive-pack") ? "push" : "read") : "read";

    // --- build upstream request ------------------------------------------
    const headers = {};
    for (const [k, v] of Object.entries(req.headers)) {
      if (k === "host" || k === "connection" || k === "proxy-connection") continue;
      if (k === "expect") continue; // body is already streaming; skip 100-continue
      if (k === "authorization") continue; // replaced below
      headers[k] = v;
    }
    headers.host = upstream.host;

    let auth = "";
    if (cfg.trustClientAuth && req.headers.authorization) {
      auth = req.headers.authorization;
    } else if (token) {
      // GitHub accepts any username with a PAT; x-access-token is the app-style one
      auth = basicHeader("x-access-token", token);
    }
    if (auth) headers.authorization = auth;

    const upPath = `${upstreamBase}/${r.owner}/${r.repo}.git${r.endpoint}${search}`;

    let bytesIn = 0;
    let pushRefs = [];
    if (isWrite && req.method === "POST") {
      // sniff the first 64 KiB for pkt-line ref commands; piping is unaffected
      let sniffed = 0;
      const chunks = [];
      req.on("data", (c) => {
        bytesIn += c.length;
        if (sniffed < 65536) {
          const take = c.slice(0, 65536 - sniffed);
          chunks.push(take);
          sniffed += take.length;
        }
      });
      req.on("end", () => {
        try {
          pushRefs = parseReceivePack(Buffer.concat(chunks));
        } catch {
          /* audit only */
        }
      });
    } else {
      req.on("data", (c) => {
        bytesIn += c.length;
      });
    }

    const opts = {
      protocol: upstream.protocol,
      hostname: upstream.hostname,
      port: upstreamPort,
      method: req.method,
      path: upPath,
      headers,
      agent,
    };

    const preq = upstreamMod.request(opts, (upRes) => {
      const rh = {};
      for (const [k, v] of Object.entries(upRes.headers)) {
        if (k === "connection" || k === "keep-alive" || k === "transfer-encoding") continue;
        // never let the upstream challenge reach the client: it would make git
        // prompt for credentials the user does not have and cannot use
        if (k === "www-authenticate") continue;
        rh[k] = v;
      }
      // Tell nginx (and anything that honours it) not to buffer this response:
      // a buffered clone waits for the whole packfile before the client sees a
      // byte, which turns a steady 5-minute clone into a 5-minute stall.
      rh["x-accel-buffering"] = "no";
      res.writeHead(upRes.statusCode || 502, rh);
      upRes.pipe(res);

      let bytesOut = 0;
      upRes.on("data", (c) => {
        bytesOut += c.length;
        stats.bytesOut += c.length;
      });
      upRes.on("end", () => {
        if (kind === "push" && upRes.statusCode < 300) stats.pushes++;
        else if (kind === "read") stats.fetches++;
        const rec = {
          ts: nowISO(),
          method: req.method,
          slug: r.slug,
          endpoint: r.endpoint,
          service: (search.match(/service=([^&]+)/) || [, ""])[1],
          status: upRes.statusCode,
          ms: Date.now() - t0,
          bytesIn,
          bytesOut,
          ip: clientIP(req),
          ua: req.headers["user-agent"],
          refs: pushRefs.length ? pushRefs : undefined,
        };
        appendLedger(cfg.ledger, rec);
        logLine(rec);
      });
    });

    preq.on("error", (err) => {
      stats.errors++;
      const rec = {
        ts: nowISO(),
        method: req.method,
        slug: r.slug,
        endpoint: r.endpoint,
        status: 502,
        ms: Date.now() - t0,
        bytesIn,
        bytesOut: 0,
        error: err.message,
        ip: clientIP(req),
      };
      appendLedger(cfg.ledger, rec);
      logLine(rec);
      if (!res.headersSent) {
        const body = `git-relay: upstream error: ${err.message}\n`;
        res.writeHead(502, { "content-type": "text/plain", "content-length": Buffer.byteLength(body) });
        res.end(body);
      } else {
        res.destroy();
      }
    });

    // client vanished mid-clone → tear down the upstream connection too
    res.on("close", () => {
      if (!res.writableFinished) {
        preq.destroy();
      }
    });

    if (cfg.maxPushBytes > 0 && isWrite) {
      const cl = parseInt(req.headers["content-length"] || "0", 10);
      if (cl > cfg.maxPushBytes) {
        req.resume();
        preq.destroy();
        return deny(res, 413, `git-relay: push body too large (${cl} > ${cfg.maxPushBytes})`);
      }
    }

    if (req.method === "GET" || req.method === "HEAD") {
      preq.end();
    } else {
      req.pipe(preq);
    }
  }

  // a throw inside handle() would otherwise just kill the socket and leave the
  // client with "Empty reply from server" and us with no trace
  const guarded = (req, res) => {
    try {
      handle(req, res);
    } catch (e) {
      stats.errors++;
      process.stderr.write(`git-relay: handler error on ${req.method} ${req.url}: ${e.stack || e}\n`);
      if (!res.headersSent) {
        const body = `git-relay: internal error: ${e.message}\n`;
        res.writeHead(500, { "content-type": "text/plain", "content-length": Buffer.byteLength(body) });
        res.end(body);
      } else {
        res.destroy();
      }
    }
  };

  const server = tls
    ? https.createServer({ cert: fs.readFileSync(tls.cert), key: fs.readFileSync(tls.key) }, guarded)
    : http.createServer(guarded);

  // clones of large repos are long; never time them out server-side
  server.requestTimeout = 0;
  server.headersTimeout = 120000;
  server.keepAliveTimeout = 3600000;
  server.setTimeout(0);

  server.on("clientError", (err, socket) => {
    if (socket.writable) socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
  });

  server.listen(cfg.port, cfg.host, () => {
    const addr = server.address();
    const port = addr.port;
    const scheme = tls ? "https" : "http";
    const lines = [
      "",
      `  git-relay ${VERSION}`,
      `  listening   ${scheme}://${cfg.host}:${port}`,
      `  upstream    ${cfg.upstream}`,
      `  auth        ${cfg.basic ? "HTTP Basic (clients)" : "anonymous"} → upstream ${token ? "PAT injected" : "NO CREDENTIAL (read-only public repos)"}`,
      `  upstream via ${cfg.proxy ? "proxy " + cfg.proxy : "direct"}`,
      `  whitelist   ${rules.length} rule(s)${rules.length ? ": " + rules.map((r) => r.body + (r.mode === "ro" ? ":ro" : "")).join(" ") : " — NOTHING ALLOWED"}`,
      `  ledger      ${cfg.ledger}`,
      "",
    ];
    for (const l of rules.slice(0, 20)) {
      lines.push(`    ${l.mode === "ro" ? "ro" : "rw"}  ${l.body}`);
    }
    if (rules.length > 20) lines.push(`    ... and ${rules.length - 20} more`);
    lines.push("");
    if (rules.length) {
      const sample = rules.find((r) => r.body.includes("/") && !r.body.includes("*")) || rules[0];
      const repoPath = sample.body.includes("*") ? "owner/repo" : sample.body;
      lines.push(`  clone  git clone ${scheme}://${cfg.host}:${port}/${repoPath}.git`);
      lines.push(`  push   git push  ${scheme}://${cfg.host}:${port}/${repoPath}.git <branch>`);
      if (cfg.basic) {
        const u = cfg.basic.split(":")[0];
        lines.push(`         (client password = the --basic value; user "${u}")`);
      }
      lines.push("");
      lines.push(`  if this machine has a global git proxy (http.proxy), exclude the relay once:`);
      lines.push(`    git config --global http."${scheme}://${cfg.host}:${port}/".proxy ""`);
      lines.push("");
    }
    if (tls && cfg.tlsGenerate) {
      lines.push(`  trust the self-signed cert once per machine:`);
      lines.push(`    git config --global http."${scheme}://${cfg.host}:${port}/".sslCAInfo ${path.resolve(tls.cert)}`);
      lines.push("");
    }
    process.stdout.write(lines.join("\n") + "\n");
  });

  const shutdown = () => {
    process.stdout.write("\ngit-relay: shutting down\n");
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 1500).unref();
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

if (require.main === module) main();

module.exports = { parseArgs, loadWhitelist, aclDecision, route, parsePktLines, parseReceivePack, makeUpstreamAgent };
