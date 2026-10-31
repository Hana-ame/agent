#!/usr/bin/env node
/*
 * git-bridge — a local HTTP server that exposes ONE git repository to many
 * agents for synchronized editing.
 *
 *     POST /commit?branch=<branch>&message=<msg>   (body = unified diff)
 *
 * For every request the server, inside a FIFO QUEUE (single worker, so all git
 * operations are strictly serialized — no races):
 *   1. git fetch origin
 *   2. clean the working tree + checkout -B <branch> at the latest
 *      origin/<branch> (or origin/<base> for a brand-new branch)
 *   3. git apply --check  then  git apply        (the posted patch)
 *   4. git add -A ; git commit                    (with supplied/derived author)
 *   5. git push origin HEAD:refs/heads/<branch>   (gh-authenticated)
 *   on a non-fast-forward push it re-fetches and replays (up to 3 attempts).
 *
 * Every POST is recorded in a durable JSONL ledger, queryable for status and
 * downstream CI:
 *     GET /commits            list records (filter: ?branch=&status=&limit=)
 *     GET /commits/<id>       one record ( ?checks=1 enriches with GitHub CI )
 *     GET /queue              current queue depth / worker state
 *     GET /health  GET /branches  GET /
 *
 * POST can be synchronous (default: waits for the result, up to --wait-timeout)
 * or asynchronous (?async=1 returns 202 + id; poll GET /commits/<id>).
 * An optional ?notify=<url> makes the server POST the final result to that URL
 * (fire-and-forget) so a downstream CI/orchestrator can be triggered.
 *
 * Auth: `gh` must be logged in; startup runs `gh auth setup-git` so git
 * fetch/push reuse gh credentials (SSH or HTTPS). Optional bearer token via
 * --token (clients send Authorization: Bearer <token>).
 *
 * Zero npm dependencies — Node 18+ builtins only.
 */

"use strict";

const http = require("node:http");
const fs = require("node:fs");
const fsp = fs.promises;
const path = require("node:path");
const os = require("node:os");
const crypto = require("node:crypto");
const { execFile } = require("node:child_process");
const url = require("node:url");

// --------------------------------------------------------------------------- //
//  config
// --------------------------------------------------------------------------- //
function parseArgs(argv) {
  const cfg = {
    repo: process.env.GIT_BRIDGE_REPO || "repo",
    repoUrl: process.env.GIT_BRIDGE_REPO_URL || "",
    base: process.env.GIT_BRIDGE_BASE || "main",
    worktrees: process.env.GIT_BRIDGE_WORKTREES || "", // unused in queue mode, kept for compat
    host: process.env.GIT_BRIDGE_HOST || "127.0.0.1",
    port: parseInt(process.env.GIT_BRIDGE_PORT || "8765", 10),
    token: process.env.GIT_BRIDGE_TOKEN || "",
    author: process.env.GIT_BRIDGE_AUTHOR || "", // "Name:email"
    ledger: process.env.GIT_BRIDGE_LEDGER || "",
    waitTimeoutMs: parseInt(process.env.GIT_BRIDGE_WAIT_TIMEOUT || "60000", 10),
    maxBodyBytes: 64 * 1024 * 1024,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--repo") cfg.repo = next();
    else if (a === "--repo-url") cfg.repoUrl = next();
    else if (a === "--base") cfg.base = next();
    else if (a === "--worktrees") cfg.worktrees = next();
    else if (a === "--host") cfg.host = next();
    else if (a === "--port") cfg.port = parseInt(next(), 10);
    else if (a === "--token") cfg.token = next();
    else if (a === "--author") cfg.author = next();
    else if (a === "--ledger") cfg.ledger = next();
    else if (a === "--wait-timeout") cfg.waitTimeoutMs = parseInt(next(), 10);
    else if (a === "--help" || a === "-h") {
      console.log(`usage: node server.js [options]
  --repo <path>           working clone to operate on (default ./repo)
  --repo-url <url>        clone this URL if --repo does not exist
  --base <branch>         default base for new branches (default main)
  --host <ip> --port <n>  bind address (default 127.0.0.1:8765)
  --token <secret>        require Authorization: Bearer <secret>
  --author "Name:email"   default commit author
  --ledger <path>         JSONL ledger file (default <repo>.ledger.jsonl)
  --wait-timeout <ms>     sync POST wait before returning 202 (default 60000)`);
      process.exit(0);
    } else {
      console.error(`unknown arg: ${a}`);
      process.exit(2);
    }
  }
  cfg.repo = path.resolve(cfg.repo);
  cfg.ledger = cfg.ledger || (cfg.repo.replace(/[\\/]+$/, "") + ".ledger.jsonl");
  return cfg;
}

const CFG = parseArgs(process.argv);

// --------------------------------------------------------------------------- //
//  process helpers (async, non-blocking)
// --------------------------------------------------------------------------- //
function runCmd(file, args, opts = {}) {
  return new Promise((resolve) => {
    const env = { ...(opts.env || process.env) };
    // never let git/gh block on an interactive prompt — fail fast instead
    env.GIT_TERMINAL_PROMPT = env.GIT_TERMINAL_PROMPT || "0";
    env.GIT_ASKPASS = env.GIT_ASKPASS || "echo";
    env.SSH_ASKPASS = env.SSH_ASKPASS || "echo";
    if (!env.GIT_SSH_COMMAND) {
      env.GIT_SSH_COMMAND = "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new";
    }
    execFile(
      file,
      args,
      {
        cwd: opts.cwd,
        env,
        encoding: "utf8",
        maxBuffer: opts.maxBuffer || 64 * 1024 * 1024,
        timeout: opts.timeout || 120000,
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"], // stdin -> EOF: no command can hang on a prompt
      },
      (err, stdout, stderr) => {
        let code = 0;
        if (err) {
          code = typeof err.code === "number" ? err.code : -1;
        }
        resolve({ code, stdout: stdout || "", stderr: stderr || "", signal: err && err.signal });
      }
    );
  });
}

class GitError extends Error {
  constructor(argv, res, cwd) {
    super(`git exit ${res.code}: ${argv.join(" ")}\n${(res.stderr || "").trim()}`);
    this.argv = argv;
    this.code = res.code;
    this.stdout = res.stdout;
    this.stderr = res.stderr;
    this.cwd = cwd;
  }
}

async function git(args, opts = {}) {
  const res = await runCmd("git", args, opts);
  if (res.code !== 0) throw new GitError(args, res, opts.cwd);
  return res.stdout;
}
function gitOk(args, opts = {}) {
  return runCmd("git", args, opts); // never throws; returns {code,...}
}

function outTrim(s) {
  return (s || "").trim();
}

// --------------------------------------------------------------------------- //
//  repo setup
// --------------------------------------------------------------------------- //
async function pathExists(p) {
  try {
    await fsp.access(p);
    return true;
  } catch {
    return false;
  }
}
async function isGitRepo(p) {
  return pathExists(path.join(p, ".git"));
}

async function ensureRepo() {
  if (!(await isGitRepo(CFG.repo))) {
    if (!CFG.repoUrl) {
      throw new Error(`${CFG.repo} is not a git repo and no --repo-url given`);
    }
    if (await pathExists(CFG.repo)) {
      throw new Error(`${CFG.repo} exists but is not a git repo; refusing to clone over it`);
    }
    await fs.promises.mkdir(path.dirname(CFG.repo) || ".", { recursive: true });
    log(`cloning ${CFG.repoUrl} -> ${CFG.repo}`);
    const res = await runCmd("git", ["clone", "--", CFG.repoUrl, CFG.repo], { timeout: 300000 });
    if (res.code !== 0) throw new Error(`clone failed: ${res.stderr}`);
  }
}

async function setupAuth() {
  // make `git` reuse gh's credentials (idempotent; harmless for SSH remotes).
  // gh writes to the global gitconfig, so a stale ~/.gitconfig.lock can make
  // this fail — clear it and retry once.
  const tryOnce = async () => (await runCmd("gh", ["auth", "setup-git"], { timeout: 30000 })).code === 0;
  if (await tryOnce()) return;
  const probe = await runCmd("gh", ["auth", "setup-git"], { timeout: 30000 });
  if (/could not lock config file/i.test(probe.stderr)) {
    const m = probe.stderr.match(/could not lock config file ([^:\s]+):/);
    const lock = m ? m[1] + ".lock" : null;
    if (lock) {
      try {
        await fsp.rm(lock, { force: true });
        log(`removed stale git config lock: ${lock}`);
      } catch {}
      if (await tryOnce()) return;
    }
  }
  log("warning: gh auth setup-git failed; SSH push still works, HTTPS may prompt");
}

// --------------------------------------------------------------------------- //
//  author resolution
// --------------------------------------------------------------------------- //
async function resolveAuthor() {
  if (CFG.author) {
    if (CFG.author.includes(":")) {
      const [n, e] = CFG.author.split(":", 2);
      return { name: n.trim(), email: e.trim() };
    }
    return { name: CFG.author.trim(), email: "" };
  }
  if (process.env.GIT_BRIDGE_AUTHOR_NAME) {
    return { name: process.env.GIT_BRIDGE_AUTHOR_NAME, email: process.env.GIT_BRIDGE_AUTHOR_EMAIL || "" };
  }
  // repo config
  try {
    const n = outTrim(await git(["config", "user.name"], { cwd: CFG.repo }));
    const e = outTrim(await git(["config", "user.email"], { cwd: CFG.repo }));
    if (n && e) return { name: n, email: e };
  } catch {}
  // gh user
  try {
    const login = outTrim((await runCmd("gh", ["api", "user", "--jq", ".login"], { timeout: 15000 })).stdout);
    if (login) return { name: login, email: `${login}@users.noreply.github.com` };
  } catch {}
  return { name: "git-bridge", email: "git-bridge@local" };
}

// --------------------------------------------------------------------------- //
//  ledger
// --------------------------------------------------------------------------- //
const ledger = {
  records: [], // oldest first
  byId: new Map(),
  _chain: Promise.resolve(),
};

function loadLedger() {
  return fsp
    .readFile(CFG.ledger, "utf8")
    .then((data) => {
      for (const line of data.split(/\r?\n/)) {
        const s = line.trim();
        if (!s) continue;
        try {
          const rec = JSON.parse(s);
          ledger.records.push(rec);
          if (rec.id) ledger.byId.set(rec.id, rec);
        } catch {}
      }
      log(`loaded ${ledger.records.length} ledger record(s) from ${CFG.ledger}`);
    })
    .catch(() => {});
}

function persistRecord(rec) {
  // append-only; serialized to avoid interleaved lines
  ledger._chain = ledger._chain
    .then(() => fsp.appendFile(CFG.ledger, JSON.stringify(rec) + "\n"))
    .catch((e) => log(`ledger write failed: ${e.message}`));
  return ledger._chain;
}

function nowISO() {
  return new Date().toISOString();
}
function newId() {
  return crypto.randomBytes(8).toString("hex");
}

function upsertRecord(rec) {
  ledger.records.push(rec);
  ledger.byId.set(rec.id, rec);
}

function listRecords({ branch, status, limit }) {
  let items = ledger.records.slice().reverse(); // newest first
  if (branch) items = items.filter((r) => r.branch === branch);
  if (status) items = items.filter((r) => r.status === status);
  return items.slice(0, limit || 100);
}

function getRecord(key) {
  if (ledger.byId.has(key)) return ledger.byId.get(key);
  for (let i = ledger.records.length - 1; i >= 0; i--) {
    const r = ledger.records[i];
    if (r.sha === key || r.short === key) return r;
  }
  return null;
}

// --------------------------------------------------------------------------- //
//  github slug + checks
// --------------------------------------------------------------------------- //
let _slug = undefined;
async function githubSlug() {
  if (_slug !== undefined) return _slug;
  _slug = null;
  try {
    const remote = outTrim(await git(["config", "--get", "remote.origin.url"], { cwd: CFG.repo }));
    const m = remote.match(/github\.com[:/]([^/]+)\/([^/]+?)(?:\.git)?$/i);
    if (m) _slug = `${m[1]}/${m[2]}`;
  } catch {}
  return _slug;
}

async function fetchChecks(sha) {
  const slug = await githubSlug();
  if (!slug || !sha) return null;
  try {
    const res = await runCmd(
      "gh",
      ["api", `repos/${slug}/commits/${sha}/check-runs`, "--paginate"],
      { timeout: 20000 }
    );
    if (res.code !== 0) return { error: outTrim(res.stderr) || "gh api failed" };
    const data = JSON.parse(res.stdout);
    const runs = (data.check_runs || []).map((r) => ({
      name: r.name,
      status: r.status,
      conclusion: r.conclusion,
      url: r.html_url,
    }));
    // roll up a simple summary
    const concl = {};
    for (const r of runs) concl[r.conclusion || r.status] = (concl[r.conclusion || r.status] || 0) + 1;
    return { total: data.total_count || runs.length, runs, summary: concl };
  } catch (e) {
    return { error: e.message };
  }
}

// --------------------------------------------------------------------------- //
//  git operations for one job
// --------------------------------------------------------------------------- //
async function refExists(ref, cwd) {
  const res = await gitOk(["rev-parse", "--verify", "--quiet", ref + "^{commit}"], { cwd });
  return res.code === 0;
}
async function revParse(ref, cwd) {
  const args = Array.isArray(ref) ? ["rev-parse", ...ref] : ["rev-parse", ref];
  return outTrim(await git(args, { cwd }));
}

const MAX_PUSH_RETRIES = 3;

async function applyPatch(cwd, patchBytes) {
  const tmp = path.join(os.tmpdir(), `gb-${crypto.randomBytes(6).toString("hex")}.patch`);
  await fsp.writeFile(tmp, patchBytes);
  try {
    const chk = await gitOk(["apply", "--whitespace=nowarn", "--check", tmp], { cwd });
    if (chk.code !== 0) {
      return { ok: false, detail: outTrim(chk.stderr + "\n" + chk.stdout) || "git apply --check failed" };
    }
    await git(["apply", "--whitespace=nowarn", tmp], { cwd });
    return { ok: true };
  } finally {
    await fsp.rm(tmp, { force: true });
  }
}

/*
 * Core: apply a patch to <branch> and push. Returns a result object.
 * Assumed to run inside the single worker (no concurrency on the repo).
 */
async function processCommit({ branch, message, base, author, patch }) {
  const repo = CFG.repo;
  let attempt = 0;
  let lastErr = null;
  for (attempt = 1; attempt <= MAX_PUSH_RETRIES; attempt++) {
    // 1. fetch
    await git(["fetch", "origin", "--prune", "--quiet"], { cwd: repo });

    // 2. resolve the start point (concrete SHA)
    const exists = await refExists(`refs/remotes/origin/${branch}`, repo);
    let startRef = exists ? `origin/${branch}` : `origin/${base}`;
    if (!(await refExists(startRef, repo))) {
      if (base !== "HEAD" && (await refExists("refs/remotes/origin/" + base, repo))) {
        startRef = `origin/${base}`;
      } else {
        startRef = "HEAD";
      }
    }
    const tip = await revParse(startRef, repo);

    // 3. clean current tree, then switch to the branch at tip
    await gitOk(["reset", "--hard", "HEAD", "--quiet"], { cwd: repo });
    await gitOk(["clean", "-fdq"], { cwd: repo });
    await git(["checkout", "-B", branch, tip], { cwd: repo });
    await gitOk(["clean", "-fdq"], { cwd: repo });

    // 4. apply patch
    const applied = await applyPatch(repo, patch);
    if (!applied.ok) {
      return {
        ok: false,
        applied: false,
        status: "failed",
        error: "patch does not apply cleanly",
        detail: applied.detail,
        branch,
        base,
        tip,
        attempts: attempt,
      };
    }

    // 5. stage + commit
    await git(["add", "-A"], { cwd: repo });
    const staged = await gitOk(["diff", "--cached", "--quiet"], { cwd: repo });
    if (staged.code === 0) {
      const sha = await revParse("HEAD", repo);
      return {
        ok: true,
        applied: true,
        committed: false,
        status: "noop",
        reason: "no changes after applying patch",
        branch,
        tip: sha,
        attempts: attempt,
      };
    }
    const commitArgs = ["commit", "--quiet", "-m", message];
    if (author && author.name && author.email) {
      commitArgs.push("--author", `${author.name} <${author.email}>`);
    }
    await git(commitArgs, { cwd: repo });
    const sha = await revParse("HEAD", repo);
    const short = await revParse(["--short=7", "HEAD"], repo);

    // 6. push
    const push = await gitOk(["push", "origin", `HEAD:refs/heads/${branch}`], { cwd: repo });
    if (push.code === 0) {
      return {
        ok: true,
        applied: true,
        committed: true,
        pushed: true,
        status: "pushed",
        branch,
        sha,
        short,
        message,
        attempts: attempt,
      };
    }
    const errText = push.stderr || "";
    const isFF =
      /non-fast-forward|fetch first|rejected|cannot lock ref|forced update/i.test(errText);
    lastErr = outTrim(errText);
    if (isFF && attempt < MAX_PUSH_RETRIES) {
      log(`${branch}: push rejected, retry ${attempt}/${MAX_PUSH_RETRIES}`);
      continue;
    }
    return {
      ok: false,
      applied: true,
      committed: true,
      pushed: false,
      status: "failed",
      error: "push failed",
      detail: lastErr,
      branch,
      sha,
      short,
      attempts: attempt,
    };
  }
  return {
    ok: false,
    status: "failed",
    error: "push failed after retries",
    detail: lastErr,
    branch,
    attempts: attempt - 1,
  };
}

// --------------------------------------------------------------------------- //
//  queue + worker
// --------------------------------------------------------------------------- //
const queue = {
  pending: [], // {job, resolve}
  working: false,
  current: null,
};

function enqueue(job) {
  return new Promise((resolve) => {
    queue.pending.push({ job, resolve });
    startWorker();
  });
}

async function startWorker() {
  if (queue.working) return;
  queue.working = true;
  while (queue.pending.length > 0) {
    const { job, resolve } = queue.pending.shift();
    queue.current = job.id;
    const rec = job.rec;
    rec.status = "working";
    rec.startedAt = nowISO();
    const t0 = Date.now();
    try {
      const result = await processCommit(job);
      Object.assign(rec, result);
      rec.elapsedMs = Date.now() - t0;
      rec.finishedAt = nowISO();
      persistRecord(rec);
      resolve(result);
    } catch (e) {
      const result = {
        ok: false,
        status: "failed",
        error: e instanceof GitError ? "git error" : String(e.message || e),
        detail: e instanceof GitError ? outTrim(e.stderr) : String(e.stack || e),
        branch: job.branch,
        attempts: 1,
      };
      Object.assign(rec, result);
      rec.elapsedMs = Date.now() - t0;
      rec.finishedAt = nowISO();
      persistRecord(rec);
      resolve(result);
    }
    if (job.notify) fireNotify(job.notify, rec);
    log(`${job.branch}: ${rec.status} ${rec.short || rec.tip || ""} (${rec.elapsedMs}ms)`);
  }
  queue.working = false;
  queue.current = null;
}

function fireNotify(notifyUrl, payload) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 15000);
  fetch(notifyUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: ctrl.signal,
  })
    .then(() => log(`notify ${notifyUrl} ok`))
    .catch((e) => log(`notify ${notifyUrl} failed: ${e.message}`))
    .finally(() => clearTimeout(t));
}

// --------------------------------------------------------------------------- //
//  HTTP
// --------------------------------------------------------------------------- //
function log(msg) {
  console.log(`[git-bridge] ${msg}`);
}

function sendJSON(res, code, obj, headers) {
  const body = JSON.stringify(obj, null, 0);
  res.writeHead(code, { "Content-Type": "application/json; charset=utf-8", "Content-Length": Buffer.byteLength(body), ...(headers || {}) });
  res.end(body);
}

function checkToken(req, parsed) {
  if (!CFG.token) return true;
  const auth = req.headers["authorization"] || "";
  if (auth.startsWith("Bearer ") && auth.slice(7) === CFG.token) return true;
  const q = parsed.query;
  if (q.token === CFG.token) return true;
  return false;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const len = parseInt(req.headers["content-length"] || "0", 10);
    if (len > CFG.maxBodyBytes) {
      reject(new Error("request body too large"));
      return;
    }
    const chunks = [];
    let total = 0;
    let aborted = false;
    req.on("data", (c) => {
      total += c.length;
      if (total > CFG.maxBodyBytes) {
        aborted = true;
        req.destroy();
        reject(new Error("request body too large"));
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

const server = http.createServer(async (req, res) => {
  const parsed = url.parse(req.url, true);
  const p = parsed.pathname;
  const q = parsed.query;

  if (req.method === "GET" && (p === "" || p === "/")) {
    return sendJSON(res, 200, {
      service: "git-bridge",
      mode: "queue (single worker)",
      endpoints: {
        "POST /commit": "apply patch + commit + push (queue)",
        "GET /commits": "list push records (?branch=&status=&limit=)",
        "GET /commits/:id": "one record (?checks=1 -> GitHub CI)",
        "GET /queue": "queue depth + worker state",
        "GET /health": "repo + gh auth",
        "GET /branches": "remote branches",
      },
      usage: `curl -X POST 'http://${CFG.host}:${CFG.port}/commit?branch=feat&message=msg' --data-binary @my.patch`,
    });
  }

  if (!checkToken(req, parsed)) return sendJSON(res, 401, { ok: false, error: "unauthorized" });

  try {
    if (req.method === "GET" && p === "/health") {
      const head = outTrim(await git(["rev-parse", "HEAD"], { cwd: CFG.repo }));
      const remote = outTrim((await gitOk(["config", "--get", "remote.origin.url"], { cwd: CFG.repo })).stdout);
      const gh = await runCmd("gh", ["auth", "status"], { timeout: 15000 });
      return sendJSON(res, 200, { ok: true, repo: CFG.repo, head, remote, gh_auth: gh.code === 0, base: CFG.base });
    }

    if (req.method === "GET" && p === "/branches") {
      const out = await git(["for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"], { cwd: CFG.repo });
      const branches = out
        .split(/\r?\n/)
        .map((s) => s.trim())
        .filter((s) => s.startsWith("origin/") && s !== "origin/HEAD")
        .map((s) => s.slice("origin/".length))
        .sort();
      return sendJSON(res, 200, { branches });
    }

    if (req.method === "GET" && p === "/queue") {
      return sendJSON(res, 200, {
        pending: queue.pending.length,
        working: queue.working,
        current: queue.current,
        processed: ledger.records.length,
      });
    }

    if (req.method === "GET" && p === "/commits") {
      const limit = Math.min(parseInt(q.limit || "100", 10) || 100, 1000);
      const items = listRecords({ branch: q.branch, status: q.status, limit });
      return sendJSON(res, 200, { count: items.length, commits: items });
    }

    if (req.method === "GET" && p.startsWith("/commits/")) {
      const key = decodeURIComponent(p.slice("/commits/".length));
      const rec = getRecord(key);
      if (!rec) return sendJSON(res, 404, { ok: false, error: "not found" });
      const out = { ...rec };
      if (q.checks === "1" && rec.sha) {
        out.checks = await fetchChecks(rec.sha);
      }
      return sendJSON(res, 200, out);
    }

    if (req.method === "POST" && p === "/commit") {
      return handleCommit(req, res, q);
    }

    return sendJSON(res, 404, { ok: false, error: "not found" });
  } catch (e) {
    const detail = e instanceof GitError ? outTrim(e.stderr) : String(e.stack || e.message || e);
    return sendJSON(res, 500, { ok: false, error: e instanceof GitError ? "git error" : "server error", detail });
  }
});

async function handleCommit(req, res, q) {
  let branch, message, patch, base, author, notify;
  const body = await readBody(req);
  const ctype = (req.headers["content-type"] || "").toLowerCase();

  if (ctype.includes("application/json")) {
    let j;
    try {
      j = JSON.parse(body.toString("utf8") || "{}");
    } catch (e) {
      return sendJSON(res, 400, { ok: false, error: "invalid JSON: " + e.message });
    }
    branch = j.branch;
    message = j.message;
    patch = typeof j.patch === "string" ? Buffer.from(j.patch, "utf8") : j.patch;
    base = j.base;
    author = j.author || null;
    notify = j.notify;
  } else {
    branch = q.branch;
    message = q.message || req.headers["x-message"];
    patch = body;
    base = q.base;
    const an = req.headers["x-author-name"];
    const ae = req.headers["x-author-email"];
    author = an || ae ? { name: an || "", email: ae || "" } : null;
    notify = q.notify;
  }

  if (!branch) return sendJSON(res, 400, { ok: false, error: "missing 'branch'" });
  if (!message) return sendJSON(res, 400, { ok: false, error: "missing 'message'" });
  if (!patch || patch.length === 0) return sendJSON(res, 400, { ok: false, error: "empty patch" });
  if (!base) base = CFG.base;
  if (!author) author = STATE.defaultAuthor;

  const id = newId();
  const rec = {
    id,
    ts: nowISO(),
    branch,
    message,
    base,
    author,
    status: "queued",
    remote: STATE.remote,
    notify: notify || undefined,
  };
  upsertRecord(rec);

  const job = { id, branch, message, base, author, patch, notify, rec };
  const done = enqueue(job);

  if (q.async === "1" || q.async === "true") {
    return sendJSON(res, 202, { ok: true, accepted: true, id, status: "queued", branch, message, poll: `/commits/${id}` });
  }

  // synchronous: wait up to waitTimeoutMs, else fall back to 202 + poll
  let result = null;
  const timer = new Promise((resolve) => setTimeout(() => resolve(null), CFG.waitTimeoutMs));
  result = await Promise.race([done, timer]);
  if (!result) {
    return sendJSON(res, 202, {
      ok: true,
      accepted: true,
      id,
      status: "working",
      branch,
      message: "still processing; poll GET /commits/" + id,
      poll: `/commits/${id}`,
    });
  }
  const code = result.ok ? 200 : result.status === "failed" && (result.error || "").includes("push") ? 502 : 400;
  return sendJSON(res, code, { id, ...result });
}

// --------------------------------------------------------------------------- //
//  state + main
// --------------------------------------------------------------------------- //
const STATE = { defaultAuthor: null, remote: null };

async function main() {
  await ensureRepo();
  await setupAuth();
  STATE.defaultAuthor = await resolveAuthor();
  try {
    STATE.remote = outTrim((await gitOk(["config", "--get", "remote.origin.url"], { cwd: CFG.repo })).stdout);
  } catch {}
  await loadLedger();

  server.listen(CFG.port, CFG.host, () => {
    log(`repo      = ${CFG.repo}`);
    log(`remote    = ${STATE.remote || "(none)"}`);
    log(`base      = ${CFG.base}`);
    log(`author    = ${STATE.defaultAuthor.name} <${STATE.defaultAuthor.email}>`);
    log(`ledger    = ${CFG.ledger}`);
    log(`token     = ${CFG.token ? "on" : "off"}`);
    log(`queue     = single worker (FIFO)`);
    log(`listening on http://${CFG.host}:${CFG.port}`);
  });

  const shutdown = () => {
    log("shutting down");
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 3000).unref();
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((e) => {
  console.error(`[git-bridge] init error: ${e.stack || e}`);
  process.exit(1);
});
