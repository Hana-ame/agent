#!/usr/bin/env node
// JS 语言的模块宿主（runtime host）。
//
// 扫描 modules/js/*/module.json → ESM dynamic import（带 ?v=mtime 破缓存）
// → 调用统一接口 name/describe/run。--watch 时轮询文件 mtime，变更即热重载。
//
// 依赖：Node >= 18，零 npm 依赖。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LAB_ROOT = path.resolve(HERE, "..");

function parseArgs(argv) {
  const opts = { root: LAB_ROOT, watch: false, payload: "world" };
  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--watch") opts.watch = true;
    else if (a === "--root") opts.root = path.resolve(argv[++i]);
    else if (a === "--payload") opts.payload = argv[++i];
    else if (a === "-h" || a === "--help") {
      console.log("用法: node js/host.mjs [--root DIR] [--payload NAME] [--watch]");
      process.exit(0);
    } else {
      console.error(`未知参数: ${a}`);
      process.exit(2);
    }
  }
  return opts;
}

const args = parseArgs(process.argv);
const MODULES_DIR = path.join(args.root, "modules", "js");

function short(sha) {
  return sha ? String(sha).slice(0, 7) : "?";
}

function readLock() {
  const out = {};
  try {
    const raw = fs.readFileSync(path.join(args.root, "modules.lock.json"), "utf8");
    for (const m of JSON.parse(raw).modules || []) out[m.name] = m;
  } catch {
    /* 没有 lock 也能靠目录发现 */
  }
  return out;
}

function discover() {
  try {
    return fs
      .readdirSync(MODULES_DIR, { withFileTypes: true })
      .filter((d) => d.isDirectory() && fs.existsSync(path.join(MODULES_DIR, d.name, "module.json")))
      .map((d) => path.join(MODULES_DIR, d.name))
      .sort();
  } catch {
    return [];
  }
}

// 带 mtime 的 URL 让 Node 绕过 ESM 模块缓存，实现热重载。
async function loadOne(dir) {
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, "module.json"), "utf8"));
  const entry = path.join(dir, manifest.entry || "index.mjs");
  const st = fs.statSync(entry);
  const ns = await import(`${pathToFileURL(entry).href}?v=${st.mtimeMs}`);
  const impl = ns.default;
  if (!impl || typeof impl.run !== "function") {
    throw new Error(`${path.basename(dir)} 的默认导出缺少 run() 方法`);
  }
  return { manifest, impl, entry };
}

async function describeAndRun(dir, lock) {
  const { manifest, impl } = await loadOne(dir);
  const name = impl.name || manifest.name || path.basename(dir);
  const info = lock[name] || {};
  const desc = typeof impl.describe === "function" ? impl.describe() : impl.describe ?? "";
  console.log(`[js] ${name} v${manifest.version || "?"} commit=${short(info.commit)} | ${desc}`);
  const out = await impl.run(args.payload);
  console.log(`[js]   run(${JSON.stringify(args.payload)}) => ${out}`);
}

function entryMap() {
  const out = new Map();
  for (const dir of discover()) {
    try {
      const manifest = JSON.parse(fs.readFileSync(path.join(dir, "module.json"), "utf8"));
      out.set(dir, path.join(dir, manifest.entry || "index.mjs"));
    } catch {
      out.set(dir, path.join(dir, "index.mjs"));
    }
  }
  return out;
}

function snapshot(entries) {
  const out = {};
  for (const [dir, entry] of entries) {
    try {
      const st = fs.statSync(entry);
      out[dir] = `${st.mtimeMs}:${st.size}`;
    } catch {
      out[dir] = "missing";
    }
  }
  return out;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const lock = readLock();
  const dirs = discover();
  if (dirs.length === 0) {
    console.log("[js] 没有发现 JS 模块（先运行：python3 sync.py once）");
    return;
  }
  console.log(`[js] 发现 ${dirs.length} 个 JS 模块`);
  for (const dir of dirs) {
    try {
      await describeAndRun(dir, lock);
    } catch (err) {
      console.log(`[js] ✗ ${path.basename(dir)} 加载失败: ${err.message}`);
    }
  }
  if (!args.watch) return;

  console.log("[js] watch 模式：文件变更后自动重新 import（无需重启）");
  let prev = snapshot(entryMap());
  for (;;) {
    await sleep(500);
    const entries = entryMap();
    const cur = snapshot(entries);
    for (const [dir, sig] of Object.entries(cur)) {
      if (prev[dir] !== sig) {
        console.log(`[js] 检测到模块更新: ${path.basename(dir)}，重新 import`);
        const freshLock = readLock();
        try {
          await describeAndRun(dir, freshLock);
        } catch (err) {
          console.log(`[js] ✗ 重载 ${path.basename(dir)} 失败: ${err.message}`);
        }
      }
    }
    prev = cur;
  }
}

main().catch((err) => {
  console.error(`[js] 宿主异常: ${err.stack || err}`);
  process.exit(1);
});
