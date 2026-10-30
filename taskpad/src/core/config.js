'use strict';

const fs = require('fs');
const path = require('path');

/**
 * 配置与路径解析。
 * 职责边界：只回答「文件在哪、上限多少、引擎有哪些」，不持有状态、不做 I/O 写操作。
 */

const DEFAULT_ENGINES = {
  v8: { label: 'V8', cwd: '.', env: { TASK_ENGINE: 'v8' }, timeoutMs: 1800000, color: '#3b82f6' },
  v9: { label: 'V9', cwd: '.', env: { TASK_ENGINE: 'v9' }, timeoutMs: 1800000, color: '#a855f7' }
};

/** 所有魔法数字集中在这里，禁止再散落到业务代码 */
const LIMITS = {
  runCap: 400,                 // 最多保留运行记录
  traceCap: 200,               // 最多保留 trace
  outputBudget: 400000,        // 单次运行输出上限（字符，超出丢最旧）
  missGraceMs: 120000,         // 定时错过多少毫秒内仍补跑
  heartbeatTimeoutMs: 90000,   // worker 心跳超时 → 回收
  sweepIntervalMs: 15000,      // 失联扫描周期
  tickIntervalMs: 1000,        // 定时器 tick 周期
  ssePingMs: 25000,            // SSE 保活
  jsonBodyLimit: 2 * 1024 * 1024,
  fileBodyLimit: 256 * 1024 * 1024,
  stateDebounceMs: 50,         // 全量 state 广播合并窗口
  workerCap: 100,
  workerPersistMs: 20000,
  inputTextLimit: 100000,      // 上游文本注入上限，防止 prompt/env 爆炸
  graphTailCap: 500            // 单次图运行最多排入的节点数
};

function readJsonIfExists(file) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (e) {
    return null;
  }
}

/**
 * @param {string} rootDir taskpad 目录
 */
function loadConfig(rootDir) {
  const ROOT = path.resolve(rootDir || path.join(__dirname, '..', '..'));

  let config = {};
  if (process.env.TASKPAD_CONFIG) {
    config = readJsonIfExists(process.env.TASKPAD_CONFIG) || {};
  } else {
    config = readJsonIfExists(path.join(ROOT, 'config.json')) || {};
  }

  const port = Number(process.env.TASKPAD_PORT || config.port || 8330);
  const projectRoot = path.resolve(ROOT, config.root || '..');

  const dataDir = process.env.TASKPAD_DATA_DIR
    ? path.resolve(process.env.TASKPAD_DATA_DIR)
    : path.join(ROOT, 'data');

  const engines = config.engines && Object.keys(config.engines).length
    ? JSON.parse(JSON.stringify(config.engines))
    : JSON.parse(JSON.stringify(DEFAULT_ENGINES));

  // 运行时补齐：节点上的 engine 现在指的是「这层怎么被执行」，
  // 只有 dag8 有 CLI，v7/v9 是库，shell 是兜底，none 表示纯架构模块（不执行）。
  const RUNTIME_DEFAULTS = {
    dag8: { label: 'V8 · dag8', cwd: 'v8', color: '#3b82f6' },
    v7: { label: 'V7 · 本体', cwd: '.', color: '#22c55e' },
    v9: { label: 'V9 · 知识推理', cwd: '.', color: '#a855f7' },
    shell: { label: 'Shell', cwd: '.', color: '#64748b' },
    none: { label: '未接线', cwd: '.', color: '#4b5563' }
  };
  for (const [id, def] of Object.entries(RUNTIME_DEFAULTS)) {
    if (!engines[id]) engines[id] = { label: def.label, cwd: def.cwd, env: {}, timeoutMs: 1800000, color: def.color };
  }

  for (const key of Object.keys(engines)) {
    const e = engines[key];
    e.label = e.label || key.toUpperCase();
    e.cwd = path.resolve(projectRoot, e.cwd || '.');
    e.timeoutMs = Number(e.timeoutMs) || 1800000;
    e.color = e.color || '#3b82f6';
    e.env = e.env || {};
    fs.mkdirSync(e.cwd, { recursive: true });
  }

  return {
    root: ROOT,
    publicDir: path.join(ROOT, 'public'),
    dataDir,
    storeFile: path.join(dataDir, 'store.json'),
    artifactDir: path.join(dataDir, 'artifacts'),
    projectRoot,
    port,
    // 任务命令里的 Python 入口。仓库文档写的是 python3（WSL），Windows 上是 python，
    // 所以不写死，交给 config.json / 环境变量决定。
    python: process.env.TASKPAD_PYTHON || config.python || 'python',
    engines,
    limits: LIMITS,
    engineIds: Object.keys(engines),
    /** 引擎对外投影（不含 cwd/env 等内部字段） */
    enginesView: Object.keys(engines).map((k) => ({ id: k, label: engines[k].label, color: engines[k].color }))
  };
}

module.exports = { loadConfig, LIMITS, DEFAULT_ENGINES };
