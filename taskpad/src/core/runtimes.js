'use strict';

const fs = require('fs');
const path = require('path');

/**
 * 运行时（runtime）—— 对 v7 / v8 / v9 的真实能力建模。
 *
 * 这是本仓库最容易搞错的地方，先把事实钉死：
 *
 *   V7  DAG 本体        纯库（v7.load / v7.step），**没有 CLI**
 *   V8  dag8 组织层      唯一有 CLI 的一层：`python -m dag8 <cmd>`，18 个子命令，
 *                        执行单位是 **config 文件**（v8/configs/*.json），
 *                        轨迹落 v8/data/dag8.sqlite3
 *   V9  知识推理层      **没有 CLI**，只有 Python API（Knowledge / InferenceEngine /
 *                        CycleEngine）+ `python -m v9.demo` 一个演示脚本，
 *                        产物是 v9/data/knowledge.sqlite3 里的实体/关系/来源
 *
 * 所以「engine = 在哪个目录跑命令」是错的建模：
 *   - 只有 v8 能按「子命令 + config」拼出命令；
 *   - v9 和 v7 只能通过脚本/API 调用；
 *   - 一颗图里的节点描述的是**模块**，跟它能不能被执行是两件事。
 *
 * runtime 描述「这层能用什么方式被执行」，action 描述「具体执行哪件事」，
 * 节点上只保留派生出来的 command。
 */

const SHELL = {
  id: 'shell',
  label: 'Shell',
  kind: 'shell',
  color: '#64748b',
  cwd: '.',
  entry: '',
  unit: 'command',
  note: '任意 shell 命令，不依赖 v7/v8/v9。'
};

/** 只有这些子命令是真的（来自 `python -m dag8 --help`，2026-10-30 核对） */
const DAG8_ACTIONS = [
  {
    id: 'run', label: '跑一张图', unit: 'config',
    argv: ['run', '{config}', '{flags}'], args: ['config', 'workers', 'rpm', 'dryRun'],
    desc: '按 config 跑整图；免费网关建议 --workers 2 --rpm 3'
  },
  { id: 'lint', label: '静态检查 config', unit: 'config', argv: ['lint', '{config}'], args: ['config'], desc: '环 / 不可达输入 / 子图路径' },
  { id: 'doctor', label: '健康检查', unit: 'none', argv: ['doctor'], args: [], desc: 'pi / provider / 网关是否可用' },
  { id: 'demo', label: '离线 demo', unit: 'none', argv: ['demo'], args: [], desc: 'command 节点多波次并行，不调模型' },
  { id: 'runs', label: '最近的运行', unit: 'none', argv: ['runs'], args: [], desc: '列出 sqlite 里的历史运行' },
  { id: 'stats', label: '运行统计', unit: 'none', argv: ['stats'], args: [], desc: '按模型统计成功率/时延/错误 Top' },
  { id: 'cost', label: '费用估算', unit: 'none', argv: ['cost'], args: [], desc: '按运行记录估 tokens / 费用' },
  { id: 'models', label: '模型记分卡', unit: 'none', argv: ['models'], args: [], desc: '价格 + 实测成功率/时延' },
  { id: 'route', label: '模型推荐', unit: 'none', argv: ['route'], args: [], desc: '按 stats + 价格推荐模型' },
  { id: 'scaffold', label: '按项目结构生成图', unit: 'path', argv: ['scaffold', '{path}'], args: ['path'], desc: '每个模块一个 command 节点' }
];

const V9_ACTIONS = [
  { id: 'demo', label: '跑一个推理周期', unit: 'none', argv: ['-m', 'v9.demo'], args: [], desc: '抓 Wikipedia → 判定关系 → 写 knowledge.sqlite3' },
  { id: 'test', label: 'v9 测试套件', unit: 'none', argv: ['-m', 'pytest', 'v9/tests', '-q'], args: [], desc: '20 个用例' }
];

const V7_ACTIONS = [
  { id: 'test', label: 'v7 测试套件', unit: 'none', argv: ['-m', 'pytest', 'v7/tests', '-q'], args: [], desc: '标签匹配 / step / Meta-DAG' }
];

function listConfigs(dir, limit) {
  const out = [];
  (function walk(d, depth) {
    if (depth > 3 || out.length >= (limit || 60)) return;
    let entries = [];
    try { entries = fs.readdirSync(d, { withFileTypes: true }); } catch (e) { return; }
    for (const e of entries) {
      if (out.length >= (limit || 60)) return;
      const fp = path.join(d, e.name);
      if (e.isDirectory()) { if (!e.name.startsWith('.') && e.name !== 'data') walk(fp, depth + 1); }
      else if (e.name.endsWith('.json') && e.name !== 'providers.json') out.push(fp);
    }
  })(dir, 0);
  return out;
}

/**
 * @param {object} cfg loadConfig() 的结果
 * @returns {{runtimes: Array, templates: Array}}
 */
function buildRuntimes(cfg) {
  const py = cfg.python || 'python';
  const v8Dir = path.join(cfg.projectRoot, 'v8');
  const configsDir = path.join(v8Dir, 'configs');
  const configs = fs.existsSync(configsDir)
    ? listConfigs(configsDir).map((p) => path.relative(v8Dir, p).split(path.sep).join('/'))
    : [];

  const runtimes = [
    {
      id: 'dag8',
      label: 'V8 · dag8 组织层',
      kind: 'cli',
      color: '#3b82f6',
      cwd: 'v8',
      entry: py + ' -m dag8',
      unit: 'config',
      db: 'v8/data/dag8.sqlite3',
      note: '唯一有 CLI 的一层：18 个子命令，执行单位是 config 文件。',
      actions: DAG8_ACTIONS.map((a) => Object.assign({}, a)),
      configs
    },
    {
      id: 'v9',
      label: 'V9 · 知识推理层',
      kind: 'library',
      color: '#a855f7',
      cwd: '.',
      entry: py + ' -m v9.demo',
      unit: 'cycle',
      db: 'v9/data/knowledge.sqlite3',
      note: '没有 CLI，只有 Python API（Knowledge / InferenceEngine / CycleEngine）；调度复用 v8。',
      actions: V9_ACTIONS.map((a) => Object.assign({}, a)),
      configs: []
    },
    {
      id: 'v7',
      label: 'V7 · DAG 本体',
      kind: 'library',
      color: '#22c55e',
      cwd: '.',
      entry: 'import v7',
      unit: 'graph',
      db: null,
      note: '纯库：v7.load / v7.step / v7.set，没有命令行入口。',
      actions: V7_ACTIONS.map((a) => Object.assign({}, a)),
      configs: []
    },
    Object.assign({}, SHELL, { actions: [], configs: [] })
  ];

  return { runtimes, templates: buildTemplates(runtimes, py) };
}

/** 把 runtime × action 展开成节点库条目（保持前端原来的 {name,engine,color,command,description} 形状） */
function buildTemplates(runtimes, py) {
  const byId = {};
  for (const r of runtimes) byId[r.id] = r;
  const out = [];

  const dag8 = byId.dag8;
  // run / lint 按真实存在的 config 各生成一条，不再写死 demo.json
  const cfgSample = dag8.configs.filter((c) => !c.startsWith('data/')).slice(0, 24);
  for (const c of cfgSample) {
    // 名字必须唯一：节点库条目靠 name 做 key，用 basename 会让 dsh_report/dag.json
    // 和 session_elo/dag.json 撞车，拖出来的永远是第一条
    const key = c.replace(/\.json$/, '');
    out.push({
      name: 'dag8 run · ' + key,
      engine: 'dag8',
      color: dag8.color,
      command: py + ' -m dag8 run ' + c + ' --workers 2 --rpm 3',
      description: '跑 ' + c + '（波次并行 2，RPM 3）'
    });
    out.push({
      name: 'dag8 lint · ' + key,
      engine: 'dag8',
      color: '#0ea5e9',
      command: py + ' -m dag8 lint ' + c,
      description: '静态检查 ' + c
    });
  }
  for (const a of dag8.actions) {
    if (a.unit === 'config') continue; // 上面已按 config 展开
    out.push({
      name: 'dag8 ' + a.id,
      engine: 'dag8',
      color: dag8.color,
      command: py + ' -m dag8 ' + a.argv.join(' ').replace('{path}', '.'),
      description: a.desc
    });
  }
  for (const r of [byId.v9, byId.v7]) {
    if (!r) continue;
    for (const a of r.actions) {
      out.push({
        name: r.id + ' ' + a.id,
        engine: r.id,
        color: r.color,
        command: py + ' ' + a.argv.join(' '),
        description: a.desc
      });
    }
  }
  out.push({
    name: '自检 echo',
    engine: 'shell',
    color: '#f43f5e',
    command: 'echo "TaskPad 自检 OK" && date',
    description: '最小示例，不依赖任何层'
  });
  return out;
}

/** 按 runtime + action + args 拼命令（节点表单用；节点库直接给 command 字符串） */
function buildCommand(runtimes, runtimeId, actionId, args) {
  const r = (runtimes || []).find((x) => x.id === runtimeId);
  if (!r) return null;
  const a = (r.actions || []).find((x) => x.id === actionId);
  if (!a) return null;
  const py = r.entry ? r.entry.split(' ')[0] : 'python';
  let argv = a.argv.map((x) => {
    if (x === '{config}') return (args && args.config) || '{config}';
    if (x === '{path}') return (args && args.path) || '.';
    if (x === '{flags}') {
      const f = [];
      if (args && args.workers) f.push('--workers ' + args.workers);
      if (args && args.rpm) f.push('--rpm ' + args.rpm);
      if (args && args.dryRun) f.push('--dry-run');
      return f.join(' ');
    }
    return x;
  }).filter((x) => x !== '');
  const prefix = r.kind === 'shell' ? '' : py + ' ';
  return (prefix + argv.join(' ')).trim();
}

/** 判断一条命令实际调的是哪一层（迁移 / 兜底用） */
function guessRuntime(command) {
  const c = String(command || '');
  if (!c.trim()) return null;
  if (/-m\s+dag8\b|\bdag8\s+[a-z]/.test(c)) return 'dag8';
  if (/-m\s+v9(\.[a-z]+)?\b|\bv9\.(demo|infer|cycle|expand|knowledge)\b|\bv9\/tests\b|\bpytest\s+v9\//.test(c)) return 'v9';
  if (/-m\s+v7(\.[a-z]+)?\b|\bv7\.(load|step|set)\b|import\s+v7\b|\bv7\/tests\b|\bpytest\s+v7\//.test(c)) return 'v7';
  return 'shell';
}

module.exports = { buildRuntimes, buildTemplates, buildCommand, guessRuntime, listConfigs, DAG8_ACTIONS };
