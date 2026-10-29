'use strict';

const { uid, nowIso } = require('./ids');
const { normalizePorts, normalizeTests } = require('./ports');

/** 节点库：前端「模板」下拉用 */
const TEMPLATES = [
  {
    name: 'v8 离线 demo', engine: 'v8', color: '#3b82f6',
    command: 'python3 -m dag8 demo',
    description: '离线 demo：command 节点多波次并行，不调模型。'
  },
  {
    name: 'v8 lint + dry-run', engine: 'v8', color: '#22c55e',
    command: 'python3 -m dag8 lint configs/demo.json && python3 -m dag8 run configs/demo.json --dry-run',
    description: '静态检查配置 + dry-run 验证占位符渲染。'
  },
  {
    name: 'v8 三边流水线', engine: 'v8', color: '#eab308',
    command: 'python3 -m dag8 run configs/demo.json --workers 2 --rpm 3',
    description: '起草 → 挑错 → 定稿；波次并行 + 限速。'
  },
  {
    name: 'v9 知识推理 demo', engine: 'v9', color: '#a855f7',
    command: 'python3 -m v9.demo',
    description: '抓 Wikipedia 三实体 → 判定关系 → 写知识库。'
  },
  {
    name: 'v9 测试套件', engine: 'v9', color: '#06b6d4',
    command: 'python3 -m pytest v9/tests -q',
    description: '运行 v9 全部测试（20 个用例）。'
  },
  {
    name: '自检 echo', engine: 'v8', color: '#f43f5e',
    command: 'echo "TaskPad 自检 OK" && date',
    description: '最小示例，什么都不依赖。'
  }
];

const SEED_DEFS = [
  {
    name: 'v8 离线 demo（最快验证）', engine: 'v8', color: '#3b82f6',
    command: 'python3 -m dag8 demo',
    description: '离线 demo：command 节点多波次并行，不调模型。用于确认服务链路正常。',
    x: 80, y: 60
  },
  {
    name: 'v8 lint + dry-run', engine: 'v8', color: '#22c55e',
    command: 'python3 -m dag8 lint configs/demo.json && python3 -m dag8 run configs/demo.json --dry-run',
    description: '先静态检查配置，再 dry-run 验证占位符渲染（不调 pi）。',
    x: 520, y: 60
  },
  {
    name: 'v8 三边流水线', engine: 'v8', color: '#eab308',
    command: 'python3 -m dag8 run configs/demo.json --workers 2 --rpm 3',
    description: '起草 → 挑错 → 定稿；依赖 lint 通过后再跑。',
    x: 120, y: 420
  },
  {
    name: 'v9 知识推理 demo', engine: 'v9', color: '#a855f7',
    command: 'python3 -m v9.demo',
    description: 'v9 开放世界推理：抓 Wikipedia 三实体 → 判定关系 → 写知识库。需要外网。',
    x: 560, y: 420
  },
  {
    name: 'v9 测试套件', engine: 'v9', color: '#06b6d4',
    command: 'python3 -m pytest v9/tests -q',
    description: '运行 v9 全部测试（20 个用例，约 1s）。',
    x: 980, y: 60
  },
  {
    name: '自检：echo 任务', engine: 'v8', color: '#f43f5e',
    command: 'echo "TaskPad 自检 OK" && date',
    description: '最小示例：输出一句话和当前时间，什么都不依赖。',
    x: 980, y: 420
  }
];

/** 首次启动时填充示例图。返回是否写入 */
function seedIfEmpty(db, cfg) {
  if (db.tasks.length) return false;

  const rootId = db.graphs[0].id;
  db.tasks = SEED_DEFS.map((d) => Object.assign({
    id: uid(), createdAt: nowIso(), graphId: rootId,
    group: d.engine === 'v9' ? 'knowledge' : 'core',
    subgraphId: null,
    scope: '示例任务：在 v8/v9 项目内执行只读检查',
    spec: '按任务描述执行命令，输出结果供下游消费。',
    inputs: normalizePorts([{ id: 'in', name: 'dep', spec: '上游任意成果（可选）', tests: [] }], false),
    outputs: normalizePorts([{ id: 'out', name: 'out', spec: '命令 stdout 文本', capture: 'stdout', tests: [] }], true)
  }, d));

  // 示例接口关系：lint 的产出 → 三边流水线的输入
  const lint = db.tasks.find((t) => t.name.indexOf('lint') >= 0);
  const pipe = db.tasks.find((t) => t.name.indexOf('三边') >= 0);
  if (lint && pipe) {
    db.edges.push({
      id: uid(), from: lint.id, fromPort: 'out', to: pipe.id, toPort: 'in',
      carryName: 'out', name: 'lint 结果', label: 'lint 结果',
      kind: 'data', proto: '', auth: '', spec: 'lint 通过的 stdout', contract: '',
      graphId: rootId, tests: []
    });
  }
  return true;
}

module.exports = { TEMPLATES, SEED_DEFS, seedIfEmpty, normalizePorts, normalizeTests };
