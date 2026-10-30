'use strict';

const { uid, nowIso } = require('./ids');
const { normalizePorts } = require('./ports');

/**
 * 首次启动的示例数据。
 *
 * 注意：节点库的模板**不在这里**——模板由 core/runtimes.js 按 v8/v9 的真实能力
 * （dag8 的 18 个子命令、v8/configs 下真实存在的 config、v9 只有 demo）动态生成，
 * 写死一份 demo 清单只会和真实能力越走越远。
 */

/** 示例模块：每条的 command 都是**真能跑通**的，不放占位 echo */
const SEED_DEFS = [
  {
    name: 'dag8 demo（离线，不调模型）', engine: 'dag8', color: '#3b82f6',
    command: 'python -m dag8 demo',
    description: 'command 节点多波次并行，验证 v8 调度链路，不花 token。',
    group: 'v8', x: 80, y: 60
  },
  {
    name: 'dag8 lint · demo.json', engine: 'dag8', color: '#0ea5e9',
    command: 'python -m dag8 lint configs/demo.json',
    description: '静态检查：环 / 不可达输入 / 子图路径。',
    group: 'v8', x: 520, y: 60
  },
  {
    name: 'dag8 run · demo.json', engine: 'dag8', color: '#eab308',
    command: 'python -m dag8 run configs/demo.json --workers 2 --rpm 3',
    description: '起草 → 挑错 → 定稿；免费网关建议 ≤2 并发 + RPM 限速。',
    group: 'v8', x: 120, y: 420
  },
  {
    name: 'v9 demo（一个推理周期）', engine: 'v9', color: '#a855f7',
    command: 'python -m v9.demo',
    description: '抓 Wikipedia → 判关系 → 写 knowledge.sqlite3。需要外网。',
    group: 'v9', x: 560, y: 420
  },
  {
    name: 'v9 测试套件', engine: 'v9', color: '#06b6d4',
    command: 'python -m pytest v9/tests -q',
    description: 'v9 全部用例。',
    group: 'v9', x: 980, y: 60
  },
  {
    name: '自检 echo', engine: 'shell', color: '#f43f5e',
    command: 'echo "TaskPad 自检 OK" && date',
    description: '最小示例：什么都不依赖。',
    group: '', x: 980, y: 420
  }
];

/** 首次启动时填充示例图。返回是否写入 */
function seedIfEmpty(db, cfg) {
  if (db.tasks.length) return false;

  const rootId = db.graphs[0].id;
  db.tasks = SEED_DEFS.map((d) => Object.assign({
    id: uid(),
    createdAt: nowIso(),
    graphId: rootId,
    subgraphId: null,
    scope: '示例模块：在对应层目录内执行只读检查',
    spec: '按模块描述执行命令，输出结果供下游消费。',
    inputs: normalizePorts([{ id: 'in', name: 'dep', spec: '上游任意成果（可选）', tests: [] }], false),
    outputs: normalizePorts([{ id: 'out', name: 'out', spec: '命令 stdout 文本', capture: 'stdout', tests: [] }], true)
  }, d));

  // 示例接口关系：lint 通过 → 才跑整图
  const lint = db.tasks.find((t) => t.name.indexOf('lint') >= 0);
  const run = db.tasks.find((t) => t.name.indexOf('run ·') >= 0);
  if (lint && run) {
    db.edges.push({
      id: uid(), from: lint.id, fromPort: 'out', to: run.id, toPort: 'in',
      carryName: 'out', name: 'lint 结果', label: 'lint 结果',
      kind: 'data', proto: '', auth: '', spec: 'lint 通过的 stdout', contract: '',
      graphId: rootId, tests: []
    });
  }
  return true;
}

module.exports = { SEED_DEFS, seedIfEmpty };
