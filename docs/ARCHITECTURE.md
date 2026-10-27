# v7 / v8 / v9 当前架构与边界

> 版本：`2fd27e3`　更新：2026-10-27  
> 接口表：[interfaces.json](interfaces.json)　可拖拽图：[interfaces.html](interfaces.html)

## 1. 一句话

**v7 是 DAG 本体，v8 是执行/组织层，v9 是知识推理层。**  
每层只做一件事；重复的执行体只保留一份。

## 2. 三层边界

| 层 | 只负责 | 不负责 | 入口 |
|---|---|---|---|
| **v7** | 标签 DAG、边匹配、单图 `step()`、processor 协议 | 多图调度、外部 I/O、长期知识 | `v7.load` / `v7.step` |
| **v8** | 节点执行、波次并行、契约、超时/限流、SQLite 轨迹 | 跨 run 知识、v7 的模糊匹配 | `python -m dag8 run` / `dag8.run` |
| **v9** | 实体/关系/来源、推理循环、多跳、实体扩展 | 通用调度（复用 v8） | `CycleEngine` / `v9.demo` |

依赖方向始终是 **v9 → v8 → v7**，不允许反向 import。

## 3. 去重决策

### 3.1 标签 DAG / 边执行：只保留 v7

- v7 是唯一的标签 DAG 引擎：tag 匹配、边选择、DefaultProcessor、Meta-DAG 都在 v7。
- v8 的 `graph` 节点只是**适配器**：把 seed 交给 v7 图，再取 `result_tag`。
- v8 的 `when` / `join` / 波次是**组织调度**，不实现 v7 的模糊匹配和 DefaultProcessor。

### 3.2 单图 `step()`：只保留 v7

v8 原来的 `Dag` / `StepResult` 是第二套 step API，已删除（`b0c3897`）：

- 删除 `v8/dag8/session.py`
- 删除 `Dag` / `StepResult` 导出
- 删除 `v8/tests/test_session.py`（3 条）
- 文档与接口表同步

单图推进现在只有 `v7.step()`；v8 只保留 `dag8.run()` 这种“整图调度”。

### 3.3 LLM 调用：v8 内统一 pi，v7 独立保留 HTTP

| 路径 | 实现 | 用途 |
|---|---|---|
| v8 `graph` 节点 | `PiLLMClient` → `PiAgent` | v8 默认；v7 图在 v8 内也走 pi |
| v8 原生 `pi` 节点 | `PiAgent` | agent 调用、tools、fallback |
| v7 独立运行 | `OpenAICompatibleClient` | 不经过 v8 时使用 |
| HTTP 显式注入 | `OpenAICompatibleClient` | 测试 / legacy；需传 `graph_llm_client` |

- v8 内部只有 **pi** 一个默认 LLM 后端，`graph` 节点不再默认走 HTTP。
- 为保留 HTTP 重试/自适应测试，`graph_llm_client` 仍可显式注入 HTTP client；这不是默认路径。
- v7 独立运行仍可用自己的 HTTP client；跨层全局只留一个后端需要再决定是否删 v7 HTTP。

## 4. 执行链

### 4.1 v8：一次整图调度

```text
config.json
  → v8 scheduler（按 layer / 依赖取就绪节点）
  → node:
       pi      → 模型生成
       script  → DAG8_INPUTS + stdin JSON → stdout
       http    → 外部请求
       db      → SQLite 查询
       graph   → v7 图
  → 契约 expect 校验
  → SQLite 轨迹
```

- 分支用 `when`；汇聚用 `join: "any"`，避免剪枝后死等
- 输出契约只挡格式（JSON / required_keys / pattern），不设字数上限
- 脚本节点失败以非 0 退出；stdout 作为输出

### 4.2 v9：一个推理周期

```text
CycleEngine.step()
  → 选未探索实体对 (a, b)
  → dag8.run(configs/cycle.json, extra_seeds={b, url_a, url_b})
      → get_a | get_b 两路 http 并行
      → gen: v8 graph 节点 → v7 relation 图
      → commit: v8 script 节点，校验来源/关系并写 claims
  → 返回 claim / no_relation / rejected
```

- 来源必须映射到实际抓取的 URL
- 无来源、来源不在抓取集合：拒绝
- 不缓存；每次重新抓取/生成

### 4.3 v9：多跳

```text
CycleEngine.derive_step()
  → find_chain: A→B + B→C
  → dag8.run(configs/derive.json)
      → gen: v7 derive 图
      → commit_derived: 写 derived_from=[c1, c2]，source_url=null
```

- 推不出就 `none`，不写
- 重复的 A→C 不再推
- `derived_from` 是推理链，不是外部来源

## 5. 当前策略

| 策略 | 状态 | 说明 |
|---|---|---|
| 缓存 | **关闭** | v9 executor `use_cache=False`；示例不加 `--cache` |
| 契约 | **保留** | JSON / required_keys / schema / pattern；不设 `min_chars` / `max_chars` |
| 字数 | **无硬上限** | 输出和 prompt 不限制长度 |
| 存储 | **只存知识** | claims / sources / explored；v8 轨迹只作调试 |
| 结果透露 | **限制** | 推理只给两段外部证据，不回灌已有 claims / 最终结果 |
| 来源 | **强校验** | `source_url` 或 `derived_from` 至少一个 |

## 6. 目录

```text
v7/                 DAG 本体（canonical）
v8/                 组织/执行层
  dag8/             config / scheduler / nodes / contracts / store / pi
  configs/use/      三个可直接跑的示例 DAG
v9/                 知识推理
  knowledge.py      entities / claims / sources / explored
  cycle.py          CycleEngine：周期 + 多跳
  configs/          cycle.json / derive.json
  scripts/          commit / commit_derived
docs/               架构 + 接口表 + 可拖拽图
GUIDE.md            使用方式、最佳实践、指标
```

## 7. 接口

- JSON：`docs/interfaces.json`（**22 条**，含 columns / rows / policies / graph）
- HTML：`docs/interfaces.html`（自包含，可筛选、搜索、拖拽）
- 接口变化时先改 JSON，再同步 HTML 内嵌数据。

## 8. 常用命令

```bash
cd v8

# 三个现成 DAG
python configs/use/setup_sales.py
python -m dag8 run configs/use/research_brief.json --workers 2 --rpm 3 --json
python -m dag8 run configs/use/sqlite_report.json --workers 2 --rpm 3 --json
python -m dag8 run configs/use/script_judge.json --workers 2 --rpm 3 --json

# 配置检查 / 不烧模型
python -m dag8 lint configs/use/research_brief.json
python -m dag8 run configs/use/research_brief.json --dry-run

# 知识推理
cd ..
python -m v9.demo

# 全量测试
python -m pytest -q
```

## 9. 已知限制

- **v8 已统一 pi**；v7 独立运行仍保留 HTTP。跨层全局统一需再决定是否删 v7 HTTP。
- **v9 多跳目前是 fixture 验证**：还没有形成“真实外部 A→B→C”链。
- **v8 轨迹仍落 SQLite**：只作调试，不复用；如要“连轨迹都不存”，需要改存储策略。
- **跨 provider fallback**：缺第二家 key，配置可写但真实切换未验证。
- **并发 evolve**：工作区有未提交的 `v8/evolve` 改动，不属于本层文档范围。

## 10. 最近变更

| 提交 | 内容 |
|---|---|
| `pending` | v8 graph 默认 LLM 切到 `PiLLMClient`（pi） |
| `2fd27e3` | 接口表同步删除 `v8.dag.step` |
| `b0c3897` | 删除 v8 `Dag` / `StepResult` 会话 |
| `36bccfb` | 取消输出字数限制 |
| `96d4035` | 接口表 + 可拖拽拓扑图 |
