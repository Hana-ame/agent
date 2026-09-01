# Vertex-Edge-Agent: Architecture Review & Advice

> 本文档按「问题/方案/修改/测试」组织：每项问题给出解决方案、实际改动与实测结果。
> 状态：评审中被确认的问题均已处置，**355 tests passed**。

---

## Part 1: `hn_ai_report` 示例 — 建议

### 问题 1：`hn_edges.py` 曾是死代码（与 config 内联代码重复）

#### 问题
旧版 config 内联 Python 逻辑，`hn_edges.py` 定义的类没人引用，两条路线并存让人困惑。

#### 方案
走「类扩展」路线：`script: hn_edges.py:ClassName` 从 config 引用子类，删除内联代码。

#### 修改
- `examples/hn_ai_report/config.json`：内联 `action_code` 改为 `script` 指向 `hn_edges.py`
  （`FetchTopStoriesEdge`/`FilterEdge`/`ProcessStoriesMap`）。每个 step 用 `script: hn_edges.py:FetchCommentsEdge` 等显式类名。

#### 测试
**测试方案**：示例端到端产出报告。**测试方法**：`python examples/hn_ai_report/demo.py`（config 内嵌 proxy）。**测试结果**：生成 `report.md`（约 99 行），5 帖有效。

### 问题 2：JSON 内联 Python 不可维护

#### 问题
旧 `action_code` 是 `\n` 转义的 Python 字符串，无法高亮/lint/debug。

#### 方案
改用 `script`（外部 `.py` 文件）承载自定义逻辑。

#### 修改
- `config.json` 移除内联代码段；逻辑进 `hn_edges.py` 子类。

#### 测试
**测试方案**：config 无内联 `action_code`。**测试方法**：`grep action_code examples/hn_ai_report/config.json`。**测试结果**：0 处。

### 问题 3：网络调用无错误处理

#### 问题
`hn_fetch` 无重试/降级，HN API 挂了就崩。

#### 方案
fetch 步骤声明 `timeout`（默认 30s），错误经 `EdgeSignal.FAILED` 隔离，不拖垮整图。

#### 修改
- `examples/hn_ai_report/hn_edges.py`：fetch 接受 settings 的 `timeout`；`tests/test_s1_edges.py` 同款覆盖。

#### 测试
**测试方案**：fetch 超时/失败不导致 Executor 崩。**测试方法**：`pytest tests/test_s1_edges.py -q`。**测试结果**：通过。

### 问题 4：示例缺独立 README

#### 问题
只在 examples 根 README 有一行，无专属说明。

#### 方案
补 `examples/hn_ai_report` 专属记录（并入 `examples/README.md` + `ai_report_notes.md`）。

#### 修改
- `examples/README.md`：补 `hn_ai_report`/`s1_ai_report_map`/`sensenova` 三行；`ai_report_notes.md` 记录架构/配置/对比。

#### 测试
**测试方案**：索引 19 行与 19 个示例目录一致。**测试方法**：`grep -c '^| **`' examples/README.md`。**测试结果**：19。

---

## Part 2: 框架架构 — 建议（处置状态）

| # | 议题 | 状态 | 处置 |
|---|---|---|---|
| 1 | Graph 与 Executor 双执行路径 | ✅ 已收敛 | 执行并入 `Executor`；`Graph` 纯数据容器（+ `to_dict/to_json`） |
| 2 | `Vertex` 巨型类（5 种 action 内联） | ✅ 已收敛 | 重构后 Vertex 为状态机容器，计算逻辑移入 Edge/子类 |
| 3 | Context 用裸 dict | ✅ 已收敛 | `settings`/channel 显式携带；`ExecutionContext` 提供 agents/memory/telemetry | 
| 4 | `exec()/eval()` 注入面 | ✅ 已移除 | framework 0 处 `exec(/eval(`；改用子类 override + `edge_transform` |
| 5 | Edge 类型不完整（ERROR/FEEDBACK） | ✅ 已收敛 | `EdgeSignal.ABORTED/FAILED` 统一信号 + Settlement Barrier 剪枝 |
| 6 | 无可观测性 | ✅ 已实现 | `executor.stream()` + `GraphEvent` + `TelemetryTracker` |
| 7 | SubGraph 浅隔离 | ✅ 已实现 | `SubgraphVertex` + input/output_map 边界翻译 + 事件冒泡 |
| 8 | HTTP client 不复用 | ✅ 已修复 | `_client_for(settings)` 按 proxy 缓存；`close()` 幂等清理 |

---

## Part 3: 确认的 Bug（处置状态）

### Bug 1：`GraphBuilder.vertex()` 忽略自定义 script
**问题**：script 存错 key（`vc["pipeline"]`）被静默丢弃。**方案/修改**：`vc["script"] = script`（已核实）。**测试**：`tests/test_improvements.py` 通过。

### Bug 2：Edge prompt 跨迭代累积
**问题**：`retry_policy` 原地改 `self.prompt`，循环后堆 `[SYSTEM FEEDBACK]`。**方案/修改**：冻结 `_base_prompt`，每次重试重建 `active_prompt`，执行后恢复（commit `121ea9e`）。**测试**：`tests/test_retry_and_stream.py` 断言单 feedback 块，通过。

### Bug 3：README 引用不存在的方法
**问题**：`from_json_file()` 不存在。**方案/修改**：文档统一 `from_json()`。**测试**：`grep from_json README.md` 无 `from_json_file`。

---

## Part 4: 待办（未开始）

| # | 事项 | 测试状态 |
|---|---|---|
| 1 | 分布式执行（ROADMAP v3 #7） | 未开始 |
| 2 | `dynamic_topology` 运行时图增长压测 | 示例可用；无专门压测 |
| 3 | S1/HN 抓取的 24h 窗口拿不到旧楼主帖 | 已知限制（MapEdge 数据窗口） |

---

> **结论**：示例（`hn_ai_report`/`s1_ai_report_map`）已是「类扩展 + script 显式类名」的推荐示范；
> 框架所列问题已全部处置，**355 tests passed**。