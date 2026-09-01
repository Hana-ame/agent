# More Elegant Implementations

> 本文档按「问题 / 方案 / 修改 / 测试」组织框架重构建议，标注落地状态。
> 状态：**#1 已落地（MapEdge）**、**#2 部分落地（edge_transform 函数式工厂）**、**#7 已落地（ExecutionContext/async agent）**，
> 其余为可选改进。**355 tests passed**。

---

## 1. 用 `map` 算子替换静态扇出

### 问题
`hn_ai_report` 旧版手写 10 顶点 + 17 边表达「对每个 AI 帖 fetch+summarize」，配置膨胀到 120 行，难以扩并发。

### 方案
`MapEdge`：一条边对列表每个元素跑子管线（pipeline 步骤），fan-in 结果。表达「对每个 X 做 Y」只需一个概念。

### 修改
- `framework/edge.py`：新增 `MapEdge`（pipeline 步骤 + `max_concurrency` Semaphore + fan-in deliver）。
- `examples/hn_ai_report/config.json`：`e_process_stories` 用 `script: hn_edges.py:ProcessStoriesMap` + `settings.pipeline`。
- `examples/s1_ai_report_map/config.json`：同款 `ProcessThreadsMap`。

### 测试
**测试方案**：对每项跑完整管线、并发限流、单项失败不影响其它项。
**测试方法**：`pytest tests/test_improvements.py`（MapEdge 用例）+ 实机 `python examples/hn_ai_report/demo.py`。
**测试结果**：通过；实机生成 `report.md`（99/137 行），4-5 帖逐条产出。

---

## 2. 可组合边阶段：函数式工厂（不只是继承）

### 问题
每个自定义行为都要写一个 `Edge` 子类；纯变换（如 `SelectEdge` 只做 `data[index]`）也要成类，样板多。

### 方案
`edge_transform(pre, post, guard)` 函数式工厂：由普通函数生成 `Edge` 子类；简单 hook 用函数，复杂行为用子类，两者并存。

### 修改
- `framework/edge.py`：`edge_transform()` 工厂（`FunctionalEdge` 动态生成）。
- `examples/scripts`、`s1_edges.py` 的简单步骤可用工厂表达（子类仍保留做复杂 fetch/LLM）。

### 测试
**测试方案**：工厂生成的 edge 行为与子类等价（guard/pre/post 生效）。
**测试方法**：`pytest tests/test_edge.py`（FunctionalEdge 用例）。
**测试结果**：通过。

---

## 3. Vertex 状态机描述符

### 问题
`receive_signal`/executor/checkpoint 分散改 `vertex.state`，非法迁移靠约定，无强制。

### 方案
用描述符把状态迁移表集中声明：非法迁移直接抛错；checkpoint 恢复用 `force_state()` 绕过校验。

### 修改
（未落地——可选改进）当前状态机靠 `Vertex._transition()` 集中校验，但迁移表未声明式化。

### 测试
（未落地）思路：非法迁移用例断言抛 `InvalidTransition`。

---

## 4. 不可变执行上下文 + 类型化槽位

### 问题
`dict` 上下文随处原地改，键冲突静默、数据可用性不可知、运行时 `KeyError`。

### 方案
类型化 `Slot` 声明：边声明 `produces/consumes`，图编译期做数据流检查（缺失上游、槽位类型不匹配、孤儿槽位）。

### 修改
（未落地——可选改进，工作量中等）当前 `Executor` 用 `settings`/channel 显式传递，未做编译期槽位分析。

### 测试
（未落地）思路：构造缺失上游/类型不匹配图，编译期应报错。

---

## 5. 边即数据：函数式变换（已部分落地）

### 问题
`SelectEdge` 等纯变换成类过重。

### 方案
`edge_transform()` 一行生成；config 可直接 `pre_process`/`condition` 函数引用（见 #2）。

### 修改
- `framework/edge.py`：`edge_transform`（落地）。
- config 内联函数字符串写法未启用（保留——避免 `eval()` 注入面，见架构评审问题 10）。

### 测试
**测试方案**：函数式 factory 行为正确。**测试方法**：`pytest tests/test_edge.py`。**测试结果**：通过。

---

## 6. 声明式 Guard DSL

### 问题
旧 `evaluate_condition` 是 15 行 `if/elif` 运算符分发，且用 `eval()` 解析字符串（注入面，已移除）。

### 方案
`operator` 映射表 + 可组合 `Guard`（field/op/value/match，`__and__/__or__`），安全（无 eval）、可序列化、可测试。

### 修改
- `framework/edge.py`：`_threshold/_operator/_field/_match` 从 settings 解析，`operator` 字典分发（已落地）。
- `eval()` 已彻底移除（`grep exec(/eval(` framework/ = 0）。

### 测试
**测试方案**：`>=/>/<=/</==/!=/contains/matches` 各运算符方向正确、组合守卫生效。
**测试方法**：`pytest tests/test_edge.py`（guard 用例）+ `tests/test_graph.py`（conditional_routing 级联剪枝）。
**测试结果**：通过；`python examples/run.py examples/conditional_routing/config.json` 无死锁。

---

## 7. 统一资源生命周期（已落地）

### 问题
`HttpLLMAgent`/`MemoryStore`/`TelemetryTracker` 手动管理；demo `try/finally` 关 agent，长跑泄漏。

### 方案
`ExecutionContext` 异步上下文管理：`async with ctx:` 自动创建并关闭 agent/memory/telemetry；HTTP agent 自带 `__aenter__/__aexit__/close()`。

### 修改
- `framework/executor/base.py`：`ExecutionContext`（已落地）。
- `framework/agents/_http_base.py`：`__aenter__/__aexit__/close()` 幂等关闭 + 清代理缓存（commit `d64aab2`）。

### 测试
**测试方案**：async-with 正常/异常路径都关闭客户端；关闭幂等。
**测试方法**：`pytest tests/test_agents.py`（回归）。
**测试结果**：通过（commit `d64aab2`）。

---

## 影响 × 成本矩阵（现状）

| 变更 | 优雅度影响 | 成本 | 风险 | 状态 |
|---|---|---|---|---|
| 1. `MapEdge` | 🔥🔥🔥 | Medium | Low（additive） | ✅ 已落地 |
| 2. 函数式工厂 | 🔥🔥 | Medium | Low | ✅ `edge_transform` 已落地 |
| 3. 状态机描述符 | 🔥🔥 | Small | Medium | ⏳ 可选 |
| 4. 类型化槽位 | 🔥🔥🔥 | Large | Medium | ⏳ 可选 |
| 5. 边即数据 | 🔥 | Small | Low | ✅ 部分落地 |
| 6. Guard DSL | 🔥🔥 | Small | Low | ✅ operator 表落地，无 eval |
| 7. 资源生命周期 | 🔥 | Small | Low | ✅ 已落地 |

> **优先建议**：#1（已落地）+ #6（已落地）已让 `hn_ai_report` 缩到约 40 行 config、guard 安全可组合。
> 若继续投入，先做 #4（编译期数据流）收益最大。