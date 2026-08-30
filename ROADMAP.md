# Vertex-Edge Agent Framework: Development Roadmap

> 每个已完成条目按「问题 / 方案 / 修改 / 测试」记录。测试结果为实机证据。

---

## ✅ v1.0: Core Architecture（已完成）

### 特性 1：事件驱动 DAG 引擎
**问题**：无统一执行引擎，节点/边各自为战。
**方案**：JSON 声明拓扑 + Executor 异步调度。
**修改**：`Graph`（JSON loader/validator）、`Executor`（event loop，`run()/stream()`）。
**测试**：
- 测试方案：拓扑加载、执行结果。
- 测试方法：`pytest tests/test_graph.py tests/test_executor.py`。
- 测试结果：通过；实机 `python examples/run.py examples/simple/config.json` 正常。

### 特性 2：统一 5 段 Edge 管线
**问题**：Edge(路由) 与 Pipeline(编排) 分离，管线每次执行重建，状态难控。
**方案**：把编排逻辑并入 `Edge`，形成 Guard→Pre-Process→Compute→Post-Process→Deliver。
**修改**：`framework/edge.py` 吸收 Pipeline。
**测试**：`tests/test_edge.py` 5 段各阶段 + `tests/test_improvements.py`，全部通过。

### 特性 3：统一消息传递（EdgeSignal）
**问题**：Vertex↔Edge 方法互调散乱。
**方案**：所有交互收敛到 `handle_edge_signal`，信号 `COMPLETED/ABORTED/FAILED`。
**修改**：`framework/vertex.py`、`framework/edge.py`。
**测试**：`tests/test_vertex.py` 状态机 + 信号用例，通过。

### 特性 4：动态分支与剪枝（Guard + Cascading Abort）
**问题**：条件边失败会导致下游死等。
**方案**：Guard 不满足即发 `ABORTED`，级联剪枝；Settlement Barrier 保证 ≥1 成功才 READY。
**修改**：`Edge.condition` + 执行器 Settlement；`examples/conditional_routing`。
**测试**：
- 测试方案：intent=code 时 image 分支被剪、sink 仍成功。
- 测试方法：`python examples/run.py examples/conditional_routing/config.json`。
- 测试结果：无死锁，仅 code 分支执行（回归在 `tests/test_graph.py`）。

### 特性 5：扩展性（子类 + script 键）
**问题**：旧文档声称 script 注入顶层 hook 函数（已失效）。
**方案**：`script` = `文件名[:类名]`，加载 `Vertex/Edge/MapEdge` 子类，行为在 override 方法中。
**修改**：`framework/utils/script_loader.py`（显式类名优先）；文档全部改为子类式。
**测试**：
- 测试方案：多子类文件按显式名加载正确类。
- 测试方法：`load_class_from_script("s1_edges.py:SummarizeEdge", Edge, "SummarizeEdge")`。
- 测试结果：返回 `SummarizeEdge` 而非 `FetchEdge`（`tests/test_script_loader.py` 锁定）；337 tests passed。

### 特性 6：并发控制
**问题**：高扇出无界并发，打爆 API/连接。
**方案**：Semaphore 按 `concurrency_type`（llm/fetch/default）限流。
**修改**：`Executor.__init__(concurrency_config)`。
**测试**：`tests/test_improvements.py` 并发上限用例，通过。

---

## ✅ v2.0: Application-Ready（已完成）

### 特性 1：业务重试与自纠错
**问题**：`post_process` 领域错误需注入反馈重试；旧实现原地改 `self.prompt`，循环后堆积 `[SYSTEM FEEDBACK]`。
**方案**：`retry_policy`（max_retries/backoff_factor/retry_on）+ `_base_prompt` 冻结，每次重试重建 `active_prompt`，执行后恢复。
**修改**：`framework/edge.py`；`tests/test_retry_and_stream.py` 回归（commit `121ea9e`）。
**测试**：
- 测试方案：多次重试/循环后 prompt 不叠加。
- 测试方法：`pytest tests/test_retry_and_stream.py`，断言仅单个 `[SYSTEM FEEDBACK]`。
- 测试结果：通过。

### 特性 2：状态持久化与 Checkpoint
**问题**：长任务中断需可恢复。
**方案**：`SQLiteStateStore` 快照 + `CheckpointedExecutor.resume()`。
**修改**：`framework/executor/checkpoint.py`、`framework/utils/store.py`（`close()`/幂等，防连接泄漏）。
**测试**：`tests/test_checkpoint.py` 快照/恢复，通过。

### 特性 3：HITL 人工审批
**问题**：敏感节点需暂停等审批。
**方案**：`VertexState.PAUSED`、`pause_for_approval()`、`approve()`、JSON `require_approval`。
**修改**：`framework/vertex.py`；`examples/hitl_approval`。
**测试**：`tests/test_hitl.py`，通过。

### 特性 4：实时事件流
**问题**：执行过程不可观测。
**方案**：`executor.stream()` 非阻塞 `asyncio.Queue` + `GraphEvent`。
**修改**：`framework/executor/base.py`。
**测试**：`tests/test_retry_and_stream.py` / `examples/realtime_streaming`，通过。

### 特性 5：有界循环
**问题**：拓扑回边可致死循环。
**方案**：回边需 `max_iterations > 0`，DFS 校验，超界抛 `GraphCycleError`。
**修改**：`framework/graph.py` validate。
**测试**：`tests/test_cycles.py`，通过。

---

## 🌌 v3.0: Enterprise-Grade（进行中）

| # | 特性 | 状态 |
|:-:|---|---|
| 1 | `SubgraphVertex` 嵌套子图（input/output_map、事件冒泡） | ✅ 完成 |
| 2 | `MemoryStore` 全局内存（TTL、命名空间、memory_read/write） | ✅ 完成 |
| 3 | `TelemetryTracker` 成本/延迟追踪 | ✅ 完成 |
| 4 | Race Mode（`wait_policy: any` 抢先完成） | ✅ 完成 |
| 5 | 异步 hook + 动态拓扑（`LinearChain.build`） | ✅ 完成 |
| 6 | `SchemaRegistry` Pydantic schema 校验 | ✅ 完成 |
| 7 | **分布式执行**（解耦 Executor，消息队列多节点 worker） | ⏳ 待做 |

### 待办：分布式执行
**问题**：单进程 `asyncio.Task` 内执行，无法多节点扩展。
**方案**：把边执行提取为可序列化单元（如 `EdgeTask`），经 Redis/RabbitMQ 分发；`GraphEvent`/`MemoryStore` 已可序列化，`SQLiteStateStore` 需抽抽象接口换 PostgreSQL/Redis。
**修改**：（未开始）
**测试**：（未开始）