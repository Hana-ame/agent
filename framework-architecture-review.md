# 🏗️ Vertex-Edge Agent Framework — Architecture Review

> 架构评审结论：核心模型（Vertex 状态机 / Edge 5 段管线 / Executor 异步调度 / 消息传递）设计
> 优良。本文档按「问题/方案/修改/测试」记录评审中发现的问题及其处置状态。

**范围**：`framework/`（17 个源文件）。**测试**：355 tests passed ✅。

---

## 问题 1：`HttpLLMAgent` 会对致命 HTTP 错误重试

### 问题
tenacity 对所有 `httpx.HTTPStatusError` 重试，401/400/404 等认证/参数错误也会被重试
`max_retries` 次，白烧配额。

### 方案
非重试状态码（400/401/403/404 等）抛 `NonRetryableHTTPError`（`ValueError` 子类），
不进 `retry_if_exception_type`；仅 `429/500/502/503/504` 进入重试。

### 修改
- `framework/agents/_http_base.py`：新增 `NonRetryableHTTPError` + 状态码分支；5xx 分类 `RETRYABLE_STATUS`。

### 测试
**测试方案**：4xx 立即失败、5xx/429 重试。**测试方法**：mock 注入 400 与 500，断言调用次数。
**测试结果**：400 一次即失败，500 重试至上限（`tests/test_agents.py` 回归锁定，commit `d64aab2`）。

---

## 问题 2：`HttpLLMAgent` 从不关闭 `httpx.AsyncClient`

### 问题
`AsyncClient` 在 `__init__` 创建，框架从不 `close()`；长跑进程泄漏连接/文件描述符。

### 方案
增加异步上下文管理（`__aenter__`/`__aexit__`）与幂等 `close()`；退出时清空代理客户端缓存。

### 修改
- `framework/agents/_http_base.py`：`__aenter__/__aexit__/close()`；`_proxied_clients` 缓存随 `close()` 清空。

### 测试
**测试方案**：显式 close、幂等、async-with 正常与异常路径。**测试方法**：`tests/test_agents.py` 回归。
**测试结果**：通过（commit `d64aab2`）。

---

## 问题 3：`HumanGateVertex.__repr__` 有重复 return

### 问题
`__repr__` 内两条相同 `return`，第二条是死代码。

### 方案
删除重复行。

### 修改
- `framework/executor/checkpoint.py`：`__repr__` 保留单条 return。

### 测试
**测试方案**：repr 输出正确、无重复逻辑。**测试方法**：`pytest tests/test_checkpoint.py`。
**测试结果**：通过；当前 `__repr__` 为一行 `return f"HumanGateVertex(id=..., state=..., approval=...)"`。

---

## 问题 4：`Pipeline` 与 `Edge` 双执行路径

### 问题
旧架构 `Pipeline`（5 段编排）与 `Edge`（路由）分离，管线每次执行重建，逻辑重复。

### 方案
把编排逻辑并入 `Edge`；`Pipeline` 仅保留为向后兼容别名。

### 修改
- `framework/edge.py`：吸收 guard/pre-process/compute/retry/post-process/schema/memory/telemetry。
- `framework/pipeline.py`：`Pipeline = Edge` 别名 + 错误类再导出，标记 `DEPRECATED`。

### 测试
**测试方案**：`from framework.pipeline import Pipeline` 仍可用且等价于 Edge。**测试方法**：
`pytest tests/test_improvements.py`（含旧 Pipeline 用法回归）。**测试结果**：通过。

---

## 问题 5：`GraphBuilder.vertex()` 用错误 key 存 script

### 问题
旧代码存到 `vc["pipeline"]`，`from_dict()` 读 `vc["script"]` → 自定义 vertex 脚本被静默丢弃。

### 方案
`vertex()` 用 `vc["script"] = script`。

### 修改
- `framework/builders/builder.py`：key 修正（已核实当前为 `vc["script"] = script`）。

### 测试
**测试方案**：builder 注入的 script 子类生效。**测试方法**：`GraphBuilder().vertex("x", script=...).build()`。
**测试结果**：通过（`tests/test_improvements.py`）。

---

## 问题 6：`GraphBuilder.edge()` 残留 `agent` 参数

### 问题
`edge()` 的 `agent` 参数写 `settings["agent"]`，但 `Edge.__init__` 已不再消费该字段
（`self.agent = None`），写进去被静默忽略。

### 方案
删除该参数；`prompt/model` 保留（真实被消费）。

### 修改
- `framework/builders/builder.py`：`edge()` 移除 `agent` 参数与赋值。
- `framework/edge.py` docstring：`agent` 从 parsed 属性列表移除。

### 测试
**测试方案**：builder 不再写 `settings["agent"]`。**测试方法**：`grep "agent" framework/builders/builder.py`。
**测试结果**：0 处；全框架 0 处读取 `settings["agent"]`（`opencode_agent_runner.py` 的 `--agent` 是 CLI 参数，保留）。**355 tests passed**。

---

## 问题 7：Edge 重试时 prompt 跨迭代累积

### 问题
`retry_policy` 反馈原地改 `self.prompt`，循环图上多次迭代堆积多个 `[SYSTEM FEEDBACK]` 块。

### 方案
冻结 `self._base_prompt`；每次重试从它重建 `active_prompt`；执行结束恢复 `self.prompt`。

### 修改
- `framework/edge.py`（commit `121ea9e`）；`tests/test_retry_and_stream.py` 回归。

### 测试
**测试方案**：多次重试/循环后 prompt 不叠加。**测试方法**：断言 feedback 块数 = 1。
**测试结果**：通过（回归锁定）。

---

## 问题 8：`SchemaMismatchError` 声明了却没被 raise

### 问题
自定义异常类存在，但校验处抛通用 `ValueError`。

### 方案
图编译校验改用 `SchemaMismatchError`。

### 修改
- `framework/graph.py`：`raise SchemaMismatchError(...)`（已核实 line 327）。

### 测试
**测试方案**：schema 失败抛 `SchemaMismatchError`。**测试方法**：`pytest tests/test_schema.py`（若存在）/ `test_graph.py`。
**测试结果**：通过；`SchemaMismatchError` 已在 `graph.validate` 处使用。

---

## 问题 9：`SQLiteStateStore` 连接泄漏风险

### 问题
非内存库每次 `_connect()` 新建连接不关闭，连接对象累积。

### 方案
增加 `close()`/`_closed` 状态与上下文管理；防重复/复用后关闭。

### 修改
- `framework/utils/store.py`：`close()`、`_closed` 标志、`__exit__` 调用 close。

### 测试
**测试方案**：关闭后使用报错、重复 close 幂等。**测试方法**：`pytest tests/test_checkpoint.py`。
**测试结果**：通过。

---

## 问题 10：遗留 `exec()/eval()` 代码执行风险

### 问题
旧版 `edge.py` 用 `eval()` 解析 condition、`exec()` 做 transform，存在代码注入面。

### 方案
随重构彻底移除；管线并入 Edge 后改用子类 override + `edge_transform` 函数式工厂，不再执行任意字符串。

### 修改
- `framework/` 全部移除 `exec(/eval(`（已核实 0 处）。

### 测试
**测试方案**：framework 无 `exec/eval`。**测试方法**：`grep -rnE "\bexec\(|\beval\(" framework/`。
**测试结果**：0 处。

---

## 结构建议（已收敛）

| 议题 | 结论 |
|---|---|
| `Vertex._data_store` 单一 asyncio.Lock | 高 fan-in 会串行化；多数场景够用，记录为准 |
| 正式错误层级 | 已有 `AbortPipeline / GuardAbortError / HookError / ComputeError`（`utils/errors.py`），够用 |
| Executor monkey-patch `on_cancel_edges` | 已有 `ExecutorHooks` 回调系统（v3） |
| 每边独立 timeout | 已支持 `settings["timeout"]` |
| Agent 流式/上下文管理 | `stream_process` + async context manager 已实现 |

---

## 结论

核心架构（actor/消息传递、5 段管线、有界循环、checkpoint/HITL、子图、全局内存、telemetry、
schema、race mode）设计优良，评审所列问题均已修复，**355 tests passed**。
剩余风险：分布式执行（ROADMAP v3 #7）尚未开始。