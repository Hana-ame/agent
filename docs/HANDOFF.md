# Vertex-Edge Agent Framework V4.0 交付与交接文档 (Handoff)

## 1. 文档概述 (Executive Summary)

本文档为 **Vertex-Edge Agent Framework V4.0** 的系统交付与交接说明书。
当前代码仓库已全面完成 V4.0 架构重构、安全性加固、分层调度、多源汇聚（Fan-In）、分布式工作节点队列接口、嵌套子图（Subgraph）以及离散 JSON / 文件夹批量加载特性的开发与验证。

- **当前分支**：`vertex-edge-agent`
- **运行环境**：`Python 3.12+` / Linux
- **自动化测试状态**：**526 项测试全部通过**（包括系统级 E2E 测试、安全防御测试、执行器调度并发测试、子图集成测试及历史回归测试），CI 验证均为绿色。

---

## 2. 核心架构与设计原则

V4 架构旨在构建确定性、高吞吐、生产级可靠的数据驱动型 AI Agent 编排框架：

```
+-------------------------------------------------------------------------+
|                        FastAPI Gateway / WebUI                          |
|    - 独立会话隔离 (SessionGraphManagerV4)                                |
|    - 路径穿越防护 / JSON 扩展名白名单校验                                |
|    - SSE 流式推送 & OpenAI Tool-Call Echo 规范                           |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                          Execution Layer                                |
|    - SSEExecutorV4: 动态子图解析、递归桥接边挂载 (Proxy Bridge)         |
|    - ExecutorV4: 事件驱动唤醒、拓扑分层调度、优先级队列                 |
|    - 并发控制: 信号量与并发组隔离、在途任务内存去重 (Active Dispatches)  |
|    - BaseWorkerQueueV4: 分布式工作节点适配器抽象接口                     |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                           Data & Store Layer                            |
|    - VertexStoreV4: SQLite 持久化、严格两阶段握手协议                   |
|    - MergeStrategyV4: 多源汇聚 (Overwrite / JSON Merge / List Append)    |
|    - 自愈与容错: 异常拒绝 (REJECT) -> 自环重试 (Reflexive) -> 熔断锁定  |
+-------------------------------------------------------------------------+
```

### 关键设计原则：
1. **持久化顶点状态机**：所有 Vertex 状态持久化于 SQLite（`data ready`、`idle`、`todo`、`todo urgent`、`reject`、`forbidden`、`pruning`），内存不持有长驻运行态，具备抗崩溃自恢复能力。
2. **两阶段握手合约 (Two-Sided Handshake)**：前向边仅在“上游为 `data ready` 且下游为 `todo` 或 `todo urgent`”时触发；执行完毕后原子更新下游内容并流转状态。
3. **自环容错自愈与熔断 (Reflexive Self-Healing)**：节点执行失败置为 `reject`，自环边响应并读取故障上下文执行恢复逻辑，累计重试次数超限自动置为 `forbidden` 熔断。
4. **事件驱动与优先级分层**：执行器废弃纯轮询，采用 `asyncio.Event` 结合首任务完成唤醒机制，按“自环自愈 (Priority 0) > 紧急任务 (Priority 1) > 常规任务 (Priority 2)”结合 DAG Topological Tiers 调度。
5. **多源汇聚屏障 (Fan-In Barrier)**：支持多个上游边同时汇入同一顶点，通过汇聚屏障与合并策略（覆盖、字典合并、列表追加、自定义归约函数）保证状态更新原子性。
6. **无缝嵌套子图 (Subgraph)**：包含 `subgraph` 属性的容器节点由 `SSEExecutorV4` 递归解析，自动构建代理输出节点与桥接边，实现跨层级数据流动。
7. **灵活的配置加载**：支持 Master Manifest JSON、单文件 JSON 路径加载、文件夹一键批量加载（支持分目录结构与扁平结构）。

---

## 3. 核心模块与文件清单

### 核心框架模块 (`framework/`)
- [`framework/vertex_v4.py`](file:///home/luminovoez/agent/framework/vertex_v4.py)：SQLite 存储引擎，定义 `VertexStoreV4`、`VertexRecordV4`、`VertexStateV4`、`VertexAttributeV4` 与 `MergeStrategyV4`。
- [`framework/edge_v4.py`](file:///home/luminovoez/agent/framework/edge_v4.py)：边基类 `EdgeV4`，以及 `CodeEdgeV4`、`LLMEdgeV4`、`ReflexiveEdgeV4` 实现，内置握手校验与灵活的入参调用适配。
- [`framework/sensenova_edge_v4.py`](file:///home/luminovoez/agent/framework/sensenova_edge_v4.py)：商汤 SenseNova 6.8 Flash Lite 大模型原生直连边。
- [`framework/graph_v4.py`](file:///home/luminovoez/agent/framework/graph_v4.py)：`GraphV4` 图定义与拓扑分析（Kahn DAG 分层），`DiscreteGraphLoaderV4`（支持离散 JSON 路径、清单以及文件夹批量加载）。
- [`framework/executor_v4.py`](file:///home/luminovoez/agent/framework/executor_v4.py)：DAG 分层事件驱动调度器，支持多源汇聚屏障与并发隔离。
- [`framework/sse_executor_v4.py`](file:///home/luminovoez/agent/framework/sse_executor_v4.py)：会话路由器、递归子图解析与桥接、OpenAI Tool-Call Echo SSE 流式适配。
- [`framework/worker_queue_v4.py`](file:///home/luminovoez/agent/framework/worker_queue_v4.py)：分布式工作节点抽象接口 `BaseWorkerQueueV4` 及内存参考实现 `InMemoryWorkerQueueV4`。
- [`framework/server_v4.py`](file:///home/luminovoez/agent/framework/server_v4.py)：FastAPI 生产服务、会话图管理 `SessionGraphManagerV4`、REST API、路径穿越安全防御。
- [`framework/templates/dashboard.html`](file:///home/luminovoez/agent/framework/templates/dashboard.html)：Web 可视化拓扑与状态监控仪表盘。

### 示例工程 (`examples/`)
- [`examples/subgraph_v4/`](file:///home/luminovoez/agent/examples/subgraph_v4/)：完整且自包含的 Subgraph 嵌套子图、离散 JSON 配置与文件夹批量加载端到端示例。
  - [`examples/subgraph_v4/run.py`](file:///home/luminovoez/agent/examples/subgraph_v4/run.py)：包含 3 种加载与执行方法的演示脚本。
  - [`examples/subgraph_v4/child/`](file:///home/luminovoez/agent/examples/subgraph_v4/child/)：子图配置、文本清洗与数据丰富处理脚本。
  - [`examples/subgraph_v4/discrete_dir_demo/`](file:///home/luminovoez/agent/examples/subgraph_v4/discrete_dir_demo/)：分目录批量加载演示。
- [`examples/sensenova_v4/`](file:///home/luminovoez/agent/examples/sensenova_v4/)：SenseNova 大模型边直连示例。

### 测试套件 (`tests/`)
- [`tests/test_v4_subgraph_and_discrete_loader.py`](file:///home/luminovoez/agent/tests/test_v4_subgraph_and_discrete_loader.py)：文件路径加载、文件夹加载、子图端到端运行集成测试。
- [`tests/test_v4_layer4_executor.py`](file:///home/luminovoez/agent/tests/test_v4_layer4_executor.py)：并发控制、优先级调度、Fan-In 汇聚屏障测试。
- [`tests/test_v4_layer5_server.py`](file:///home/luminovoez/agent/tests/test_v4_layer5_server.py)：网关路径穿越攻击防护、Pydantic 输入校验、JSON 扩展名白名单测试。
- [`tests/test_v4_layer6_e2e.py`](file:///home/luminovoez/agent/tests/test_v4_layer6_e2e.py)：多步全链路、深层递归子图、分布式 Worker Queue 接口测试。
- [`tests/test_v4_system.py`](file:///home/luminovoez/agent/tests/test_v4_system.py)：系统核心链路与自环恢复测试。
- [`tests/test_v4_server.py`](file:///home/luminovoez/agent/tests/test_v4_server.py)：API 路由、检查器与 SSE 端点测试。

---

## 4. 关键特性与典型用法

### 4.1 离散 JSON 配置文件与文件夹加载
```python
from framework import GraphV4

graph = GraphV4(session_id="my_session")

# 1. 直接传入 JSON 配置文件路径加载
graph.add_vertex("examples/subgraph_v4/parent_in.json")
graph.add_edge("examples/subgraph_v4/e_start_to_subgraph.json")

# 2. 从文件夹一键批量加载
dir_graph = GraphV4.from_directory("examples/subgraph_v4/discrete_dir_demo")
```

### 4.2 嵌套子图 (Subgraph) 编排
父图中声明包含 `subgraph` 属性的节点：
```json
{
  "name": "text_subgraph_processor",
  "attributes": ["subgraph"],
  "state": "todo",
  "content": {
    "subgraph_manifest": "child/child_graph.json"
  }
}
```
执行时由 `SSEExecutorV4` 自动解析并挂接桥接边：
```python
from framework import SSEExecutorV4, SessionGraphManagerV4, VertexStoreV4

store = VertexStoreV4(":memory:")
manager = SessionGraphManagerV4(store)
executor = SSEExecutorV4(manager=manager, store=store, default_manifest="parent_graph.json")
result = await executor.execute_harness_call()
```

### 4.3 多源汇聚 (Fan-In Merge)
```python
from framework import MergeStrategyV4

# 支持 OVERWRITE、JSON_MERGE、LIST_APPEND 与 REDUCER_SCRIPT
store.apply_merge_strategy(session_id, "output_node", incoming_json, strategy=MergeStrategyV4.JSON_MERGE)
```

---

## 5. 常用运维与验证命令

### 运行全套自动化测试
```bash
# 运行离线测试套件 (526 passed)
.venv/bin/python -m pytest tests/ -v -m "not live"

# 针对离散加载与子图的独立测试
.venv/bin/python -m pytest tests/test_v4_subgraph_and_discrete_loader.py -v
```

### 运行可执行演示示例
```bash
# 运行子图与离散配置加载示例
python3 examples/subgraph_v4/run.py

# 运行 SenseNova 大模型示例 (需设置 SENSENOVA_API_KEY)
python3 examples/sensenova_v4/demo.py
```

### 启动服务与访问仪表盘
```bash
uvicorn framework.server_v4:app --host 0.0.0.0 --port 8000
# 浏览器访问: http://localhost:8000/dashboard
```

---

## 6. 后续维护建议 (Next Steps)

1. **分布式队列落地**：如需接入 Celery 或 Redis Queue，直接继承并实现 [`framework/worker_queue_v4.py`](file:///home/luminovoez/agent/framework/worker_queue_v4.py) 中的 `BaseWorkerQueueV4` 抽象接口。
2. **文档规范守则**：
   - 根目录 `README.md` 保持作为面向开发者的**纯使用手册**，不包含任何设计对比或历史长篇大论。
   - 所有历史架构演进草案、缺陷排查报告已归档在 [`docs/archive/`](file:///home/luminovoez/agent/docs/archive/) 统一维护。
   - 严格避免使用任何禁止词汇，统一称呼为“代码仓库”或“repo”。
