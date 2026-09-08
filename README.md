# Vertex-Edge Agent Framework

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()

Vertex-Edge Agent Framework 是一个面向生产环境的高性能、数据驱动型 AI Agent 编排框架。框架基于**两阶段状态握手协议（Two-Sided Handshake Contract）**，结合 **SQLite 事务级持久化**与**反应式事件驱动调度（Event-Driven Scheduler）**，提供确定性的状态流转、闭环自愈重试、多源并发汇聚屏障、声明式 JSON 配置与开箱即用的大模型集成能力。

---

## 目录

- [一、核心特性](#一核心特性)
- [二、架构概览：V4.0 vs 旧版本](#二架构概览v40-vs-旧版本)
- [三、环境安装与快速上手](#三环境安装与快速上手)
  - [1. 安装与依赖](#1-安装与依赖)
  - [2. 方式 A：基于声明式 JSON 配置驱动（推荐）](#2-方式-a基于声明式-json-配置驱动推荐)
  - [3. 方式 B：基于 Python 代码编排](#3-方式-b基于-python-代码编排)
  - [4. 真实大模型推理编排 (SenseNovaEdgeV4)](#4-真实大模型推理编排-sensenovaedgev4)
- [四、进阶特性与最佳实践](#四进阶特性与最佳实践)
  - [1. 多源汇聚与合并策略 (Fan-In Barrier)](#1-多源汇聚与合并策略-fan-in-barrier)
  - [2. 故障自愈重试与熔断保护 (ReflexiveEdgeV4)](#2-故障自愈重试与熔断保护-reflexiveedgev4)
  - [3. 层次化嵌套子图 (Hierarchical Subgraph)](#3-层次化嵌套子图-hierarchical-subgraph)
- [五、服务模式：Web 仪表盘与 API 网关](#五服务模式web-仪表盘与-api-网关)
  - [1. 启动 HTTP / SSE 服务](#1-启动-http--sse-服务)
  - [2. Web 可视化仪表盘](#2-web-可视化仪表盘)
  - [3. REST 与 SSE 核心接口](#3-rest-与-sse-核心接口)
- [六、分布式工作队列扩展 (BaseWorkerQueueV4)](#六分布式工作队列扩展-baseworkerqueuev4)
- [七、旧版本 (Legacy V1-V3) 兼容指南](#七旧版本-legacy-v1-v3-兼容指南)
- [八、测试与持续集成](#八测试与持续集成)
- [九、文档中心与历史归档](#九文档中心与历史归档)

---

## 一、核心特性

- 🔒 **两阶段状态握手合约**：执行前严格校验输入节点的就绪态 (`DATA_READY`) 与输出节点的需求态 (`TODO` / `TODO_URGENT`)，保障数据流拓扑与执行顺序完全确定。
- 💾 **全持久化执行与无脏内存**：执行调度前从 SQLite 存储层实时拉取更新最新图拓扑与节点状态，所有顶点增删改查及图操作直接入库，规避多并发与长生命周期中的内存脏状态。
- ⚡ **反应式事件驱动调度**：使用 `asyncio.Event` 毫秒级唤醒，结合动态信号量与拓扑分层（DAG Tiers），消除无谓的轮询开销与高并发死锁。
- 📜 **配置驱动与解耦设计**：边（Edge）的构建与解析完全支持通过 JSON 声明式配置驱动，核心框架不硬编码业务边类型。
- 🛡️ **声明式自愈循环与熔断锁死**：内置 `ReflexiveEdgeV4`，在节点发生计算异常并标红置为 `REJECT` 时自动触发介入、自愈重置并累加重试计数；超过重试上限后安全熔断置为 `FORBIDDEN`。
- 🌐 **开箱即用的大模型集成**：内置 `SenseNovaEdgeV4` 直连商汤 SenseNova 6.8 远程推理端点，支持 JSON 严格校验与 Markdown 围栏剥离。
- 📊 **可视化 Web 仪表盘与 API 网关**：内置独立模板的 Web 控制台与完整的 FastAPI REST / SSE 接口，提供实时拓扑呈现与状态观测。

---

## 二、架构概览：V4.0 vs 旧版本

| 维度 | 旧版本 (Legacy V1-V3) | 新版本 (V4.0) |
| :--- | :--- | :--- |
| **推荐状态** | 兼容保留（用于老旧脚本向后兼容） | 🚀 **高度推荐，生产就绪 (Production-Ready)** |
| **存储层** | 纯内存 Python 字典对象（进程退出即丢） | SQLite 事务持久化（支持 WAL 模式与故障重载） |
| **调度引擎** | 定时 Sleep 轮询检测 | `asyncio.Event` 反应式事件通知 + 并发信号量 |
| **数据流汇聚** | 简单覆盖写入，无屏障同步 | `MergeStrategyV4`（字典合并/列表追加/自定义函数）+ 汇聚屏障 |
| **自愈机制** | 需用户外挂 try-except 逻辑 | `ReflexiveEdgeV4` 声明式自愈重试与防死锁断路器 |
| **交互模式** | 仅支持本地 Python 脚本调用 | 包含完整 REST API、SSE 流式管道、独立 Web 控制台 |

---

## 三、环境安装与快速上手

### 1. 安装与依赖

环境要求：Python 3.12+

```bash
# 克隆代码仓库
git clone git@github.com:Hana-ame/agent.git
cd agent

# 安装依赖与当前包
pip install -r requirements.txt
pip install -e .
```

---

### 2. 方式 A：基于声明式 JSON 配置驱动（推荐）

通过主清单 (`graph.json`) 与离散边/节点配置定义工作流：

**主清单定义 (`workflow/graph.json`)：**

```json
{
  "version": "4.0",
  "session_id": "demo_session",
  "metadata": {"name": "Text Processing Pipeline"},
  "vertices": [
    {"name": "input_node", "content": "hello world", "attributes": ["start"], "state": "data ready"},
    {"name": "upper_node", "content": "", "state": "todo"},
    {"name": "output_node", "content": "", "attributes": ["end"], "state": "todo"}
  ],
  "edges": [
    {
      "id": "e_upper",
      "input_vertex": "input_node",
      "output_vertex": "upper_node",
      "script": "workflow/trans.py:to_uppercase"
    },
    {
      "id": "e_tag",
      "input_vertex": "upper_node",
      "output_vertex": "output_node",
      "script": "workflow/trans.py:add_footer"
    }
  ]
}
```

**业务转换脚本 (`workflow/trans.py`)：**

```python
def to_uppercase(content: str, settings: dict, staging: dict) -> str:
    return content.upper()

def add_footer(content: str, settings: dict, staging: dict) -> str:
    return f"{content}\n-- processed by VEA v4"
```

**加载并执行工作流：**

```python
import asyncio
from framework import VertexStoreV4, DiscreteGraphLoaderV4, ExecutorV4

async def main():
    store = VertexStoreV4(":memory:")
    # 从 JSON 清单加载完整拓扑
    graph = DiscreteGraphLoaderV4.load_from_manifest("workflow/graph.json")
    DiscreteGraphLoaderV4.populate_store(graph, store)

    # 调度执行
    executor = ExecutorV4(graph=graph, store=store, max_concurrency=2)
    res = await executor.run()

    print("执行成功:", res.success)
    final_node = store.get_vertex(graph.session_id, "output_node")
    print("输出内容:\n", final_node.content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 3. 方式 B：基于 Python 代码编排

直接在代码中通过数据类动态装配节点与边：

```python
import asyncio
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, CodeEdgeV4
)

async def main():
    store = VertexStoreV4(":memory:")
    session_id = "code_pipeline_sess"
    graph = GraphV4(session_id=session_id)

    # 添加起点与终点节点
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="src",
        content="Antigravity Agent",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="dst",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    # 添加处理边
    def reverse_text(data, settings, staging):
        return data[::-1]

    graph.add_edge(CodeEdgeV4("edge_rev", "src", "dst", script=reverse_text))
    graph.validate()

    # 执行引擎将在运行前自动将图状态同步并重载自 SQLite
    executor = ExecutorV4(graph=graph, store=store)
    result = await executor.run()

    dst_v = store.get_vertex(session_id, "dst")
    print("结果:", dst_v.content)  # 输出: tnegA ytivargitnA

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 4. 真实大模型推理编排 (SenseNovaEdgeV4)

内置商汤 SenseNova 6.8 开箱即用支持，支持自动剔除 Markdown 代码块标记并严格验证 JSON 输出：

```python
import asyncio
import json
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, SenseNovaEdgeV4, CodeEdgeV4
)

async def main():
    store = VertexStoreV4(":memory:")
    session_id = "sensenova_demo"
    graph = GraphV4(session_id=session_id)

    # 1. 提示词节点
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="prompt_node",
        content="计算 15 + 25。请严格以 JSON 格式输出: {\"result\": 40}，不要包含任何额外文字或解释。",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))

    # 2. 模型输出节点（带 JSON 校验属性）
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="model_node",
        content="",
        attributes=[VertexAttributeV4.JSON.value],
        state=VertexStateV4.TODO.value
    ))

    # 3. 业务消费节点
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="final_node",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    # 4. 连接 SenseNova 边与后处理计算边
    graph.add_edge(SenseNovaEdgeV4(
        edge_id="e_llm",
        input_vertex="prompt_node",
        output_vertex="model_node",
        settings={"temperature": 0.1}
    ))

    def process_result(content, settings, staging):
        data = json.loads(content)
        return f"计算结果为: {data['result']}"

    graph.add_edge(CodeEdgeV4("e_calc", "model_node", "final_node", script=process_result))
    graph.validate()

    executor = ExecutorV4(graph=graph, store=store)
    await executor.run()

    print("终点内容:", store.get_vertex(session_id, "final_node").content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 四、进阶特性与最佳实践

### 1. 多源汇聚与合并策略 (Fan-In Barrier)

当多个上游边同时写入同一个下游节点时，框架提供**汇聚屏障保护**，确保所有上游非自环入边全部执行结算后才唤醒下游。同时支持 4 种汇聚策略：

```python
from framework import MergeStrategyV4

# 1. OVERWRITE（覆盖，默认）
store.apply_merge_strategy(sess, "node", incoming_content, strategy=MergeStrategyV4.OVERWRITE)

# 2. JSON_MERGE（合并 JSON 对象: {**existing, **incoming}）
store.apply_merge_strategy(sess, "node", '{"b": 2}', strategy=MergeStrategyV4.JSON_MERGE)

# 3. LIST_APPEND（追加成 JSON 列表: [item1, item2, ...]）
store.apply_merge_strategy(sess, "node", '"new_item"', strategy=MergeStrategyV4.LIST_APPEND)

# 4. REDUCER_SCRIPT（自定义归约函数）
def custom_sum(old_val, new_val):
    return int(old_val or 0) + int(new_val)

store.apply_merge_strategy(sess, "node", "10", strategy=MergeStrategyV4.REDUCER_SCRIPT, reducer_fn=custom_sum)
```

---

### 2. 故障自愈重试与熔断保护 (ReflexiveEdgeV4)

当某个计算节点执行抛出异常时，节点状态自动转为 `REJECT`。此时 `ReflexiveEdgeV4` 自环边响应，并在重试次数限额内自动重置为 `TODO_URGENT` 发起自愈重试：

```python
from framework import ReflexiveEdgeV4, VertexStateV4

def recovery_modifier(old_content, settings, staging):
    # 可在重试前修正提示词或清理中间脏数据
    return f"{old_content} (请简化回答)"

reflexive_edge = ReflexiveEdgeV4(
    edge_id="e_retry_worker",
    vertex_name="worker_node",
    trigger_state=VertexStateV4.REJECT.value,
    target_state=VertexStateV4.TODO_URGENT.value,
    max_retries=3,
    script=recovery_modifier
)
```

若重试超过 `max_retries`，节点被安全锁死在 `FORBIDDEN` 状态，调度器停止对该节点的无效重试，保护系统资源。

---

### 3. 层次化嵌套子图 (Hierarchical Subgraph)

通过给节点赋予 `VertexAttributeV4.SUBGRAPH` 属性，并在 `content` 中声明子图清单路径，`SSEExecutorV4` 会自动递归将子图展开并接入代理桥接边（Bridge Edge），支持三层或更深层次的递归组合执行。

---

## 五、服务模式：Web 仪表盘与 API 网关

### 1. 启动 HTTP / SSE 服务

```bash
# 启动 V4 服务端口（默认 8000 端口）
uvicorn framework.server_v4:app --host 0.0.0.0 --port 8000
```

或在 Python 中以内嵌形式拉起：

```python
import uvicorn
from framework.vertex_v4 import VertexStoreV4
from framework.server_v4 import create_v4_server

store = VertexStoreV4("agents_workflow.db")
app = create_v4_server(store=store)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

### 2. Web 可视化仪表盘

打开浏览器访问：`http://localhost:8000/dashboard`

- 实时查看当前会话的图拓扑、节点颜色（绿: DATA_READY / 黄: TODO / 蓝: TODO_URGENT / 红: REJECT / 灰: FORBIDDEN）。
- 实时检查顶点内容、历史更新时间与执行计数。
- 在线重入测试与调试。

### 3. REST 与 SSE 核心接口

- **执行会话工作流**：
  ```bash
  curl -X POST http://localhost:8000/api/sessions/{session_id}/run \
       -H "Content-Type: application/json" \
       -d '{"max_concurrency": 4}'
  ```
- **SSE 流式执行**：
  ```bash
  curl -N http://localhost:8000/api/sse/execute \
       -H "Content-Type: application/json" \
       -d '{"session_id": "test_sess", "manifest_path": "examples/simple/graph.json"}'
  ```
- **动态更新顶点**：
  ```bash
  curl -X POST http://localhost:8000/api/sessions/{session_id}/graph/vertices \
       -H "Content-Type: application/json" \
       -d '{"name": "node_a", "content": "new payload", "state": "data ready"}'
  ```
- **动态创建边**：
  ```bash
  curl -X POST http://localhost:8000/api/sessions/{session_id}/graph/edges \
       -H "Content-Type: application/json" \
       -d '{"id": "e_new", "type": "code", "input_vertex": "node_a", "output_vertex": "node_b"}'
  ```

---

## 六、分布式工作队列扩展 (BaseWorkerQueueV4)

针对多机、多进程边缘计算扩展需求，框架提供了抽象队列适配接口 [`BaseWorkerQueueV4`](framework/worker_queue_v4.py)：

```python
from framework.worker_queue_v4 import BaseWorkerQueueV4, EdgeTaskPayload, EdgeTaskResult

class RedisWorkerQueue(BaseWorkerQueueV4):
    async def submit_edge(self, task: EdgeTaskPayload) -> str:
        # 将任务投递至 Redis 任务队列
        ...

    async def wait_result(self, task_id: str, timeout: float = 120.0) -> EdgeTaskResult:
        # 等待远端 Worker 消费完成
        ...
```

内置用于单元测试与单机模拟的 `InMemoryWorkerQueueV4`，完整测试覆盖异步任务提交、等待、取消与健康检查。

---

## 七、旧版本 (Legacy V1-V3) 兼容指南

若您需要维护早期的工作流脚本，框架在根命名空间完整保留了 Legacy 相关的组件：

```python
# 旧版纯内存流程式编排
import asyncio
from framework import GraphBuilder, Executor

g = (GraphBuilder("legacy_pipeline")
     .vertex("input", initial_data=[{"channel": "text", "value": "legacy"}])
     .vertex("output")
     .edge("input", "output")
     .build())

result = asyncio.run(Executor(g).run())
```

> 详细的旧版特性说明与迁移对比，请查阅完整使用手册：[USAGE_GUIDE.md](USAGE_GUIDE.md#六旧版本-legacy-v1-v3-兼容使用指南)。

---

## 八、测试与持续集成

项目配有覆盖层级完善的自动化测试套件（520+ 测试用例）：

```bash
# 执行单元与集成测试（排除需外部网络的 live 测试）
python -m pytest tests/ -v -m "not live"

# 执行完整端到端测试（需联网）
python -m pytest tests/ -v
```

CI 工作流在 [.github/workflows/ci.yml](.github/workflows/ci.yml) 中配置，每次分支推送均自动运行 Python 3.12 环境全量检验。

---

## 九、文档中心与历史归档

- 📖 **[USAGE_GUIDE.md](USAGE_GUIDE.md)**：开发人员完整实战指南。
- 📚 **[docs/README.md](docs/README.md)**：项目文档中心导航。
- 📦 **[docs/archive/v4-runtime-issues-analysis.md](docs/archive/v4-runtime-issues-analysis.md)**：运行时问题（死锁/会话锁/并发重入）分析与加固报告归档。
- 📦 **[docs/archive/legacy-issues-log.md](docs/archive/legacy-issues-log.md)**：早期架构缺陷与优化记录归档（Issue 1 至 Issue 11）。
- 🗺️ **[ROADMAP.md](ROADMAP.md)**：演进技术路线图。
