# Vertex-Edge Agent Framework

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()

Vertex-Edge Agent Framework 是一个面向生产环境的高性能、数据驱动型 AI Agent 编排框架。框架基于**两阶段状态握手协议（Two-Sided Handshake Contract）**，采用 **SQLite 事务级持久化**与**反应式事件驱动调度（Event-Driven Scheduler）**，提供确定性的状态流转、闭环故障自愈、多源并发汇聚屏障、声明式 JSON 配置与开箱即用的大模型集成能力。

---

## 目录

- [一、核心概念](#一核心概念)
- [二、环境安装](#二环境安装)
- [三、快速上手](#三快速上手)
  - [1. 方式 A：声明式 JSON 配置驱动（推荐）](#1-方式-a声明式-json-配置驱动推荐)
  - [2. 方式 B：Python 代码直接编排](#2-方式-bpython-代码直接编排)
  - [3. 真实大模型推理管道 (SenseNova)](#3-真实大模型推理管道-sensenova)
- [四、进阶特性使用指南](#四进阶特性使用指南)
  - [1. 多源汇聚与合并策略 (Fan-In Merge)](#1-多源汇聚与合并策略-fan-in-merge)
  - [2. 故障自愈重试与熔断保护 (ReflexiveEdge)](#2-故障自愈重试与熔断保护-reflexiveedge)
  - [3. 层次化嵌套子图 (Hierarchical Subgraph)](#3-层次化嵌套子图-hierarchical-subgraph)
- [五、服务模式：Web 仪表盘与 API 网关](#五服务模式web-仪表盘与-api-网关)
  - [1. 启动 HTTP / SSE 服务](#1-启动-http--sse-服务)
  - [2. Web 可视化仪表盘](#2-web-可视化仪表盘)
  - [3. REST 与 SSE 核心接口调用](#3-rest-与-sse-核心接口调用)
- [六、分布式工作队列扩展 (BaseWorkerQueueV4)](#六分布式工作队列扩展-baseworkerqueuev4)
- [七、自动化测试与验证](#七自动化测试与验证)

---

## 一、核心概念

框架通过**顶点（Vertex）**承载状态与数据，通过**边（Edge）**承载业务逻辑与推理计算。

### 1. 两阶段状态握手协议

每一条边在执行前必须同时满足**输入就绪**与**输出有需求**两重校验：

```
[ 输入节点: data ready ] ───( 边 Edge: 业务计算 / LLM 推理 )───> [ 输出节点: todo -> data ready ]
```

### 2. 顶点状态 (VertexStateV4)

- `data ready`：数据生产完成，下游就绪可消费。
- `todo`：普通需求信号，表示等待上游数据进入。
- `todo urgent`：高优先级需求信号（例如重试唤醒），调度器优先处理。
- `reject`：计算发生异常，触发自愈边介入。
- `forbidden`：超过重试阈值后安全熔断锁死。
- `idle`：空闲等待态。

### 3. 顶点属性标签 (VertexAttributeV4)

- `start`：工作流起始入口点。
- `end`：工作流最终汇聚点。
- `json`：约束节点内容必须为有效 JSON，自动校验并剥离 Markdown 代码围栏。
- `subgraph`：标记该节点为嵌套子图容器。

---

## 二、环境安装

环境要求：**Python 3.12+**

```bash
# 1. 克隆代码仓库
git clone git@github.com:Hana-ame/agent.git
cd agent

# 2. 安装项目依赖
pip install -r requirements.txt

# 3. 安装本地包
pip install -e .
```

---

## 三、快速上手

### 1. 方式 A：声明式 JSON 配置驱动（推荐）

通过 JSON 清单文件解耦拓扑定义与业务脚本：

#### 步骤 1：编写业务脚本 (`workflow/trans.py`)

```python
def to_uppercase(content: str, settings: dict, staging: dict) -> str:
    """将文本转换为大写。"""
    return content.upper()

def add_signature(content: str, settings: dict, staging: dict) -> str:
    """为文本添加署名。"""
    author = settings.get("author", "VEA")
    return f"{content}\n-- Processed by {author}"
```

#### 步骤 2：定义工作流清单 (`workflow/graph.json`)

```json
{
  "version": "4.0",
  "session_id": "demo_pipeline",
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
      "id": "e_sign",
      "input_vertex": "upper_node",
      "output_vertex": "output_node",
      "script": "workflow/trans.py:add_signature",
      "settings": {"author": "VertexEdgeAgent"}
    }
  ]
}
```

#### 步骤 3：加载并执行

```python
import asyncio
from framework import VertexStoreV4, DiscreteGraphLoaderV4, ExecutorV4

async def main():
    # 初始化存储
    store = VertexStoreV4(":memory:")

    # 从 JSON 清单加载图拓扑并同步至存储
    graph = DiscreteGraphLoaderV4.load_from_manifest("workflow/graph.json")
    DiscreteGraphLoaderV4.populate_store(graph, store)

    # 调度执行
    executor = ExecutorV4(graph=graph, store=store, max_concurrency=2)
    result = await executor.run()

    print("执行状态:", result.success)
    final_vertex = store.get_vertex(graph.session_id, "output_node")
    print("最终输出:\n", final_vertex.content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 2. 方式 B：Python 代码直接编排

无需额外文件，直接在代码中定义顶点与边：

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

    # 添加顶点
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="src",
        content="Antigravity",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="dst",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    # 定义转换函数并绑定边
    def reverse_text(data, settings, staging):
        return data[::-1]

    graph.add_edge(CodeEdgeV4("edge_rev", "src", "dst", script=reverse_text))
    graph.validate()

    # 启动调度执行（运行前自动从 store 读取并更新图状态）
    executor = ExecutorV4(graph=graph, store=store)
    result = await executor.run()

    dst_v = store.get_vertex(session_id, "dst")
    print("反转结果:", dst_v.content)  # 输出: ytivargitnA

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 3. 真实大模型推理管道 (SenseNova)

框架开箱即用支持商汤 SenseNova 6.8 远程大模型推理（支持通过环境变量 `SENSENOVA_API_KEY` 注入密钥，未提供时默认使用公共端点）：

```python
import asyncio
import json
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, SenseNovaEdgeV4, CodeEdgeV4
)

async def main():
    store = VertexStoreV4(":memory:")
    session_id = "llm_pipeline_sess"
    graph = GraphV4(session_id=session_id)

    # 1. 提示词起始节点
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="prompt_node",
        content="计算 15 + 25。请严格以 JSON 格式输出: {\"result\": 40}，不要包含任何多余文字或解释。",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))

    # 2. 模型推理输出节点（声明 JSON 校验属性）
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="model_node",
        content="",
        attributes=[VertexAttributeV4.JSON.value],
        state=VertexStateV4.TODO.value
    ))

    # 3. 最终结果节点
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

    def parse_and_format(content, settings, staging):
        data = json.loads(content)
        return f"计算得到结果: {data['result']}"

    graph.add_edge(CodeEdgeV4("e_format", "model_node", "final_node", script=parse_and_format))
    graph.validate()

    executor = ExecutorV4(graph=graph, store=store)
    res = await executor.run()

    print("执行结果:", res.success)
    print("终点内容:", store.get_vertex(session_id, "final_node").content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 四、进阶特性使用指南

### 1. 多源汇聚与合并策略 (Fan-In Merge)

当多个上游边同时写入同一个下游节点时，调度器内置**汇聚屏障保护**，确保所有上游依赖边均结算完成后方才唤醒下游。提供 4 种汇聚合并策略：

```python
from framework import MergeStrategyV4

# 1. OVERWRITE：直接覆盖（后到者胜，默认）
store.apply_merge_strategy(session_id, "target_node", incoming_str, strategy=MergeStrategyV4.OVERWRITE)

# 2. JSON_MERGE：合并 JSON 对象字典 {**existing, **incoming}
store.apply_merge_strategy(session_id, "target_node", '{"score_b": 95}', strategy=MergeStrategyV4.JSON_MERGE)

# 3. LIST_APPEND：追加为 JSON 列表 [item1, item2, ...]
store.apply_merge_strategy(session_id, "target_node", '"log_entry"', strategy=MergeStrategyV4.LIST_APPEND)

# 4. REDUCER_SCRIPT：自定义归约合并函数
def custom_reducer(existing_content: str, incoming_content: str) -> str:
    return f"{existing_content}\n{incoming_content}".strip()

store.apply_merge_strategy(session_id, "target_node", "new line", strategy=MergeStrategyV4.REDUCER_SCRIPT, reducer_fn=custom_reducer)
```

---

### 2. 故障自愈重试与熔断保护 (ReflexiveEdge)

当某个计算节点因网络波动或异常代码崩溃时，框架自动将节点标红置为 `REJECT` 状态。`ReflexiveEdgeV4` 自环边侦测到该状态后，自动介入修复并重置为 `TODO_URGENT` 发起自愈重试：

```python
from framework import ReflexiveEdgeV4, VertexStateV4

def recovery_script(error_content, settings, staging):
    # 可在此处修正输入、剥离错误字符或添加重试提示词
    return f"{error_content} (请简化回答并严格遵循格式)"

reflexive_edge = ReflexiveEdgeV4(
    edge_id="e_retry_agent",
    vertex_name="worker_node",
    trigger_state=VertexStateV4.REJECT.value,
    target_state=VertexStateV4.TODO_URGENT.value,
    max_retries=3,
    script=recovery_script
)
```

- 若连续重试次数达到 `max_retries`，节点状态变更为 `FORBIDDEN`，调度器安全熔断终止无效循环。

---

### 3. 层次化嵌套子图 (Hierarchical Subgraph)

支持多层级子图的嵌套与桥接：
1. 为父图中的节点添加 `VertexAttributeV4.SUBGRAPH` 属性。
2. 在该节点的 `content` 中传入包含 `subgraph_manifest` 路径的 JSON 字符串。
3. `SSEExecutorV4` 会自动递归解析子图拓扑，自动挂接输入输出代理桥接边，实现多级子图的端到端数据流动。

---

## 五、服务模式：Web 仪表盘与 API 网关

### 1. 启动 HTTP / SSE 服务

```bash
# 启动服务端口（默认 8000 端口）
uvicorn framework.server_v4:app --host 0.0.0.0 --port 8000
```

或在 Python 脚本中内嵌拉起：

```python
import uvicorn
from framework.vertex_v4 import VertexStoreV4
from framework.server_v4 import create_v4_server

store = VertexStoreV4("workflow.db")
app = create_v4_server(store=store)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

### 2. Web 可视化仪表盘

打开浏览器访问：`http://localhost:8000/dashboard`

- **可视化拓扑**：实时展示节点网络与数据流向。
- **状态高亮**：
  - 🟢 绿色：`DATA_READY`（已完成）
  - 🟡 黄色：`TODO`（待执行）
  - 🔵 蓝色：`TODO_URGENT`（高优先级执行）
  - 🔴 红色：`REJECT`（异常失败）
  - ⚫ 灰色：`FORBIDDEN`（熔断锁定）
- **数据检测器**：点击任意节点可实时查看该节点最新持久化内容、执行计数及元数据。

### 3. REST 与 SSE 核心接口调用

#### 执行指定会话工作流
```bash
curl -X POST http://localhost:8000/api/sessions/demo_session/run \
     -H "Content-Type: application/json" \
     -d '{"max_concurrency": 4}'
```

#### SSE 事件流式执行
```bash
curl -N http://localhost:8000/api/sse/execute \
     -H "Content-Type: application/json" \
     -d '{"session_id": "demo_session", "manifest_path": "workflow/graph.json"}'
```

#### 动态新增/更新顶点
```bash
curl -X POST http://localhost:8000/api/sessions/demo_session/graph/vertices \
     -H "Content-Type: application/json" \
     -d '{"name": "node_new", "content": "hello", "state": "data ready"}'
```

#### 动态新增边
```bash
curl -X POST http://localhost:8000/api/sessions/demo_session/graph/edges \
     -H "Content-Type: application/json" \
     -d '{"id": "e_dynamic", "type": "code", "input_vertex": "node_new", "output_vertex": "node_target"}'
```

---

## 六、分布式工作队列扩展 (BaseWorkerQueueV4)

框架提供抽象队列适配层 [`BaseWorkerQueueV4`](framework/worker_queue_v4.py)，用于将边计算任务卸载至分布式 Worker 集群（如 Celery、Redis Queue、Ray）：

```python
from framework.worker_queue_v4 import BaseWorkerQueueV4, EdgeTaskPayload, EdgeTaskResult

class DistributedQueue(BaseWorkerQueueV4):
    async def submit_edge(self, task: EdgeTaskPayload) -> str:
        # 将 Edge 任务推入分布式队列并返回 task_id
        ...

    async def poll_result(self, task_id: str):
        # 轮询任务完成状态
        ...

    async def wait_result(self, task_id: str, timeout: float = 120.0) -> EdgeTaskResult:
        # 等待计算结果返回
        ...

    async def cancel_task(self, task_id: str) -> bool:
        # 取消正在排队或执行的任务
        ...

    async def health_check(self) -> bool:
        # 队列健康度探测
        return True
```

---

## 七、自动化测试与验证

框架测试套件包含 520+ 项自动化测试用例，覆盖状态机流转、事件调度、并发控制、自愈容错及 E2E 流水线：

```bash
# 运行离线测试套件（排除外部网络依赖）
python -m pytest tests/ -v -m "not live"

# 运行全量测试套件（包含 SenseNova 实时端点联调）
python -m pytest tests/ -v
```

GitHub Actions CI 在 [.github/workflows/ci.yml](.github/workflows/ci.yml) 中配置，保障代码提交自动验证通过。
