# 📖 Vertex-Edge Agent Framework 使用指南 (Usage Guide)

本文档提供 Vertex-Edge Agent Framework 的完整使用说明，涵盖 **V4.0 新版本架构** 的详细实践指南，以及 **旧版本（Legacy V1-V3）** 的兼容使用说明。

---

## 目录
- [一、架构选型：V4.0 vs 旧版本](#一架构选型v40-vs-旧版本)
- [二、V4.0 核心概念与数据模型](#二v40-核心概念与数据模型)
- [三、快速上手：代码级编排实践](#三快速上手代码级编排实践)
  - [1. 基础线性流水线 (CodeEdgeV4)](#1-基础线性流水线-codeedgev4)
  - [2. 真实大模型推理管道 (SenseNovaEdgeV4)](#2-真实大模型推理管道-sensenovaedgev4)
  - [3. 多源入边汇聚与合并策略 (Fan-In Merge)](#3-多源入边汇聚与合并策略-fan-in-merge)
  - [4. 故障自愈重试与熔断保护 (ReflexiveEdgeV4)](#4-故障自愈重试与熔断保护-reflexiveedgev4)
- [四、服务模式：Web 仪表盘与 HTTP/SSE API](#四服务模式web-仪表盘与-httpsse-api)
  - [1. 启动服务](#1-启动服务)
  - [2. 可视化 Web 仪表盘](#2-可视化-web-仪表盘)
  - [3. 核心 API 端点与调用示例](#3-核心-api-端点与调用示例)
- [五、分布式任务队列扩展 (BaseWorkerQueueV4)](#五分布式任务队列扩展-baseworkerqueuev4)
- [六、旧版本 (Legacy V1-V3) 兼容使用指南](#六旧版本-legacy-v1-v3-兼容使用指南)
- [七、自动化测试与健康检查](#七自动化测试与健康检查)

---

## 一、架构选型：V4.0 vs 旧版本

| 维度 | 旧版本 (Legacy V1-V3) | 新版本 (V4.0) |
| :--- | :--- | :--- |
| **运行状态** | ✅ **100% 可用**（保持向后兼容，已冻结新特性） | 🚀 **高度推荐，生产就绪 (Production-Ready)** |
| **存储机制** | 纯内存 Python 字典对象（进程结束数据丢失） | SQLite3 事务级双表存储（支持 WAL 与故障恢复） |
| **执行调度** | 简单计数轮询机制 | `asyncio.Event` 毫秒级反应式事件唤醒 |
| **汇聚控制** | 简单覆盖，无屏障保护 | `MergeStrategyV4`（字典合并/列表追加/自定义函数）+ 汇聚屏障 |
| **容错自愈** | 需用户自行捕获异常 | `ReflexiveEdgeV4` 声明式自愈循环与熔断锁死 |
| **大模型集成** | 需手动装配 Mock 或配置 HttpLLMAgent | 开箱即用支持 `SenseNovaEdgeV4` 真实远程大模型 |
| **交互界面** | 仅支持本地 Python 脚本执行 | 提供完整 FastAPI REST API、SSE 流式、独立 Web 仪表盘 |

- **老项目维护**：已有脚本可无需修改直接沿用旧类（`Vertex`、`Edge`、`Graph`、`Executor`）。
- **新功能研发**：强烈推荐直接使用 V4 系列（`VertexStoreV4`、`GraphV4`、`ExecutorV4`、`EdgeV4`）。

---

## 二、V4.0 核心概念与数据模型

V4.0 采用严格的**两阶段状态握手合约**：

```
[ 顶点 A: data ready ] ───( 边 Edge: 业务计算 / LLM 推理 )───> [ 顶点 B: todo -> data ready ]
```

1. **顶点状态 (VertexStateV4)**：
   - `data ready`：数据生产完成，下游可消费。
   - `todo` / `todo urgent`：下游需求信号，表示等待数据进入。
   - `reject`：计算发生异常，触发自愈边介入。
   - `forbidden`：超过重试上限后永久熔断。
2. **顶点属性标签 (VertexAttributeV4)**：
   - `start`：图输入入口点。
   - `end`：图最终汇聚点。
   - `json`：强制下游对模型输出进行 JSON 格式解析与 Markdown 围栏剥离。
   - `subgraph`：标记该节点为嵌套子图容器。
3. **边类型 (EdgeV4)**：
   - `CodeEdgeV4`：执行本地 Python 同步或异步处理函数。
   - `SenseNovaEdgeV4`：开箱即用直连商汤 SenseNova 6.8 远程大模型。
   - `LLMEdgeV4`：适配任意符合 Chat / Generate / Process 协议的自定义 Agent。
   - `ReflexiveEdgeV4`：自环自愈边，捕获 `reject` 状态并重置为 `todo urgent` 触发重试。

---

## 三、快速上手：代码级编排实践

### 1. 基础线性流水线 (CodeEdgeV4)

```python
import asyncio
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, CodeEdgeV4
)

async def run_linear_pipeline():
    # 1. 初始化 SQLite 存储
    store = VertexStoreV4(":memory:")
    session_id = "sess_demo_linear"
    graph = GraphV4(session_id=session_id)

    # 2. 创建三个节点：起点 -> 中间点 -> 终点
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="v_start",
        content="hello world",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value  # 起始点预置就绪
    ))
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="v_mid",
        content="", state=VertexStateV4.TODO.value
    ))
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="v_end",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    # 3. 编写变换处理函数
    def to_upper(content, settings, staging):
        return content.upper()

    def add_tag(content, settings, staging):
        return f"[{content}] processed by vea-v4"

    graph.add_edge(CodeEdgeV4("e1", "v_start", "v_mid", script=to_upper))
    graph.add_edge(CodeEdgeV4("e2", "v_mid", "v_end", script=add_tag))
    graph.validate()

    # 4. 运行
    executor = ExecutorV4(graph=graph, store=store, max_concurrency=2)
    res = await executor.run()
    
    assert res.success is True
    print("最终结果:", store.get_vertex(session_id, "v_end").content)
    # 输出: [HELLO WORLD] processed by vea-v4

if __name__ == "__main__":
    asyncio.run(run_linear_pipeline())
```

---

### 2. 真实大模型推理管道 (SenseNovaEdgeV4)

本框架自带免费的 SenseNova 6.8 Flash Lite 接入端点，无需必须配置 Key 即可直接发起远端调用（如配置环境变量 `SENSENOVA_API_KEY` 则自动采用您的鉴权头）：

```python
import asyncio
import json
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, CodeEdgeV4, SenseNovaEdgeV4
)

async def run_sensenova_pipeline():
    store = VertexStoreV4(":memory:")
    session_id = "sess_llm_demo"
    graph = GraphV4(session_id)

    # 提示词节点
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="prompt_in",
        content='计算 25 * 4，以严格 JSON 返回: {"question": "25*4", "answer": 100}',
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))

    # 模型输出节点（声明 json 属性，自动校验结构）
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="model_out",
        content="",
        attributes=[VertexAttributeV4.JSON.value],
        state=VertexStateV4.TODO.value
    ))

    # 最终业务提取节点
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="final_res",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    # 真实远端大模型边
    llm_edge = SenseNovaEdgeV4(
        edge_id="e_llm",
        input_vertex="prompt_in",
        output_vertex="model_out",
        settings={"temperature": 0.1}
    )

    # 下游提取边
    def parse_answer(content, settings, staging):
        data = json.loads(content)
        return f"计算结果为: {data.get('answer')}"

    code_edge = CodeEdgeV4("e_parse", "model_out", "final_res", script=parse_answer)

    graph.add_edge(llm_edge)
    graph.add_edge(code_edge)
    graph.validate()

    executor = ExecutorV4(graph=graph, store=store)
    try:
        await executor.run()
        print(store.get_vertex(session_id, "final_res").content)
        # 输出: 计算结果为: 100
    finally:
        await llm_edge.close_agent()

if __name__ == "__main__":
    asyncio.run(run_sensenova_pipeline())
```

---

### 3. 多源入边汇聚与合并策略 (Fan-In Merge)

当多个并发边指向同一个目标节点时，可配置 `merge_strategy`。执行器自带**汇聚屏障（Settlement Barrier）**，在所有前驱边全部就绪之前维持 `todo`，杜绝下游提前触发：

```python
# settings 支持策略:
# 1. "merge_strategy": "json_merge"   -> 字典合并 {**existing, **incoming}
# 2. "merge_strategy": "list_append"  -> 数组追加 [item1, item2, ...]
# 3. "merge_strategy": "overwrite"    -> 覆盖模式 (默认)
# 4. "merge_strategy": "reducer_script", "reducer_script": "my_reducer.py:reduce_fn"

edge_a = CodeEdgeV4(
    "e_a", "in_a", "out_merge",
    script=lambda c, s, st: '{"module_a": "ok"}',
    settings={"merge_strategy": "json_merge"}
)

edge_b = CodeEdgeV4(
    "e_b", "in_b", "out_merge",
    script=lambda c, s, st: '{"module_b": "ok"}',
    settings={"merge_strategy": "json_merge"}
)

graph.add_edge(edge_a)
graph.add_edge(edge_b)
# 调度器自动执行屏障等待，最终目标节点将原子汇聚为: {"module_a": "ok", "module_b": "ok"}
```

---

### 4. 故障自愈重试与熔断保护 (ReflexiveEdgeV4)

```python
from framework import ReflexiveEdgeV4, CodeEdgeV4

# 业务处理边（如果抛出异常，目标节点将被自动置为 reject）
graph.add_edge(CodeEdgeV4("e_task", "v_in", "v_work", script=flaky_worker))

# 声明自环自愈边
graph.add_edge(ReflexiveEdgeV4(
    edge_id="e_retry",
    vertex_name="v_work",
    trigger_state="reject",       # 捕获异常触发状态
    target_state="todo urgent",   # 重新置为高优先级待执行
    max_retries=3                 # 超过 3 次触发熔断锁定为 'forbidden'
))
```

---

## 四、服务模式：Web 仪表盘与 HTTP/SSE API

框架提供开箱即用的 FastAPI 服务，自带可视化的图监控仪表板。

### 1. 启动服务

```bash
cd /home/luminovoez/agent
.venv/bin/python -m framework.server_v4 --host 127.0.0.1 --port 8000 --db workflow.db
```

### 2. 可视化 Web 仪表盘

打开浏览器访问：
👉 **`http://127.0.0.1:8000/dashboard`** 或 **`http://127.0.0.1:8000/`**

- **拓扑画布**：实时渲染 DAG 节点与连线，随节点生命周期自动刷新颜色。
- **在线图增删改**：无需写代码，直接在表单中新增 Vertex、连接 Edge、更改参数。
- **一键触发运行**：点击 `Run Workflow` 即可在服务端执行并实时输出事件。
- **SQLite 数据检视**：直接查看底层 `vertices` 与 `session_staging` 暂存日志。

### 3. 核心 API 端点与调用示例

#### ① 在线动态增添节点与连线
```bash
# 1. 创建入口节点
curl -X POST http://127.0.0.1:8000/api/sessions/demo/graph/vertices \
     -H "Content-Type: application/json" \
     -d '{"name": "v_start", "content": "hello", "state": "data ready", "attributes": ["start"]}'

# 2. 创建出口节点
curl -X POST http://127.0.0.1:8000/api/sessions/demo/graph/vertices \
     -H "Content-Type: application/json" \
     -d '{"name": "v_end", "content": "", "state": "todo", "attributes": ["end"]}'

# 3. 创建连接边
curl -X POST http://127.0.0.1:8000/api/sessions/demo/graph/edges \
     -H "Content-Type: application/json" \
     -d '{"id": "e_step", "type": "code", "input_vertex": "v_start", "output_vertex": "v_end"}'
```

#### ② 运行会话工作流
```bash
curl -X POST http://127.0.0.1:8000/api/sessions/demo/run \
     -H "Content-Type: application/json" \
     -d '{"max_concurrency": 4, "timeout": 60.0}'
```

#### ③ 监听实时 SSE 执行事件流
```bash
curl -N http://127.0.0.1:8000/api/sessions/demo/events
```

#### ④ 安全导出拓扑结构
```bash
# 具备路径边界安全检查与 .json 白名单校验
curl -X POST "http://127.0.0.1:8000/api/sessions/demo/graph/dump?path=./exported_graph.json"
```

---

## 五、分布式任务队列扩展 (BaseWorkerQueueV4)

对于跨节点、异步分布式图任务，框架在 [`framework/worker_queue_v4.py`](framework/worker_queue_v4.py) 中定义了标准的调度协议：

```python
from framework import BaseWorkerQueueV4, EdgeTaskPayload, EdgeTaskResult

class RedisWorkerQueue(BaseWorkerQueueV4):
    async def submit_edge(self, task: EdgeTaskPayload) -> str:
        # 将任务发布至 Redis / Celery 任务队列，返回 task_id
        ...
    async def poll_result(self, task_id: str) -> Optional[EdgeTaskResult]:
        # 非阻塞轮询任务执行状态
        ...
    async def wait_result(self, task_id: str, timeout: float = 120.0) -> EdgeTaskResult:
        # 阻塞等待完成事件
        ...
    async def cancel_task(self, task_id: str) -> bool:
        # 撤销任务排队或执行
        ...
    async def health_check(self) -> bool:
        # 队列连通性检测
        ...
```

---

## 六、旧版本 (Legacy V1-V3) 兼容使用指南

旧版本的 API 依然完整保留，无需改动即可在既有业务中平稳运行：

```python
from framework import Vertex, Edge, Graph, Executor

# 1. 纯内存 Vertex
v1 = Vertex(name="start", initial_data={"val": 10})
v2 = Vertex(name="end")

# 2. 基础 Edge
e = Edge(
    name="e_calc",
    source="start",
    dest="end",
    transform=lambda d: {"val": d["val"] * 2}
)

# 3. 基础 Graph 与调度器
g = Graph()
g.add_vertex(v1)
g.add_vertex(v2)
g.add_edge(e)

executor = Executor(g)
result = executor.run()

print("Legacy 输出:", v2.get("val"))  # 20
```

---

## 七、自动化测试与健康检查

无论在开发还是生产部署前，均可随时执行全量测试套件进行健康校验：

```bash
cd /home/luminovoez/agent

# 1. 运行 V4 全栈 6 层及真实大模型集成测试 (143 项全部通过)
.venv/bin/python -m pytest tests/test_v4_layer1_store.py \
                           tests/test_v4_layer2_edge.py \
                           tests/test_v4_layer3_graph.py \
                           tests/test_v4_layer4_executor.py \
                           tests/test_v4_layer5_server.py \
                           tests/test_v4_layer6_e2e.py \
                           tests/test_v4_system.py \
                           tests/test_v4_server.py \
                           tests/test_sensenova_live.py -v

# 2. 运行旧版本兼容性测试 (56 项全部通过)
.venv/bin/python -m pytest tests/test_vertex.py \
                           tests/test_edge.py \
                           tests/test_graph.py \
                           tests/test_executor.py -v
```
