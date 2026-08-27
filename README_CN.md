# Vertex-Edge Agent Framework 中文使用指南

**Vertex-Edge Agent** 是一个专为生产级 AI Agent 管道编排与调度设计的**数据驱动、高扩展、基于 Actor 消息传递模型的图执行引擎**。

它摒弃了传统工作流中繁琐耦合的方法调用，采用统一的 **点（Vertex - 状态机容器）** 与 **边（Edge - 计算管道）** 模型，原生支持**动态条件剪枝、有界循环反馈、人类在回路审批、业务级自纠错重试、实时事件流式传输、嵌套子图代理以及全局共享内存**。

---

## 🌟 核心特性概览

| 架构层级 | 核心特性 | 功能描述 |
| :--- | :--- | :--- |
| **v1.0 核心引擎** | **统一 5 阶段 Edge 管道** | `Guard`(守卫拦截) $\rightarrow$ `Pre-process`(前置处理) $\rightarrow$ `Compute`(LLM计算) $\rightarrow$ `Post-process`(后置解析) $\rightarrow$ `Deliver`(信号交付)。 |
| | **Actor 消息驱动与防死锁** | 节点与边完全通过统一信号（`EdgeSignal`）通信，支持菱形并发分支自动剪枝（Cascading Aborts），杜绝死锁。 |
| | **配置驱动与脚本解耦** | 纯 JSON/Dict 声明拓扑，支持外部 Python 脚本无侵入式挂载拦截钩子。 |
| **v2.0 弹性与交互** | **业务级自纠错重试** | 捕获大模型输出解析异常（如 `KeyError`, `JSONDecodeError`），自动将错误反馈注入 Prompt（`[SYSTEM FEEDBACK]`）并指数退避重试。 |
| | **状态持久化与快照恢复** | 基于 SQLite 实现微步状态快照保存，支持工作流中断恢复与审计溯源。 |
| | **人类在回路 (HITL)** | 节点支持 `PAUSED` 状态，支持配置驱动（`require_approval`）自动暂停等待人工审批，外部调用 `approve(data)` 唤醒。 |
| | **有界状态循环 (Loops)** | 拓扑支持有向有环图（Cycles），DFS 严格校验回边 `max_iterations > 0`，支持自动重入调度。 |
| | **非阻塞实时事件流** | 暴露 `executor.stream()` 异步生成器，无阻塞派发结构化 `GraphEvent`，适用于 WebSockets 与前端监控。 |
| **v3.0 企业级编排** | **分层嵌套子图 (`SubgraphVertex`)** | “盒中盒”模型，单个节点封装完整的多智能体子图（如调研团队），支持边界映射与多级事件冒泡。 |
| | **全局共享内存 (`MemoryStore`)** | 跨节点共享键值总线，支持 TTL 自动过期、作用域隔离（`scope`）与边声明式读写。 |
| | **Token 与成本追踪 (`TelemetryTracker`)** | 自动记录每条边的 Prompt/Completion Token、耗时与模型预估美元成本，汇总输出财务明细。 |

---

## 🚀 快速上手

### 1. 安装与环境准备

依赖 Python 3.10+：

```bash
git clone https://github.com/gekkasayu/vertex_edge_agent.git
cd vertex_edge_agent
pip install pytest pytest-asyncio
```

### 2. 最简单的 3 节点线性工作流

创建一个简单的工作流：`InputNode -> LLM Analysis -> OutputNode`。

创建 `config.json` 文件声明图拓扑：

```json
{
  "vertices": [
    {
      "id": "InputNode",
      "initial_data": [{"channel": "text", "value": "人工智能的未来趋势"}]
    },
    {"id": "OutputNode"}
  ],
  "edges": [
    {
      "id": "e_analyze",
      "source": "InputNode",
      "destination": "OutputNode",
      "channel": "text",
      "prompt": "请用一句话总结以下主题的核心观点：",
      "model": "gemini-pro"
    }
  ]
}
```

创建 `main.py` 加载并执行：

```python
import asyncio
from framework import Graph, Executor, MockAgent

async def main():
    # 1. 从 JSON 文件加载图配置
    graph = Graph.from_json_file("config.json")
    
    # 2. 运行执行器
    executor = Executor(graph, agents=MockAgent(), max_concurrency=4)
    result = await executor.run()
    
    # 3. 打印执行摘要
    print(result.summary())

asyncio.run(main())
```

---


---

## ⚙️ 节点与边配置指南 (Configuration Guide)

整个框架采用 JSON 驱动的配置模式。你可以将拓扑结构完全写在 `.json` 文件中，并由 `Graph.from_json_file()` 自动加载解析。

### 1. 节点 (Vertex) 配置参数

节点是状态机的容器。在 JSON 的 `vertices` 数组中定义：

| 字段名 (`Key`) | 类型 | 必填 | 默认值 | 说明与可选值 |
| :--- | :--- | :---: | :--- | :--- |
| **`id`** | `str` | **是** | - | 节点的唯一标识符（如 `"DataIngest"`）。 |
| **`type`** | `str` | 否 | `"vertex"` | `"vertex"` (标准单节点) 或 `"subgraph"` (嵌套子图节点)。 |
| **`initial_data`** | `list[dict]` | 否 | `[]` | 节点的初始注入数据列表（通常用于源节点）。每个字典包含 `channel` 与 `value`。 |
| **`script`** | `str` | 否 | `null` | 外挂 Python 脚本路径，用于注入 `on_receive` 或 `on_ready` 钩子。 |
| **`settings`** | `dict` | 否 | `{}` | 节点的业务高级配置字典。 |

**`settings` 高级控制选项：**
* `"require_approval"`: (`bool`) 设为 `true` 启用人类在回路 (HITL)，执行到此节点自动暂停并打入 SQLite 快照。
* `"graph_config"`: (`str` / `dict`) 仅当 `type="subgraph"` 时必填，子图配置的 **JSON 文件路径**（如 `"subgraphs/team.json"`）。
* `"input_map"` / `"output_map"`: 嵌套子图的输入/输出变量重定向映射。

### 2. 边 (Edge) 配置参数

边是承载 5 阶段计算与路由的分支管道。在 JSON 的 `edges` 数组中定义：

| 字段名 (`Key`) | 类型 | 必填 | 默认值 | 说明与可选值 |
| :--- | :--- | :---: | :--- | :--- |
| **`id`** | `str` | **是** | - | 边的唯一标识符（如 `"e_analyze"`）。 |
| **`source`** | `str` | **是** | - | 起始节点 ID。 |
| **`destination`** | `str` | **是** | - | 目标节点 ID。 |
| **`channel`** | `str` | 否 | `"default"` | 数据流转通道名称。 |
| **`prompt`** | `str` | 否 | `""` | 提示词模版。如果不填，边作为**透明通道**直接透传数据。 |
| **`model`** | `str` | 否 | `"default"`| 大模型名称（如 `"gemini-1.5-pro"`）。 |
| **`max_iterations`** | `int` | 否 | `0` | **有界循环控制**：设为 `> 0` 将此边标记为回环边，允许循环 `N` 次。 |
| **`script`** | `str` | 否 | `null` | 外挂 Python 脚本路径，用于注入数据前置/后置处理钩子。 |
| **`settings`** | `dict` | 否 | `{}` | 边的计算、条件守卫、自纠错与内存配置字典。 |

**`settings` 高级控制选项：**
* **条件路由 (Guard)**: `"threshold"` (阈值), `"operator"` (`">=", "=="` 等), `"field"` (用于读取字典里的数字提取对比)。未达标分支会自动触发 `ABORTED` 剪枝。
* **业务自纠错 (`retry_policy`)**: 包含 `"max_retries"`, `"backoff_factor"`, `"retry_on"`(如 `["KeyError", "JSONDecodeError"]`)。
* **全局内存 (`memory_read` / `memory_write`)**: 数组形式声明前置读取（`["token"]`），字典形式声明后置写入（`{"auth": "token"}`）。

## 📖 核心功能使用指南

### 1. 实时事件流监控 (Real-Time Event Streaming)

通过 `async for event in executor.stream()` 实时捕获节点与边的生命周期，而无需等待全图结束：

```python
executor = Executor(graph, agents=MockAgent())

async for event in executor.stream():
    # event 是标准 GraphEvent 对象
    print(f"[{event.timestamp}] {event.event_type} - Node: {event.vertex_id}")
    if event.event_type == "edge_completed":
        print(f"  ↳ 边完成，数据：{event.payload.get('result')}")

# 获取最终结果
result = executor._result
```

---

### 2. 业务级自纠错重试 (LLM Self-Correction)

当大模型返回格式错误（如缺少 JSON 键）导致后置脚本报错时，配置 `retry_policy` 会将报错原因自动注入 Prompt 引导大模型自我修正：

```jsonc
{
  "id": "e_extract",
  "source": "RawText",
  "destination": "StructuredData",
  "channel": "text",
  "prompt": "提取实体为 JSON 格式: {\"entities\": [...] }",
  "settings": {
    "retry_policy": {
      "max_retries": 3,
      "backoff_factor": 1.0,
      "retry_on": ["KeyError", "JSONDecodeError", "ValueError"]
    }
  }
}
```

---

### 3. 人类在回路审批流 (HITL & Checkpoints)

将敏感节点配置为需审批，引擎执行到该节点时会自动挂起并打上快照：

将敏感节点配置为需审批，引擎执行到该节点时会自动挂起并打上快照：

```jsonc
// hitl_config.json
{
  "vertices": [
    {"id": "Order", "initial_data": [{"channel": "amt", "value": 50000}]},
    {"id": "PaymentRiskGate", "settings": {"require_approval": true}}, // 挂起点
    {"id": "BankTransfer"}
  ],
  "edges": [
    {"id": "e1", "source": "Order", "destination": "PaymentRiskGate", "channel": "amt"},
    {"id": "e2", "source": "PaymentRiskGate", "destination": "BankTransfer", "channel": "amt"}
  ]
}
```

```python
from framework import Graph, CheckpointedExecutor, SQLiteStateStore, MockAgent

store = SQLiteStateStore("my_workflow.db")

# 阶段 1：启动执行，自动暂停在 PaymentRiskGate
graph = Graph.from_json_file("hitl_config.json")
executor = CheckpointedExecutor(graph, store=store, agents=MockAgent())
await executor.run()  # 状态变为 awaiting_approval

# 阶段 2：人工审查并批准
resumed_graph = Graph.from_json_file("hitl_config.json")
gate = resumed_graph.vertices["PaymentRiskGate"]
gate.approve({"auth_by": "Compliance_Manager", "auth_code": "AUTH-2026"})

# 阶段 3：从 SQLite 快照恢复继续执行
resume_executor = await CheckpointedExecutor.resume(
    run_id=executor.run_id,
    graph=resumed_graph,
    store=store,
    agents=MockAgent()
)
```

---

### 4. 分层嵌套子图 (`SubgraphVertex`)

将独立的子图（如由 SearchAgent + FactChecker 组成的调研团队）定义在另一个 JSON 中，并在父图中作为单节点引入：

```jsonc
{
  "id": "ResearchDepartment",
  "type": "subgraph",  // 指定为嵌套子图类型
  "settings": {
    // 引用外部子图配置文件
    "graph_config": "subgraphs/research_team.json", 
    // 输入映射：父图入边 'topic' -> 子图 'SearchAgent' 的 'query'
    "input_map": { "topic": "SearchAgent.query" },
    // 输出映射：子图 'FactChecker' 的 'verified' -> 父图出边 'report'
    "output_map": { "FactChecker.verified": "report" }
  }
}
```

---

### 5. 全局共享内存与 Token/成本遥测

无需在复杂的长链路中层层透传参数，利用 `MemoryStore` 与 `TelemetryTracker`：

```jsonc
// memory_config.json
{
  "vertices": [{"id": "A", "initial_data": [{"channel": "x", "value": 1}]}, {"id": "B"}],
  "edges": [
    {
      "id": "e1",
      "source": "A",
      "destination": "B",
      "channel": "x",
      "model": "gemini-1.5-pro",
      "settings": {
        "memory_write": {"auth_token": "global_auth_token"}, // 写入全局内存
        "memory_read": ["user_profile"]                      // 读取全局内存
      }
    }
  ]
}
```

```python
from framework import Graph, Executor, MemoryStore

# 初始化全局内存注入
memory = MemoryStore({"user_profile": {"role": "admin"}})

graph = Graph.from_json_file("memory_config.json")
executor = Executor(graph, memory=memory)
result = await executor.run()

# 查看 Token 与预估成本分析
print(result.summary())
```

---

## 💻 官方演示案例 (Examples)

`examples/` 目录提供了框架核心特性的独立运行示例。其目的是展示如何在实际场景中组合配置节点与边，以及如何调用执行器 API。

### 1. 案例说明与实现功能

| 案例目录 | 核心功能 | 运行命令 |
| :--- | :--- | :--- |
| **`realtime_streaming/`** | 展示非阻塞 `executor.stream()`，拦截底层 `GraphEvent` 并渲染 ANSI 颜色日志。 | `python examples/realtime_streaming/demo.py` |
| **`self_correction/`** | 模拟 LLM 输出格式错误，触发 `retry_policy` 捕获异常，并带入错误堆栈引导 LLM 自我修复。 | `python examples/self_correction/demo.py` |
| **`hitl_approval/`** | 演示敏感节点的 `require_approval` 挂起，生成 SQLite 快照后，人工调用 `approve()` 唤醒继续。 | `python examples/hitl_approval/demo.py` |
| **`subgraph/`** | 演示层级嵌套。父图引用 `research_team.json`，通过边界映射连接父图数据与子智能体网络。 | `python examples/subgraph/demo.py` |

### 2. 如何从零配置一个新案例 (From Scratch)

以配置一个简单的自定义应用为例：

1. **建立目录结构**：
   ```text
   my_agent/
   ├── config.json         # 必须：声明图拓扑
   ├── run.py              # 必须：执行器入口
   └── hooks.py            # 可选：自定义处理逻辑
   ```

2. **编写配置 (`config.json`)**：
   声明最简的源节点、目的节点及连接它们的边。若需挂载脚本，配置 `"script": "hooks.py"`。

3. **编写脚本 (`hooks.py`)**：
   ```python
   def pre_process(data, settings):
       # 处理边的数据
       return data
   ```

4. **编写入口 (`run.py`)**：
   ```python
   import asyncio
   from framework import Graph, Executor, HttpLLMAgent
   
   async def main():
       # 配置 LLM Agent (需设置 API Key 环境变量)
       agent = HttpLLMAgent()
       
       # 加载与执行
       graph = Graph.from_json_file("config.json")
       executor = Executor(graph, agents=agent)
       await executor.run()
   
   asyncio.run(main())
   ```

### 3. 注意事项 (Precautions)

1. **绝对路径与相对路径**：JSON 配置文件中的 `"script"` 或 `"graph_config"` 路径，相对于**运行入口 (`run.py`) 所在的执行目录 (CWD)** 计算。推荐使用相对于当前工作目录的路径。
2. **死锁防御**：如果一条边配置了条件守卫（`threshold`），该节点上**必须**有默认兜底边（或者所有边都被拒绝时能合法触发 `ABORTED` 级联取消），否则目标节点将永远处于等待状态（死锁）。
3. **无限循环防护**：在设计具有回退重试（闭环）的拓扑时，产生回边的 Edge **必须**配置 `"max_iterations": N`。若不配置，系统将在启动前抛出 `GraphCycleError`。
4. **LLM 模型配置**：示例中通常使用 `MockAgent`。实战中需替换为 `HttpLLMAgent`，并确保环境变量已配置（如 `OPENAI_API_KEY` 或 `GEMINI_API_KEY`）。

---

## 🧪 自动化测试

运行全量单元测试与集成测试：

```bash
python -m pytest tests/ -v
```

当前包含 **125 个全量通过的测试用例**，覆盖拓扑静态校验、循环回边、菱形死锁防护、自纠错重试、事件流冒泡、子图映射、内存 TTL 与成本核算。
