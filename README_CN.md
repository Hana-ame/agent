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

```python
import asyncio
from framework import Graph, Executor, MockAgent

# 1. 声明式配置工作流拓扑
config = {
    "vertices": [
        {"id": "InputNode", "initial_data": [{"channel": "text", "value": "人工智能的未来趋势"}]},
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

async def main():
    # 2. 从字典或 JSON 文件加载图
    graph = Graph.from_dict(config)
    
    # 3. 运行执行器
    executor = Executor(graph, agents=MockAgent(), max_concurrency=4)
    result = await executor.run()
    
    # 4. 打印执行摘要
    print(result.summary())

asyncio.run(main())
```

---

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

```python
from framework import Graph, CheckpointedExecutor, SQLiteStateStore, MockAgent

store = SQLiteStateStore("my_workflow.db")

config = {
    "vertices": [
        {"id": "Order", "initial_data": [{"channel": "amt", "value": 50000}]},
        {"id": "PaymentRiskGate", "settings": {"require_approval": True}}, # 挂起点
        {"id": "BankTransfer"}
    ],
    "edges": [
        {"id": "e1", "source": "Order", "destination": "PaymentRiskGate", "channel": "amt"},
        {"id": "e2", "source": "PaymentRiskGate", "destination": "BankTransfer", "channel": "amt"}
    ]
}

# 阶段 1：启动执行，自动暂停在 PaymentRiskGate
graph = Graph.from_dict(config)
executor = CheckpointedExecutor(graph, store=store, agents=MockAgent())
await executor.run()  # 状态变为 awaiting_approval

# 阶段 2：人工审查并批准
resumed_graph = Graph.from_dict(config)
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

将独立的子图（如由 SearchAgent + FactChecker 组成的调研团队）作为单节点嵌入父图：

```jsonc
{
  "id": "ResearchDepartment",
  "type": "subgraph",  // 指定为嵌套子图类型
  "settings": {
    "graph_config": {
      "vertices": [{"id": "SearchAgent"}, {"id": "FactChecker"}],
      "edges": [...]
    },
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

```python
from framework import Graph, Executor, MemoryStore

config = {
    "vertices": [{"id": "A", "initial_data": [{"channel": "x", "value": 1}]}, {"id": "B"}],
    "edges": [
        {
            "id": "e1",
            "source": "A",
            "destination": "B",
            "channel": "x",
            "model": "gemini-1.5-pro",
            "settings": {
                "memory_write": {"auth_token": "global_auth_token"}, # 写入全局内存
                "memory_read": ["user_profile"]                      # 读取全局内存
            }
        }
    ]
}

memory = MemoryStore({"user_profile": {"role": "admin"}})
executor = Executor(Graph.from_dict(config), memory=memory)
result = await executor.run()

# 查看 Token 与预估成本分析
print(result.summary())
```

---

## 💻 实战演示案例 (Examples)

框架自带完整的开箱即用终端演示，位于 `examples/` 目录下：

```bash
# 1. 🌈 实时 ANSI 彩色事件流监控演示
python examples/realtime_streaming/demo.py

# 2. 🔁 大模型业务报错与 Prompt 自我纠错演示
python examples/self_correction/demo.py

# 3. 🛑 人类在回路（HITL）审批与 SQLite 快照恢复演示
python examples/hitl_approval/demo.py

# 4. 🏢 嵌套子图分层多智能体团队协作演示
python examples/subgraph/demo.py
```

---

## 🧪 自动化测试

运行全量单元测试与集成测试：

```bash
python -m pytest tests/ -v
```

当前包含 **125 个全量通过的测试用例**，覆盖拓扑静态校验、循环回边、菱形死锁防护、自纠错重试、事件流冒泡、子图映射、内存 TTL 与成本核算。
