# Vertex-Edge Agent Framework (顶点-边 智能体框架)

一个**非交互式**、数据驱动、高度可扩展的 DAG（有向无环图）执行引擎，专为编排和调度 AI Agent 生产级流水线而设计。

## 核心架构 (Unified Architecture)

框架采用了高度统一的 **Actor / Message-Passing (消息传递)** 模型。Vertex（节点）和 Edge（边）之间没有任何零散的方法调用，所有的交互全部通过单一的信号管道 `handle_edge_signal` 完成。

```
┌──────────┐    ┌───────────────────────────────────────────────┐    ┌──────────┐
│ Vertex A │───▶│                     Edge 1                    │───▶│ Vertex B │
│ (Source) │    │ Guard -> PreProcess -> Compute -> PostProcess │    │ (Sink)   │
└──────────┘    └───────────────────────────────────────────────┘    └──────────┘
```

### 1. Edge: 统一的 5 阶段流水线 (5-Stage Pipeline)
`Edge` 不再区分为普通边或条件边，而是统一为一个标准的 5 阶段流水线：
1. **Guard (门限拦截)**: 调用 `evaluate_condition` 进行前置校验（支持 JSON 声明式规则或外部 Python 脚本）。若不满足条件，直接产生 `ABORTED` 信号，触发向下的雪崩级分支剪枝，避免死锁。
2. **Pre-Process (预处理)**: 触发 `pre_process` 钩子处理原始数据。
3. **Compute (计算)**: 如果配置了 Prompt 和 Model，则通过 LLM (PI Agent) 计算；若未配置，则化身为透明的 Pass-through edge 直接透传数据。
4. **Post-Process (后处理)**: 触发 `post_process` 钩子进行解析或格式化。
5. **Deliver (交付)**: 向目标 Vertex 发送 `COMPLETED` 信号并写入结果。

### 2. Vertex: 统一的 3 阶段容器 (3-Stage Container)
`Vertex` 作为一个纯粹的黑盒状态机容器，分为三个生命周期：
1. **Ingest (摄入)**: 当收到边的 `COMPLETED` 信号时，触发 `on_receive` 拦截器/钩子。
2. **Settle (沉淀/屏障)**: 采用动态结算屏障（Settlement Barrier Check）。实时统计 `COMPLETED` 与 `ABORTED` 信号。若所有入边皆有定论，只要有一条成功则进入 `READY`，若全军覆没则进入 `ABORTED`。
3. **Fuse (融合)**: 结算完成后，引擎触发 `prepare_outputs()` (即 `on_ready` 钩子)，将零散的数据融合为出边所需的状态。

## JSON 配置规范 (Configuration Schema)

图的拓扑结构与执行规则完全由 JSON 驱动，支持声明式的阈值控制、脚本挂载与大模型配置：

```jsonc
{
  "metadata": { "name": "...", "description": "..." },
  "vertices": [
    {
      "id": "v1",
      "settings": { /* 任意配置字典 */ },
      "script": "path/to/vertex_script.py",      // 可选：挂载外部扩展脚本
      "initial_data": [                          // 可选：初始注入数据
        { "data_id": "text", "tags": ["en"], "value": "Hello" }
      ]
    }
  ],
  "edges": [
    {
      "id": "e1",
      "source": "v1",
      "destination": "v2",
      "data_id": "text",
      "tags": ["en"],
      "prompt": "Summarize this:",
      "model": "gemini-pro",
      "settings": {
        "threshold": 80,                         // 可选：Guard 门限配置（声明式）
        "operator": ">="
      },
      "script": "path/to/edge_script.py"         // 可选：挂载外部扩展脚本
    }
  ]
}
```

## 外部扩展脚本 (External Scripts)

通过配置 `script` 字段，可以将普通节点与边瞬间升级为具备复杂逻辑的组件，无需修改底层框架源码。

### Vertex Scripts (节点脚本)

```python
def on_receive(data, data_id, tags, settings):
    """【Ingest 阶段】数据到达时触发。可转换数据，或抛出异常以拒绝接收该数据。"""
    if not valid(data):
        raise ValueError("rejected")
    return data.upper()

def on_ready(all_data, settings):
    """【Fuse 阶段】节点就绪，即将触发下游出边前调用。用于将多个输入融合为最终输出。"""
    return {("output_id", ("tag",)): merged_value}
```

### Edge Scripts (边脚本)

```python
def guard(data, settings):
    """【Guard 阶段】条件门限，返回 False 则剪枝当前分支。也叫 evaluate_condition。"""
    return data.get("score", 0) >= 80

def pre_process(data, settings):
    """【Pre-process 阶段】在进入 LLM 之前转换数据。"""
    return f"【请分析以下内容】\n{data}"

def post_process(data, settings):
    """【Post-process 阶段】解析 LLM 的输出。"""
    return data.strip()
```

## 运行方式 (Usage)

```python
import asyncio
from framework import Graph, Executor, MockPIAgent

async def main():
    # 1. 解析 DAG 图配置
    graph = Graph.from_json("config.json")
    # 2. 注入真实或 Mock 的 Agent，配置并发度并启动引擎
    result = await Executor(graph, MockPIAgent(), max_concurrency=8).run()
    # 3. 打印执行摘要
    print(result.summary())

asyncio.run(main())
```

## 示例 (Examples)

```bash
# 简单的线性流水线
python examples/run.py examples/simple/config.json

# 复杂的 DAG（支持扇出 Fan-out、扇入 Fan-in、外部脚本）
python examples/run.py examples/complex/config.json

# 动态分支路由与条件剪枝 (Guard & Routing)
python examples/run.py examples/conditional_routing/config.json

# 面向对象的高级用法 (自定义子类重载)
python examples/run.py examples/custom_classes/config.json
```
*每个示例文件夹均包含专门的 `README.md` 教程。*

## 单元测试 (Tests)

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

当前包含 **72 个全覆盖测试**，涵盖：状态机、统一信号传递 (EdgeSignal)、标签排序、并发信号量、动态路由剪枝 (Diamond Routing)、死锁预防、脚本钩子拦截、图循环检测、超时与错误抛出。
