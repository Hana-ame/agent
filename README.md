# Vertex-Edge Agent Framework

A **non-interactive**, JSON-driven graph execution engine for orchestrating AI agent pipelines.

## Architecture

```
┌──────────┐    ┌──────┐    ┌──────────┐    ┌──────┐    ┌──────────┐
│ Vertex A │───▶│Edge 1│───▶│ Vertex B │───▶│Edge 2│───▶│ Vertex C │
│ (source) │    │PI Agt│    │(process) │    │PI Agt│    │  (sink)  │
└──────────┘    └──────┘    └──────────┘    └──────┘    └──────────┘
```

### Core Concepts

| Component    | Role |
|-------------|------|
| **Vertex**   | Stores data keyed by `(data_id, tags[])`. Has state machine: `IDLE → READY → PROCESSING → DONE`. Can reject data via scripts. |
| **Edge**     | Connects source → destination. Reads from source via `get(id, tags)`, processes through PI Agent, writes to dest via `set(data, id, tags)`. |
| **Executor** | Scans for READY vertices, fires outgoing edges concurrently (semaphore-bounded), detects deadlocks. |
| **PI Agent** | Agent 抽象接口 (`PIAgent`)。实现：`MockPIAgent`(测试)、`PICLIPIAgent`(通过 pi CLI 调真实 LLM)、`OpenCodeAgent`(通过 opencode CLI 调真实 LLM)、`ExternalPIAgent`(委托第三方 `pi_agent` 包)。 |
| **Scripts**  | External `.py` files for vertex hooks (`on_receive`, `on_ready`) and edge hooks (`pre_process`, `post_process`). |

### Vertex States

```
IDLE ──(all inputs received)──▶ READY ──(executor picks up)──▶ PROCESSING ──(edges done)──▶ DONE
                                                                    │
                                                                    └──(error)──▶ ERROR
```

## JSON Configuration Schema

```jsonc
{
  "metadata": { "name": "...", "description": "..." },
  "vertices": [
    {
      "id": "v1",
      "settings": { /* arbitrary */ },
      "script": "path/to/vertex_script.py",      // optional
      "initial_data": [                            // optional
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
      "settings": {},                              // optional
      "script": "path/to/edge_script.py"           // optional
    }
  ]
}
```

## External Scripts

### Vertex Scripts

```python
def on_receive(data, data_id, tags, settings):
    """Called when data arrives. Return transformed data or raise to reject."""
    if not valid(data):
        raise ValueError("rejected")
    return data.upper()

def on_ready(all_data, settings):
    """Called before outgoing edges fire. Merge inputs → outputs."""
    return {("output_id", ("tag",)): merged_value}
```

### Edge Scripts

```python
def pre_process(data, settings):
    """Transform data BEFORE the PI Agent."""
    return f"[PREFIX] {data}"

def post_process(data, settings):
    """Transform result AFTER the PI Agent."""
    return f"{data} [SUFFIX]"
```

## Usage

> 字段与自定义教程详见 [`docs/JSON-SCHEMA.md`](docs/JSON-SCHEMA.md)；
> 可直接运行的通用模板见根目录 [`config.template.json`](config.template.json)。

```python
import asyncio
from framework import Graph, Executor, MockPIAgent, PICLIPIAgent, OpenCodeAgent

async def main():
    graph = Graph.from_json("config.json")
    # 测试: 用确定性的 Mock agent
    result = await Executor(graph, MockPIAgent(), max_concurrency=8).run()
    print(result.summary())

    # 真实模型: 用 pi CLI 后端
    agent = PICLIPIAgent(provider="scnet", model="DeepSeek-V4-Flash-0731-Event")
    result = await Executor(graph, agent, max_concurrency=8, timeout=600).run()

    # 真实模型: 用 opencode CLI 后端（默认免费模型 opencode-zen/hy3-free）
    agent = OpenCodeAgent()
    result = await Executor(graph, agent, max_concurrency=8, timeout=600).run()
    print(result.summary())

asyncio.run(main())
```

## Examples

```bash
# Simple linear pipeline (3 vertices, 2 edges)
python examples/simple/run.py

# Complex DAG with fan-out, fan-in, scripts (5 vertices, 5 edges)
# 默认用 opencode 后端(免费模型 opencode-zen/hy3-free) 生成多源情报周报
python examples/complex/run.py

# 通用运行器：默认后端 = opencode-zen/hy3-free，也可显式选择后端
python3 scripts/run_graph.py config.template.json                            # 默认: opencode + hy3-free
python3 scripts/run_graph.py config.template.json --agent mock               # Mock(快速调试)
python3 scripts/run_graph.py config.template.json --agent pi --model DeepSeek-V4-Flash-0731-Event   # pi CLI

# Custom vertex subclassing (3 vertices, 2 edges)
python examples/custom_vertex/run.py
```

### simple — linear chain

输入 → 处理 → 输出，数据逐级流动的最简管线：

```
┌─────────┐  e1 (text:en)  ┌────────────┐  e2 (text:en)  ┌─────────┐
│  input  │───────────────▶│ processor  │───────────────▶│ output  │
│ (source)│                │  (process) │                │ (sink)  │
└─────────┘                └────────────┘                └─────────┘
```

### complex — fan-out / fan-in DAG (real pi agent)

真实业务目的：多源科技情报分析 —— 一篇新闻 + 分析关注点送入图，
经 fan-out/fan-in 汇聚后由真实 pi agent 生成结构化周报摘要。

两个源顶点，`transform` 与 `merge` 各汇入 2 个输入，`input_a` 扇出两条支路：

```
                          ┌─────────────────┐
 input_a ──e1───────────▶ │                 │
 (text:raw)               │    transform    │
                          │  (uppercase +   │
 input_b ──e2───────────▶ │     merge)      │
 (text:context)           └────────┬────────┘
                                   │
                                   │ e3 (result:analysis)
                                   ▼
                          ┌─────────────────┐
 input_a ──e4───────────▶ │      merge      │
 (text:original)          │  (validator +   │
                          │      report)    │
                          └────────┬────────┘
                                   │
                                   │ e5 (final:report)
                                   ▼
                          ┌─────────────────┐
                          │     output      │
                          └─────────────────┘
```

运行方式(真实调用 LLM，5 次)：

```bash
python examples/complex/run.py
```

### custom_vertex — subclassed vertices

用 Python 继承 `Vertex` 自定义顶点行为（清洗/统计、报告聚合），
并手动组装图运行（`from_json` 只能构造内置 `Vertex`）：

```
┌─────────────────┐  e1 (text:en)  ┌─────────────────┐  e2 (stats:summary)  ┌───────────────┐
│  input          │───────────────▶│  processor      │─────────────────────▶│  output       │
│ SanitizeVertex  │               │ SanitizeVertex  │                      │ ReportVertex  │
│ (source)        │               │ (clean + count) │                      │ (sink)        │
└─────────────────┘               └─────────────────┘                      └───────────────┘
```

## Tests

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

**62 tests** covering: vertex state machine, get/set, tag ordering, readiness
semaphore, script hooks (transform/reject/on_ready), edge execution, graph
loading/validation/cycle detection, executor (linear/diamond/fan-out/fan-in),
concurrency, timeout, error handling, deep chains, rejection pipelines.

## Project Structure

```
vertex_edge_agent/
├── scripts/               # 通用运行器
│   └── run_graph.py       # 跑任意 config，支持 mock/pi/opencode 后端
├── docs/
│   └── JSON-SCHEMA.md     # JSON 配置中文文档
├── framework/
│   ├── __init__.py          # Package exports
│   ├── vertex.py            # Vertex with state machine & data store
│   ├── edge.py              # Edge: source → PI Agent → destination
│   ├── graph.py             # JSON loader & DAG validator
│   ├── executor.py          # Async executor with concurrency control
│   ├── pi_agent.py          # PI Agent interface (Mock + External)
│   └── script_loader.py     # Dynamic .py script loader
├── examples/
│   ├── scripts/             # Reusable vertex/edge scripts
│   │   ├── uppercase_handler.py
│   │   ├── validator.py
│   │   └── prefix_handler.py
│   ├── simple/              # Linear pipeline example
│   │   ├── config.json
│   │   └── run.py
│   ├── complex/             # DAG with fan-out/fan-in (real pi agent)
│   │   ├── config.json
│   │   └── run.py
│   └── custom_vertex/       # Custom vertex subclassing example
│       ├── custom_vertex.py
│       └── run.py
├── tests/                   # 62 tests
│   ├── conftest.py
│   ├── test_vertex.py
│   ├── test_edge.py
│   ├── test_graph.py
│   ├── test_executor.py
│   └── test_integration.py
├── pyproject.toml
├── requirements.txt
└── README.md
```
