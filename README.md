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
| **PI Agent** | Interface for AI/LLM processing. Mock included; plug in real agent when installed. |
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

```python
import asyncio
from framework import Graph, Executor, MockPIAgent

async def main():
    graph = Graph.from_json("config.json")
    result = await Executor(graph, MockPIAgent(), max_concurrency=8).run()
    print(result.summary())

asyncio.run(main())
```

## Examples

```bash
# Simple linear pipeline
python examples/run.py examples/simple/config.json

# Complex DAG with fan-out, fan-in, scripts
python examples/run.py examples/complex/config.json

# Object-Oriented Subclassing (Dynamic classes)
python examples/run.py examples/custom_classes/config.json
```

Each example folder contains its own `README.md` tutorial detailing how the pipeline is constructed.

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
├── framework/
│   ├── __init__.py          # Package exports
│   ├── vertex.py            # Vertex with state machine & data store
│   ├── edge.py              # Edge: source → PI Agent → destination
│   ├── graph.py             # JSON loader & DAG validator
│   ├── executor.py          # Async executor with concurrency control
│   ├── pi_agent.py          # PI Agent interface (Mock + External)
│   └── script_loader.py     # Dynamic .py script loader
├── examples/
│   ├── run.py               # Unified runner for all examples
│   ├── scripts/             # Reusable vertex/edge module hooks
│   ├── simple/              # Linear pipeline (with README tutorial)
│   ├── complex/             # Complex DAG pipeline (with README tutorial)
│   └── custom_classes/      # Object-Oriented subclasses (with README tutorial)
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
