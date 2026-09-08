# Vertex-Edge Agent Framework Guide

Vertex-Edge Agent Framework is a data-driven agent orchestration system where **vertices** store states and data while **edges** encapsulate logic, transforms, and model inference.

---

## 1. Installation & Setup

- **Python Requirement**: `Python 3.12+`

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install editable package
pip install -e .
```

---

## 2. Quick Start

### Approach A: Orchestration via JSON Manifest (Recommended)

#### Step 1: Write execution script (`workflow/trans.py`)

```python
def to_uppercase(content: str, settings: dict, staging: dict) -> str:
    """Convert input content to uppercase."""
    return content.upper()

def add_signature(content: str, settings: dict, staging: dict) -> str:
    """Append author signature to content."""
    author = settings.get("author", "VEA")
    return f"{content}\n-- Processed by {author}"
```

#### Step 2: Define graph manifest (`workflow/graph.json`)

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

#### Step 3: Load and execute

```python
import asyncio
from framework import VertexStoreV4, DiscreteGraphLoaderV4, ExecutorV4

async def main():
    store = VertexStoreV4(":memory:")
    graph = DiscreteGraphLoaderV4.load_from_manifest("workflow/graph.json")
    DiscreteGraphLoaderV4.populate_store(graph, store)

    executor = ExecutorV4(graph=graph, store=store, max_concurrency=2)
    result = await executor.run()

    print("Success:", result.success)
    final_node = store.get_vertex(graph.session_id, "output_node")
    print("Output:\n", final_node.content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Approach B: Programmatic Orchestration via Python API

```python
import asyncio
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, CodeEdgeV4
)

async def main():
    store = VertexStoreV4(":memory:")
    session_id = "code_sess"
    graph = GraphV4(session_id=session_id)

    # 1. Add vertices
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

    # 2. Add transformation edge
    def reverse_text(data, settings, staging):
        return data[::-1]

    graph.add_edge(CodeEdgeV4("edge_rev", "src", "dst", script=reverse_text))
    graph.validate()

    # 3. Run executor
    executor = ExecutorV4(graph=graph, store=store)
    result = await executor.run()

    print("Result:", store.get_vertex(session_id, "dst").content)  # Output: ytivargitnA

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Approach C: Load via Discrete JSON File Paths or Directories

`graph.add_vertex()` and `graph.add_edge()` accept direct `.json` paths or directory paths:

```python
from framework import GraphV4

graph = GraphV4(session_id="path_loading_session")

# 1. Load vertices directly via individual JSON files
graph.add_vertex("examples/subgraph_v4/parent_in.json")
graph.add_vertex("examples/subgraph_v4/parent_subgraph.json")
graph.add_vertex("examples/subgraph_v4/parent_out.json")

# 2. Load edges directly via individual JSON files
graph.add_edge("examples/subgraph_v4/e_start_to_subgraph.json")
graph.add_edge("examples/subgraph_v4/e_subgraph_to_output.json")

# 3. Batch load all vertices and edges from a directory
dir_graph = GraphV4.from_directory("examples/subgraph_v4/discrete_dir_demo", session_id="dir_sess")
```

---

## 3. Remote Model Pipeline (SenseNova)

Built-in support for remote model inference (optionally set credentials via `SENSENOVA_API_KEY`):

```python
import asyncio
import json
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, SenseNovaEdgeV4, CodeEdgeV4
)

async def main():
    store = VertexStoreV4(":memory:")
    session_id = "llm_sess"
    graph = GraphV4(session_id=session_id)

    # 1. Prompt vertex
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="prompt_node",
        content="Calculate 15 + 25. Output strictly JSON: {\"result\": 40}",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))

    # 2. Model result vertex
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="model_node",
        content="",
        attributes=[VertexAttributeV4.JSON.value],
        state=VertexStateV4.TODO.value
    ))

    # 3. Final target vertex
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="final_node",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    # 4. Connect model edge and post-processing code edge
    graph.add_edge(SenseNovaEdgeV4(
        edge_id="e_llm",
        input_vertex="prompt_node",
        output_vertex="model_node",
        settings={"temperature": 0.1}
    ))

    def parse_result(content, settings, staging):
        data = json.loads(content)
        return f"Computed result: {data['result']}"

    graph.add_edge(CodeEdgeV4("e_calc", "model_node", "final_node", script=parse_result))
    graph.validate()

    executor = ExecutorV4(graph=graph, store=store)
    res = await executor.run()

    print("Final content:", store.get_vertex(session_id, "final_node").content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. Core Features

### 1. Fan-In Merge Strategies

When multiple upstream edges converge on a single vertex, use `MergeStrategyV4` to specify data merging behavior:

```python
from framework import MergeStrategyV4

# Overwrite (default)
store.apply_merge_strategy(sess, "target", incoming, strategy=MergeStrategyV4.OVERWRITE)

# Merge JSON dictionaries: {**existing, **incoming}
store.apply_merge_strategy(sess, "target", '{"b": 2}', strategy=MergeStrategyV4.JSON_MERGE)

# Append to JSON list: [item1, item2, ...]
store.apply_merge_strategy(sess, "target", '"new_item"', strategy=MergeStrategyV4.LIST_APPEND)

# Custom reducer function
def custom_reducer(existing: str, incoming: str) -> str:
    return f"{existing},{incoming}".strip(",")

store.apply_merge_strategy(sess, "target", "item", strategy=MergeStrategyV4.REDUCER_SCRIPT, reducer_fn=custom_reducer)
```

---

### 2. Error Recovery and Circuit Breaking (ReflexiveEdge)

When a vertex execution fails, it enters `REJECT`. A reflexive edge triggers self-loop recovery:

```python
from framework import ReflexiveEdgeV4, VertexStateV4

def recovery_fn(old_content, settings, staging):
    return f"{old_content} (retry with simplified query)"

reflexive_edge = ReflexiveEdgeV4(
    edge_id="e_retry",
    vertex_name="worker_node",
    trigger_state=VertexStateV4.REJECT.value,
    target_state=VertexStateV4.TODO_URGENT.value,
    max_retries=3,
    script=recovery_fn
)
```

If retry count exceeds `max_retries`, the vertex is locked into `FORBIDDEN`.

---

### 3. Nested Subgraphs

Tag a vertex with the `subgraph` attribute and supply the child graph manifest or directory path in `content`:

```json
{
  "name": "subgraph_box",
  "content": {
    "subgraph_manifest": "child/child_graph.json"
  },
  "attributes": ["subgraph"],
  "state": "todo"
}
```

During execution, `SSEExecutorV4` resolves the nested graph and attaches execution bridge edges:
- Data from the parent vertex routes to the child vertex with the `start` attribute.
- Internal child edges execute in topological DAG order.
- Data from the child vertex with the `end` attribute returns to the parent vertex.

Runnable example: [examples/subgraph_v4/](examples/subgraph_v4/):
```bash
python3 examples/subgraph_v4/run.py
```

---

## 5. Server & Web Dashboard

### 1. Launch API Server

```bash
uvicorn framework.server_v4:app --host 0.0.0.0 --port 8000
```

### 2. Web Dashboard

Navigate to `http://localhost:8000/dashboard` in a browser.
- Real-time visualization of DAG topology and vertex state highlights.
- Inspect vertex data, states, and execution metrics.

### 3. API Invocation

- **Execute session workflow**:
  ```bash
  curl -X POST http://localhost:8000/api/sessions/{session_id}/run \
       -H "Content-Type: application/json" \
       -d '{"max_concurrency": 4}'
  ```

- **SSE stream execution**:
  ```bash
  curl -N http://localhost:8000/api/sse/execute \
       -H "Content-Type: application/json" \
       -d '{"session_id": "sess_1", "manifest_path": "workflow/graph.json"}'
  ```

---

## 6. Testing

```bash
# Run offline test suite
python -m pytest tests/ -v -m "not live"

# Run full test suite
python -m pytest tests/ -v
```

---

## 7. Documentation Archive

Historical specifications, review records, and troubleshooting reports are located in [docs/archive/](docs/archive/).
