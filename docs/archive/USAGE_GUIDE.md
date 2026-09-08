# Vertex-Edge Agent Framework Comprehensive Guide (Archived)

> **Document Status**: Archived. For the current streamlined usage manual, refer to root [`README.md`](../../README.md).

---

## Table of Contents
- [1. Architecture Comparison: V4.0 vs Legacy](#1-architecture-comparison-v40-vs-legacy)
- [2. V4.0 Core Concepts & Data Models](#2-v40-core-concepts--data-models)
- [3. Code-Level Orchestration Guide](#3-code-level-orchestration-guide)
  - [1. Basic Linear Pipeline (CodeEdgeV4)](#1-basic-linear-pipeline-codeedgev4)
  - [2. Remote Model Inference Pipeline (SenseNovaEdgeV4)](#2-remote-model-inference-pipeline-sensenovaedgev4)
  - [3. Fan-In Merge Strategies](#3-fan-in-merge-strategies)
  - [4. Fault Tolerance & Self-Healing (ReflexiveEdgeV4)](#4-fault-tolerance--self-healing-reflexiveedgev4)
- [4. Service Mode: Web Dashboard & HTTP/SSE API](#4-service-mode-web-dashboard--httpsse-api)
  - [1. Start Service](#1-start-service)
  - [2. Visual Web Dashboard](#2-visual-web-dashboard)
  - [3. API Endpoints & Usage](#3-api-endpoints--usage)
- [5. Distributed Worker Queue Adapter (BaseWorkerQueueV4)](#5-distributed-worker-queue-adapter-baseworkerqueuev4)
- [6. Legacy (V1-V3) Compatibility Guide](#6-legacy-v1-v3-compatibility-guide)
- [7. Automated Testing](#7-automated-testing)

---

## 1. Architecture Comparison: V4.0 vs Legacy

| Dimension | Legacy (V1-V3) | V4.0 |
|---|---|---|
| **Status** | Maintained for backward compatibility | Recommended for production |
| **Storage Engine** | Ephemeral in-memory dicts | SQLite dual-table transactional persistence with WAL |
| **Scheduling** | Polling counter loop | `asyncio.Event` sub-millisecond reactive event waking |
| **Fan-In Merge** | Overwrite only, no barrier | `MergeStrategyV4` (JSON merge / list append / reducer) + settlement barrier |
| **Fault Tolerance** | Manual exception catching | `ReflexiveEdgeV4` declarative self-healing loop and circuit breaking |
| **Model Integration** | Manual mock or HttpLLMAgent setup | Built-in `SenseNovaEdgeV4` remote inference support |
| **User Interface** | Python script execution only | Complete FastAPI REST API, SSE streaming, standalone Web dashboard |

---

## 2. V4.0 Core Concepts & Data Models

V4.0 enforces a strict **two-sided state handshake protocol**:

```
[ Vertex A: data ready ] ---> ( Edge: computation / inference ) ---> [ Vertex B: todo -> data ready ]
```

1. **Vertex States (`VertexStateV4`)**:
   - `data ready`: Data has been produced and is ready for downstream consumption.
   - `todo` / `todo urgent`: Signal indicating downstream node is awaiting data.
   - `reject`: Calculation failure, triggering reflexive self-healing edges.
   - `forbidden`: Circuit-broken permanently after exceeding retry limits.

2. **Vertex Attributes (`VertexAttributeV4`)**:
   - `start`: Graph input entry point.
   - `end`: Graph terminal convergence point.
   - `json`: Enforces JSON parsing and markdown fence stripping on output.
   - `subgraph`: Marks the vertex as a nested child graph container.

3. **Edge Types (`EdgeV4`)**:
   - `CodeEdgeV4`: Executes local synchronous or asynchronous Python functions.
   - `SenseNovaEdgeV4`: Native direct connection to remote SenseNova 6.8 models.
   - `LLMEdgeV4`: Adapter for custom agent protocols.
   - `ReflexiveEdgeV4`: Self-loop edge capturing `reject` and resetting to `todo urgent`.

---

## 3. Code-Level Orchestration Guide

### 1. Basic Linear Pipeline (CodeEdgeV4)

```python
import asyncio
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, CodeEdgeV4
)

async def main():
    # 1. Initialize SQLite store
    store = VertexStoreV4(":memory:")
    session_id = "demo_linear"
    graph = GraphV4(session_id=session_id)

    # 2. Add vertices
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="v_start",
        content="hello world",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="v_mid",
        content="",
        state=VertexStateV4.TODO.value
    ))
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="v_end",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    # 3. Define transform functions
    def step1_upper(content, settings, staging):
        return f"[{content.upper()}]"

    def step2_sign(content, settings, staging):
        return f"{content} processed by vea-v4"

    graph.add_edge(CodeEdgeV4("e1", "v_start", "v_mid", script=step1_upper))
    graph.add_edge(CodeEdgeV4("e2", "v_mid", "v_end", script=step2_sign))
    graph.validate()

    # 4. Execute
    executor = ExecutorV4(graph=graph, store=store)
    result = await executor.run()

    print("Result:", store.get_vertex(session_id, "v_end").content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 2. Remote Model Inference Pipeline (SenseNovaEdgeV4)

```python
import asyncio
import json
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, SenseNovaEdgeV4, CodeEdgeV4
)

async def main():
    store = VertexStoreV4(":memory:")
    session_id = "demo_llm"
    graph = GraphV4(session_id=session_id)

    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="v_prompt",
        content='Calculate 25 * 4, return strictly JSON: {"question": "25*4", "answer": 100}',
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="v_model",
        content="",
        attributes=[VertexAttributeV4.JSON.value],
        state=VertexStateV4.TODO.value
    ))
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="v_out",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    graph.add_edge(SenseNovaEdgeV4(
        edge_id="e_model",
        input_vertex="v_prompt",
        output_vertex="v_model",
        settings={"temperature": 0.1}
    ))

    def parse_fn(content, settings, staging):
        data = json.loads(content)
        return f"Calculated answer: {data.get('answer')}"

    graph.add_edge(CodeEdgeV4("e_extract", "v_model", "v_out", script=parse_fn))
    graph.validate()

    executor = ExecutorV4(graph=graph, store=store)
    await executor.run()

    print("Output:", store.get_vertex(session_id, "v_out").content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### 3. Fan-In Merge Strategies

```python
# Supported merge strategies:
# 1. "merge_strategy": "json_merge"   -> Dict merge {**existing, **incoming}
# 2. "merge_strategy": "list_append"  -> List append [item1, item2, ...]
# 3. "merge_strategy": "overwrite"    -> Overwrite mode (default)

graph.add_edge(CodeEdgeV4(
    edge_id="e_merge_a",
    input_vertex="v_in_a",
    output_vertex="v_converge",
    script=lambda c, s, st: json.dumps({"module_a": "ok"}),
    settings={"merge_strategy": "json_merge"}
))
graph.add_edge(CodeEdgeV4(
    edge_id="e_merge_b",
    input_vertex="v_in_b",
    output_vertex="v_converge",
    script=lambda c, s, st: json.dumps({"module_b": "ok"}),
    settings={"merge_strategy": "json_merge"}
))
```

---

### 4. Fault Tolerance & Self-Healing (ReflexiveEdgeV4)

```python
reflexive_edge = ReflexiveEdgeV4(
    edge_id="e_retry",
    vertex_name="worker_node",
    trigger_state="reject",
    target_state="todo urgent",
    max_retries=3,
    script=recovery_fn
)
```

---

## 4. Service Mode: Web Dashboard & HTTP/SSE API

### 1. Start Service
```bash
uvicorn framework.server_v4:app --host 0.0.0.0 --port 8000
```

### 2. Web Dashboard
Navigate to `http://127.0.0.1:8000/dashboard` in a browser.

### 3. API Endpoints
- Run session: `POST /api/sessions/{session_id}/run`
- SSE stream: `GET /api/sse/execute`
- Dump graph: `POST /api/sessions/{session_id}/graph/dump`

---

## 5. Distributed Worker Queue Adapter (BaseWorkerQueueV4)

Defined in [`framework/worker_queue_v4.py`](framework/worker_queue_v4.py):
- `submit_edge(task: EdgeTaskPayload) -> str`
- `poll_result(task_id: str) -> Optional[EdgeTaskResult]`
- `wait_result(task_id: str, timeout: float) -> EdgeTaskResult`
- `cancel_task(task_id: str) -> bool`
- `health_check() -> bool`

---

## 6. Legacy (V1-V3) Compatibility Guide

```python
from framework import Vertex, Edge, Graph, Executor

v1 = Vertex("v1", initial_data={"val": 10})
v2 = Vertex("v2")
e = Edge("e1", source="v1", destination="v2", script=lambda val: val * 2)

graph = Graph("legacy_graph")
graph.add_vertex(v1)
graph.add_vertex(v2)
graph.add_edge(e)

executor = Executor(graph)
executor.run()
```

---

## 7. Automated Testing

```bash
# Run all tests
python -m pytest tests/ -v -m "not live"
```
