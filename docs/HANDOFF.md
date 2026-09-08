# Vertex-Edge Agent Framework V4.0 System Handoff

## 1. Executive Summary

This document serves as the official system handoff and technical specification for **Vertex-Edge Agent Framework V4.0**.
The codebase has undergone complete architectural hardening, multi-source fan-in consolidation, priority-based tiered DAG scheduling, path traversal security enforcement, distributed worker queue abstractions, nested subgraph execution, and modular JSON / directory batch loading.

- **Active Branch**: `vertex-edge-agent`
- **Execution Environment**: `Python 3.12+` / Linux
- **Automated Test Suite**: **526 automated tests passing with 0 failures** (covering system E2E, server security defenses, executor scheduling concurrency, subgraph integration, and legacy regression suites).

---

## 2. Core Architecture & Design Tenets

The V4 architecture provides a deterministic, high-throughput, production-grade agent orchestration framework:

```
+-------------------------------------------------------------------------+
|                        FastAPI Gateway / WebUI                          |
|    - Session Isolation (SessionGraphManagerV4)                          |
|    - Path Traversal Security & JSON Whitelist Validation                |
|    - Server-Sent Events (SSE) & OpenAI Tool-Call Echo Format            |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                          Execution Layer                                |
|    - SSEExecutorV4: Dynamic Subgraph Resolution & Proxy Bridge Edges    |
|    - ExecutorV4: Event-Driven Waking, Tiered DAG Priority Scheduling     |
|    - Concurrency Controls: Semaphores, Concurrency Groups, Deduping     |
|    - BaseWorkerQueueV4: Distributed Edge Worker Adapter Protocol        |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                           Data & Store Layer                            |
|    - VertexStoreV4: SQLite Engine, Strict Two-Sided Handshake           |
|    - MergeStrategyV4: Fan-In (Overwrite / JSON Merge / List Append)     |
|    - Fault Tolerance: Error REJECT -> Reflexive Retry -> Circuit Break  |
+-------------------------------------------------------------------------+
```

### Foundational Principles:
1. **Persistent Vertex State Machine**: All vertex states reside in SQLite (`data ready`, `idle`, `todo`, `todo urgent`, `reject`, `forbidden`, `pruning`). In-flight transient states are tracked in memory without holding database row locks, ensuring crash-resilience.
2. **Strict Two-Sided Handshake**: Forward edges fire if and only if upstream is `data ready` and downstream is `todo` or `todo urgent`. On completion, content updates and target state transitions occur atomically.
3. **Reflexive Self-Healing & Circuit Breaking**: Edge execution failures transition downstream to `reject`. Reflexive edges (`input_vertex == output_vertex`) read diagnostic staging, execute optional recovery logic, and increment `processed_count`. Vertices exceeding `max_retries` lock into `forbidden`.
4. **Event-Driven Tiered Scheduling**: Replaces polling loops with `asyncio.Event` synchronization combined with task wait timeouts. Tasks are prioritized by edge type (`reflexive` = 0, `todo urgent` = 1, `todo` = 2) and DAG topological tiers.
5. **Fan-In Accumulation Barrier**: Coordinates multiple upstream edges writing into a shared downstream vertex. Supports configurable merge strategies: `overwrite`, `json_merge`, `list_append`, and `reducer_script`.
6. **Hierarchical Subgraphs**: Container vertices with the `subgraph` attribute are resolved dynamically by `SSEExecutorV4`, automatically constructing proxy output vertices and execution bridges.
7. **Modular Discrete Configuration Loading**: Graph definitions support master manifest JSON files, discrete individual JSON file paths, and one-step directory batch loading.

---

## 3. Core Modules & Key Files

### Core Framework Modules (`framework/`)
- [`framework/vertex_v4.py`](file:///home/luminovoez/agent/framework/vertex_v4.py): SQLite storage engine, defining `VertexStoreV4`, `VertexRecordV4`, `VertexStateV4`, `VertexAttributeV4`, and `MergeStrategyV4`.
- [`framework/edge_v4.py`](file:///home/luminovoez/agent/framework/edge_v4.py): Base class `EdgeV4`, alongside `CodeEdgeV4`, `LLMEdgeV4`, and `ReflexiveEdgeV4` implementations with flexible callable arity handling.
- [`framework/sensenova_edge_v4.py`](file:///home/luminovoez/agent/framework/sensenova_edge_v4.py): Remote SenseNova 6.8 Flash Lite model edge implementation.
- [`framework/graph_v4.py`](file:///home/luminovoez/agent/framework/graph_v4.py): `GraphV4` topological validation (Kahn algorithm), DAG tiering, and `DiscreteGraphLoaderV4` (supporting discrete JSON files and directory loading).
- [`framework/executor_v4.py`](file:///home/luminovoez/agent/framework/executor_v4.py): Event-driven scheduler with fan-in accumulation barrier, concurrency groups, and task deduplication.
- [`framework/sse_executor_v4.py`](file:///home/luminovoez/agent/framework/sse_executor_v4.py): Session routing, recursive subgraph resolution, proxy bridge construction, and OpenAI Tool-Call Echo SSE formatting.
- [`framework/worker_queue_v4.py`](file:///home/luminovoez/agent/framework/worker_queue_v4.py): Abstract distributed worker adapter interface `BaseWorkerQueueV4` and reference `InMemoryWorkerQueueV4`.
- [`framework/server_v4.py`](file:///home/luminovoez/agent/framework/server_v4.py): FastAPI application, `SessionGraphManagerV4`, REST APIs, and path traversal defense.
- [`framework/templates/dashboard.html`](file:///home/luminovoez/agent/framework/templates/dashboard.html): Interactive DAG visualization and node inspector dashboard.

### Example Projects (`examples/`)
- [`examples/subgraph_v4/`](file:///home/luminovoez/agent/examples/subgraph_v4/): Complete nested subgraph, discrete JSON path, and directory batch loading demo.
  - [`examples/subgraph_v4/run.py`](file:///home/luminovoez/agent/examples/subgraph_v4/run.py): Runnable script showcasing 3 loading and execution methods.
  - [`examples/subgraph_v4/child/`](file:///home/luminovoez/agent/examples/subgraph_v4/child/): Child subgraph manifest, normalization script, and enrichment script.
  - [`examples/subgraph_v4/discrete_dir_demo/`](file:///home/luminovoez/agent/examples/subgraph_v4/discrete_dir_demo/): Directory batch loading structure demo.
- [`examples/sensenova_v4/`](file:///home/luminovoez/agent/examples/sensenova_v4/): SenseNova model inference edge demo.

### Test Suites (`tests/`)
- [`tests/test_v4_subgraph_and_discrete_loader.py`](file:///home/luminovoez/agent/tests/test_v4_subgraph_and_discrete_loader.py): Integration tests for path-based loading, directory loading, and nested subgraph execution.
- [`tests/test_v4_layer4_executor.py`](file:///home/luminovoez/agent/tests/test_v4_layer4_executor.py): Concurrency limits, priority queues, and fan-in merge barrier tests.
- [`tests/test_v4_layer5_server.py`](file:///home/luminovoez/agent/tests/test_v4_layer5_server.py): Path traversal security attacks, Pydantic validation, and JSON extension whitelist tests.
- [`tests/test_v4_layer6_e2e.py`](file:///home/luminovoez/agent/tests/test_v4_layer6_e2e.py): Multi-step pipeline, deep recursive subgraphs, and worker queue interface tests.
- [`tests/test_v4_system.py`](file:///home/luminovoez/agent/tests/test_v4_system.py): Core workflow execution and reflexive self-loop recovery tests.
- [`tests/test_v4_server.py`](file:///home/luminovoez/agent/tests/test_v4_server.py): API routing, node inspector, and SSE streaming tests.

---

## 4. Key Patterns & Common Usages

### 4.1 Loading via Discrete JSON Paths or Directories
```python
from framework import GraphV4

graph = GraphV4(session_id="my_session")

# 1. Load vertices and edges by JSON file path
graph.add_vertex("examples/subgraph_v4/parent_in.json")
graph.add_edge("examples/subgraph_v4/e_start_to_subgraph.json")

# 2. Batch load entire directory
dir_graph = GraphV4.from_directory("examples/subgraph_v4/discrete_dir_demo")
```

### 4.2 Subgraph Orchestration
Define a container vertex in the parent graph:
```json
{
  "name": "enrichment_box",
  "attributes": ["subgraph"],
  "state": "todo",
  "content": {
    "subgraph_manifest": "child/child_graph.json"
  }
}
```
Execute with `SSEExecutorV4`:
```python
from framework import SSEExecutorV4, SessionGraphManagerV4, VertexStoreV4

store = VertexStoreV4(":memory:")
manager = SessionGraphManagerV4(store)
executor = SSEExecutorV4(manager=manager, store=store, default_manifest="parent_graph.json")
result = await executor.execute_harness_call()
```

### 4.3 Multi-Source Fan-In Accumulation
```python
from framework import MergeStrategyV4

# Supported strategies: OVERWRITE, JSON_MERGE, LIST_APPEND, REDUCER_SCRIPT
store.apply_merge_strategy(session_id, "output_node", incoming_json, strategy=MergeStrategyV4.JSON_MERGE)
```

---

## 5. Verification & Commands

### Run Full Test Suite
```bash
# Offline test suite (526 passed)
.venv/bin/python -m pytest tests/ -v -m "not live"

# Subgraph and discrete loader tests
.venv/bin/python -m pytest tests/test_v4_subgraph_and_discrete_loader.py -v
```

### Run Demos
```bash
# Subgraph and discrete loading demo
python3 examples/subgraph_v4/run.py

# Remote model inference demo (requires SENSENOVA_API_KEY)
python3 examples/sensenova_v4/demo.py
```

### Start Server & Web Dashboard
```bash
uvicorn framework.server_v4:app --host 0.0.0.0 --port 8000
# Browser UI: http://localhost:8000/dashboard
```

---

## 6. Maintenance & Future Roadmap

1. **Distributed Worker Backend**: Implement `BaseWorkerQueueV4` in [`framework/worker_queue_v4.py`](file:///home/luminovoez/agent/framework/worker_queue_v4.py) for external distributed message brokers (e.g. Celery or Redis Queue).
2. **Documentation Integrity**:
   - Keep root `README.md` strictly as a user guide without comparison tables or legacy essays.
   - Maintain historical reports and architecture drafts in [`docs/archive/`](file:///home/luminovoez/agent/docs/archive/).
