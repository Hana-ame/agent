# Vertex-Edge Agent Framework V4.0 System Handoff

## 1. Executive Summary

This document serves as the official system handoff and technical specification for **Vertex-Edge Agent Framework V4.0**.
The codebase provides a production-grade, data-driven agent orchestration framework where **vertices** store persistent states and data, and **edges** encapsulate transformation logic, tool calls, and model inference.

- **Active Branch**: `vertex-edge-agent`
- **Execution Environment**: `Python 3.12+` / Linux
- **Default Server Port**: `11434` (Ollama-compatible standard port)
- **Automated Test Suite**: **551 automated tests passing with 0 failures (100% pass rate)**, covering E2E workflows, server security defenses, executor scheduling concurrency, Agent Harness HTTP integration, tool edges, and edge-level metrics benchmarking.

---

## 2. Core Architecture & Design Tenets

```
+-------------------------------------------------------------------------+
|                    FastAPI Server Gateway (Port 11434)                  |
|  - Native OpenAI Endpoints: /v1/chat/completions, /v1/models            |
|  - Session Isolation (SessionGraphManagerV4)                            |
|  - REST & Mutation APIs: Vertices, Edges, Subgraphs, Reentry, Clear     |
|  - Performance & Benchmarking APIs: /api/sessions/{id}/metrics          |
|  - Live Visual Dashboard: /dashboard (Topology, Node Inspector)         |
+-------------------------------------------------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+------------------------------------+   +------------------------------------+
|     Passive HTTP Harness Engine    |   |     Standalone / Batch Engine      |
|      (HttpHarnessExecutorV4)       |   |            (ExecutorV4)            |
|  - Request-driven (no while-loops) |   |  - Event-driven (asyncio.Event)    |
|  - Pure 2-sided handshake ranking  |   |  - Multi-level concurrency semas   |
|  - ToolEdge & LLMToolEdge yields   |   |  - Multi-source fan-in barrier     |
|  - Ingests role: "tool" sandboxes  |   |  - Worker Queue abstraction        |
|  - message.content observability   |   |  - Reflexive retry & circuit break |
+------------------------------------+   +------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                       Data & Persistence Layer                          |
|  - VertexStoreV4 (SQLite3 engine with WAL mode and row concurrency)    |
|  - vertices: Key-indexed persistent state machine                       |
|  - edges: Declarative edge configurations and settings                  |
|  - session_staging: Isolated intermediate scratchpad & diagnostics      |
|  - edge_metrics: Microsecond latency, token usage, cost, error logging  |
|  - MergeStrategyV4: Fan-In (overwrite, json_merge, list_append, reducer)|
+-------------------------------------------------------------------------+
```

### Foundational Principles:
1. **Persistent Vertex State Machine**: All vertex states reside in SQLite (`data ready`, `idle`, `todo`, `todo urgent`, `reject`, `forbidden`, `pruning`). In-flight transient states are tracked in memory without holding database row locks, ensuring crash-resilience.
2. **Strict Two-Sided Handshake**: Forward edges fire if and only if upstream is `data ready` and downstream is `todo` or `todo urgent`. On completion, content updates and target state transitions occur atomically.
3. **Agent Harness Clock-Tick Model**: For HTTP integrations (`HttpHarnessExecutorV4`), the server does not run an internal infinite loop. Every incoming HTTP request advances the graph by one step. Edges yield real environment tool calls (e.g. `bash(command="...")`) to the external harness sandbox, which executes them and returns results via `role: "tool"`.
4. **No Tier Concurrency Conflicts**: Scheduling is strictly determined by urgency rank (`reflexive recovery` = 0, `todo urgent` = 1, `todo` = 2), edge priority descending, and edge ID ascending. DAG topological tiers are not used as blocking batches in sequential request-driven flows.
5. **Edge-Level Benchmarking & Metrics**: Every edge execution (local script, LLM inference, or harness sandbox tool execution) records latency, prompt/completion tokens, costs, success status, and error traces into the `edge_metrics` SQLite table.
6. **Reflexive Self-Healing & Circuit Breaking**: Edge execution failures transition downstream vertices to `reject`. Reflexive edges (`input_vertex == output_vertex`) read diagnostic staging, execute recovery logic, and increment `processed_count`. Vertices exceeding `max_retries` lock into `forbidden`.
7. **Fan-In Accumulation Barrier**: Coordinates multiple upstream edges writing into a shared downstream vertex. Supports configurable merge strategies: `overwrite`, `json_merge`, `list_append`, and `reducer_script`.
8. **Online Graph Mutation & Dynamic Reentry**: Graphs in SQLite can be mutated live (add/reconnect edges, add vertices, splice/insert subgraphs, or trigger replay/reentry of existing vertices with automated downstream invalidation).

---

## 3. Core Modules & Key Files

### Framework Source (`framework/`)
- [`framework/vertex_v4.py`](../framework/vertex_v4.py): SQLite storage engine defining `VertexStoreV4`, `VertexRecordV4`, `EdgeRecordV4`, `StagingRecordV4`, `EdgeMetricRecordV4`, `VertexStateV4`, `VertexAttributeV4`, and `MergeStrategyV4`.
- [`framework/edge_v4.py`](../framework/edge_v4.py): Base class `EdgeV4`, `CodeEdgeV4`, `ToolEdgeV4` (declarative tool calls), `LLMToolEdgeV4` (dynamic LLM tool calls), `LLMEdgeV4`, and `ReflexiveEdgeV4`.
- [`framework/http_executor_v4.py`](../framework/http_executor_v4.py): Request-driven `HttpHarnessExecutorV4` powering passive graph stepping, tool call yielding, tool output ingestion, and observability reporting in `message.content`.
- [`framework/executor_v4.py`](../framework/executor_v4.py): Standalone `ExecutorV4` scheduler with fan-in accumulation barrier, concurrency groups, edge-level metrics instrumentation, and task deduplication.
- [`framework/server_v4.py`](../framework/server_v4.py): FastAPI application running on default port `11434`, implementing native `/v1/chat/completions`, `/v1/models`, REST graph mutation APIs, performance metric endpoints, and `/dashboard`.
- [`framework/graph_v4.py`](../framework/graph_v4.py): `GraphV4` topology management, validation (Kahn algorithm), and `DiscreteGraphLoaderV4` (discrete JSON files and directory loading).
- [`framework/sse_executor_v4.py`](../framework/sse_executor_v4.py): Session routing, recursive subgraph resolution, and OpenAI Tool-Call Echo SSE streaming.
- [`framework/sensenova_edge_v4.py`](../framework/sensenova_edge_v4.py): Remote SenseNova 6.8 Flash Lite model edge implementation.
- [`framework/worker_queue_v4.py`](../framework/worker_queue_v4.py): Abstract distributed worker adapter interface `BaseWorkerQueueV4` and reference `InMemoryWorkerQueueV4`.
- [`framework/templates/dashboard.html`](../framework/templates/dashboard.html): Interactive DAG visualizer and SQLite inspector UI.

### Test Suites (`tests/`)
- [`tests/test_v4_edge_metrics.py`](../tests/test_v4_edge_metrics.py): Edge metric persistence, latency/token aggregation, error rates, and REST endpoints.
- [`tests/test_harness_http_executor.py`](../tests/test_harness_http_executor.py): Request-driven `HttpHarnessExecutorV4` stepping, declarative tool calls, and sandbox tool settling.
- [`tests/test_openai_v4_endpoints.py`](../tests/test_openai_v4_endpoints.py): OpenAI `/v1/models` and `/v1/chat/completions` compatibility across full, hop-by-hop, streaming, and tool execution modes.
- [`tests/test_v4_subgraph_and_discrete_loader.py`](../tests/test_v4_subgraph_and_discrete_loader.py): Modular JSON loading, directory batch loading, and nested subgraph execution.
- [`tests/test_v4_layer4_executor.py`](../tests/test_v4_layer4_executor.py): Concurrency limits, priority queues, and fan-in merge barriers.
- [`tests/test_v4_layer5_server.py`](../tests/test_v4_layer5_server.py): Security defenses, path traversal protection, and Pydantic validation.
- [`tests/test_v4_layer6_e2e.py`](../tests/test_v4_layer6_e2e.py): Deep recursive subgraphs, multi-step pipelines, and worker queues.
- [`tests/test_v4_system.py`](../tests/test_v4_system.py): Core workflow execution and reflexive self-loop recovery.
- [`tests/test_v4_server.py`](../tests/test_v4_server.py): Online graph mutation APIs, vertex reentry, and SSE streaming.

---

## 4. Key Patterns & Common Usages

### 4.1 OpenAI Compatibility & Agent Harness Integration

The server provides a standard `/v1/chat/completions` endpoint on port `11434`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

# 1. Start a workflow turn
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Process task input"}],
)

# 2. If the graph encountered a ToolEdgeV4, it yields finish_reason="tool_calls":
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    func_name = tool_call.function.name       # e.g. "bash"
    func_args = tool_call.function.arguments  # e.g. '{"command": "ls -la"}'
    
    # 3. External harness runs the command in its sandbox and posts back:
    sandbox_output = "file1.txt  file2.txt"
    next_response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "user", "content": "Process task input"},
            response.choices[0].message,
            {"role": "tool", "tool_call_id": tool_call.id, "content": sandbox_output},
        ],
    )
```

### 4.2 Declarative Tool Edge (`ToolEdgeV4`)

Define edges that trigger environment tools directly in an external agent harness:

```python
from framework import ToolEdgeV4

# Static command
tool_edge = ToolEdgeV4(
    edge_id="e_run_tests",
    input_vertex="repo_ready",
    output_vertex="test_results",
    tool_name="bash",
    arguments={"command": "pytest tests/"},
)

# Templated command using input vertex content
templated_edge = ToolEdgeV4(
    edge_id="e_cat_file",
    input_vertex="filename_node",
    output_vertex="file_content_node",
    tool_name="bash",
    arguments_template='{"command": "cat {input}"}',
)
```

### 4.3 Edge Performance Metrics & Benchmarking

Every edge automatically records execution telemetry in SQLite:

```python
from framework import VertexStoreV4

store = VertexStoreV4("agent_data.db")

# Query metrics list
metrics = store.list_edge_metrics(session_id="my_sess")
for m in metrics:
    print(f"Edge {m.edge_id} ({m.edge_type}): {m.execution_time_ms:.2f}ms, {m.total_tokens} tokens")

# Query aggregate performance summary
summary = store.get_edge_metrics_summary(session_id="my_sess")
print("Total Executions:", summary["total_executions"])
print("Error Rate:", summary["error_rate"])
print("Avg Latency (ms):", summary["avg_execution_time_ms"])
print("Per-edge breakdown:", summary["by_edge"])
```

### 4.4 Online Graph Mutation & Dynamic Reentry

Mutate vertices and edges on the fly through REST endpoints on port 11434:
- `POST /api/sessions/{session_id}/vertices`: Dynamically add a vertex.
- `POST /api/sessions/{session_id}/edges`: Dynamically register a new edge.
- `POST /api/sessions/{session_id}/edges/{edge_id}/reconnect`: Reconnect an edge to new input/output endpoints.
- `POST /api/sessions/{session_id}/vertices/{vertex_name}/reenter`: Re-execute from a specific vertex, automatically resetting its downstream nodes.
- `GET /api/sessions/{session_id}/metrics`: Query performance benchmarks and execution timing.
- `GET /api/metrics/summary`: Global benchmarking summary across all sessions.

---

## 5. Verification & Commands

### Run Full Test Suite
```bash
# Run all 551 tests
pytest

# Run OpenAI endpoint & Harness integration tests
pytest tests/test_openai_v4_endpoints.py tests/test_harness_http_executor.py -v

# Run Edge Metrics tests
pytest tests/test_v4_edge_metrics.py -v
```

### Launch Server
```bash
# Launch server on default port 11434
python3 -m framework.server_v4 --port 11434

# Open Web Dashboard
http://localhost:11434/dashboard
```

---

## 6. Maintenance & Best Practices

1. **Harness Integrations**: Never reintroduce DAG `Tier` batching into `HttpHarnessExecutorV4`. The agent harness drives the execution sequentially via incoming requests.
2. **Backward Compatibility**: `framework/serve/app.py` remains 100% untouched for legacy V1 callers. All new features reside in `server_v4.py` and the V4 framework modules.
3. **Storage Cleanliness**: Always use `store.clear_session(session_id)` to clean up vertices, staging scratchpads, and edge metrics atomically.
