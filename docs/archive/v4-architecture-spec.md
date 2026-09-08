# ⚡ Vertex-Edge Framework v4.0: Complete Destructive Architecture Specification

> **Refactoring Directive**: Complete destructive overhaul of the Vertex, Edge, Graph, and Execution systems. All v4 modules reside flatly within the `framework/` directory (`vertex_v4.py`, `edge_v4.py`, `graph_v4.py`, `executor_v4.py`, `server_v4.py`, `sse_executor_v4.py`).

---

## 1. Executive Summary & Core Architectural Tenets

### 1.1 The v4.0 Paradigm Shift
1. **Pure SQLite3 Key-Indexed Storage (`framework/vertex_v4.py`)**:
   - A Vertex is no longer an active in-memory object; it is a keyed row in an ACID SQLite3 table uniquely identified by `(session_id, name)`.
   - Fields: auto-incrementing `id`, `session_id`, vertex `name`, `content`, multiple `attributes`, strict lifecycle `state`, and `processed_count`.
2. **Dedicated Session Staging Table (`session_staging`)**:
   - Intermediate drafts, validation traces, and LLM reasoning steps reside in an isolated `session_staging` table.
   - Explicit attribution tracks the producing `edge_id` alongside timestamps and metadata.
3. **Standalone 2-Vertex Edge Execution (`framework/edge_v4.py`)**:
   - Edges are standalone units runnable from CLI or in Python without the full executor loop.
   - Binds strictly to two vertices: `(session_id, input_vertex_name)` and `(session_id, output_vertex_name)`.
   - Supports Python code transformations and LLM foundation models.
   - **Two-Sided Handshake Contract**:
     $$\text{upstream.state} = \text{"data ready"} \quad \land \quad \text{downstream.state} \in \{\text{"todo"}, \text{"todo urgent"}\}$$
   - **No `running` state in database**: In-flight task tracking is deduplicated in memory via `active_dispatches: Set[Tuple[str, str, str]]`.
4. **Reflexive Self-Loop Recovery Edges**:
   - Edges where `input_vertex == output_vertex`.
   - Triggered on `reject`, inspects staging diagnostics, applies recovery, resets downstream to `todo urgent`, and increments `processed_count`.
5. **Discrete & Modular Graph Manifests (`framework/graph_v4.py`)**:
   - Master `graph.json` contains only paths to discrete `vertex.json` and `edge.json` files.
6. **DAG & Concurrency-Driven Executor (`framework/executor_v4.py`)**:
   - Directly schedules edges according to `max_concurrency` semaphore and topological DAG tier order.
7. **Online Graph Mutation API & Standalone Database Dashboard (`framework/server_v4.py`)**:
   - REST API to dynamically add, update, and delete vertices and edges per session.
   - Real-time Server-Sent Events (SSE) streaming live progress.
   - Standalone web dashboard to inspect raw SQLite tables (`vertices`, `session_staging`).
   - **Interactive Selection & Modification**: Clicking any vertex row inspects and edits its full content, state, attributes, and count; clicking any edge row inspects and edits its script, endpoints, and settings.
8. **Session-Aware SSEExecutor with Subgraphs & Harness Tool Call Echo (`framework/sse_executor_v4.py`)**:
   - Dynamic session routing: auto-creates new session graphs on demand.
   - Subgraph vertices: vertices with attribute `subgraph` load nested child graphs tracked under the session namespace.
   - Returns responses formatted as tool call echo `"info"` compatible with agent harness test frameworks.

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    graph.json (Manifest)                    │
                    │  - vertices: ["v_in.json", "v_out.json"]                    │
                    │  - edges:    ["e_compute.json", "e_recover.json"]           │
                    └─────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                      SQLite3 Database                       │
                    ├──────────────────────────────┬──────────────────────────────┤
                    │       Table: vertices        │    Table: session_staging    │
                    │  (id, session_id, name,      │  (id, session_id, edge_id,  │
                    │   content, attributes, state,│   vertex_name, key, value,   │
                    │   processed_count)           │   metadata, created_at)      │
                    └──────────────────────────────┴──────────────────────────────┘
                                   │                              ▲
        1. Two-Sided Handshake     │                              │ 2. Stage Diagnostics /
           - Upstream: data ready  │                              │    Deliver Content
           - Downstream: todo      │                              │
                                   ▼                              │
                    ┌─────────────────────────────────────────────┴───────────────┐
                    │               Standalone Executable EdgeV4                  │
                    │  Standard:  (session_id, "source") -> (session_id, "sink")  │
                    │  Reflexive: (session_id, "node")   -> (session_id, "node")  │
                    │                                                             │
                    │               [ Python Code | LLM Inference ]               │
                    └─────────────────────────────────────────────────────────────┘
```

---

## 2. Vertex Storage Engine (`framework/vertex_v4.py`)

### 2.1 Database Schema
```sql
CREATE TABLE IF NOT EXISTS vertices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    attributes TEXT NOT NULL DEFAULT '[]', -- JSON array of tags
    state TEXT NOT NULL DEFAULT 'idle' CHECK (
        state IN ('data ready', 'idle', 'forbidden', 'todo', 'todo urgent', 'reject', 'pruning')
    ),
    processed_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(session_id, name)
);

CREATE INDEX IF NOT EXISTS idx_vertices_session_state ON vertices(session_id, state);
CREATE INDEX IF NOT EXISTS idx_vertices_session_name ON vertices(session_id, name);
```

### 2.2 Field Definitions
| Field | Type | Description |
|---|---|---|
| `id` | `INTEGER` | Autoincrement primary key. |
| `session_id` | `TEXT` | Session / run boundary identifier. |
| `name` | `TEXT` | Semantic vertex name, unique within session. |
| `content` | `TEXT` | Raw data payload (string, JSON, or text). |
| `attributes` | `TEXT` (JSON) | Array of functional tags. |
| `state` | `TEXT` | Current execution state. |
| `processed_count` | `INTEGER` | Processing counter tracking executions and retries. |
| `created_at` | `TEXT` | ISO 8601 creation timestamp. |
| `updated_at` | `TEXT` | ISO 8601 update timestamp. |

### 2.3 Formal States (`VertexStateV4`)
- `data ready`: Upstream data is ready and verified.
- `idle`: Inactive/dormant. Forward edge will not trigger.
- `todo`: Standard priority execution demand.
- `todo urgent`: High priority execution demand.
- `reject`: Computation failed or rejected by validation.
- `pruning`: Branch abandoned by conditional routing.
- `forbidden`: Locked out by circuit breaker or policy.

### 2.4 Allowed Attributes (`VertexAttributeV4`)
- `start`: Workflow entry point.
- `end`: Workflow sink/terminal point.
- `llm result`: Content generated by LLM.
- `llm prompt`: Content formatted as model prompt.
- `json`: Content structured as JSON.
- `plain text`: Content stored as raw string.
- `subgraph`: Vertex encapsulates a child graph.

---

## 3. Session Staging Table (`session_staging`)

```sql
CREATE TABLE IF NOT EXISTS session_staging (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    edge_id TEXT NOT NULL,           -- Must record which edge produced this entry
    vertex_name TEXT,
    key TEXT NOT NULL,
    value TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_staging_session_edge ON session_staging(session_id, edge_id);
CREATE INDEX IF NOT EXISTS idx_staging_session_key ON session_staging(session_id, key);
CREATE INDEX IF NOT EXISTS idx_staging_session_vertex ON session_staging(session_id, vertex_name);
```

---

## 4. Standalone Edge Engine (`framework/edge_v4.py`)

### 4.1 Two-Sided Handshake Contract
An `EdgeV4` binds to exactly two vertex references within a session:
1. `(session_id, input_vertex)`
2. `(session_id, output_vertex)`

Execution precondition for forward edges:
$$\text{store.get\_vertex}(s, \text{in}).\text{state} = \text{"data ready"} \quad \land \quad \text{store.get\_vertex}(s, \text{out}).\text{state} \in \{\text{"todo"}, \text{"todo urgent"}\}$$

### 4.2 Reflexive Self-Loop Recovery (`input_vertex == output_vertex`)
- Trigger Condition: `target.state == "reject"`.
- Reads diagnostic error trace from `session_staging` via `get_latest_staged_for_vertex()`.
- Circuit Breaker: If `target.processed_count >= max_retries`, locks vertex to `forbidden`.
- Otherwise: increments `processed_count` and resets `target.state` back to `todo urgent`.

### 4.3 Standalone Execution (CLI & Python)
```bash
python3 -m framework.edge_v4 \
  --db workflow.db \
  --session session_001 \
  --edge-id e1 \
  --type code \
  --input v_in \
  --output v_out
```

---

## 5. Discrete Graph Manifest (`framework/graph_v4.py`)

### 5.1 Master Manifest: `graph.json`
```json
{
  "version": "4.0",
  "session_id": "sess_prod",
  "metadata": {"name": "Discrete Workflow"},
  "vertices": [
    "vertices/start.json",
    "vertices/subgraph_box.json",
    "vertices/end.json"
  ],
  "edges": [
    "edges/e_to_box.json",
    "edges/e_from_box.json",
    "edges/e_recovery.json"
  ]
}
```

### 5.2 Topological DAG Tiering
- `Tier -1`: Reflexive recovery edges (urgent recovery).
- `Tier 0`: Forward edges originating from root or start vertices.
- `Tier k`: Forward edges depending on Tier k-1 outputs.

---

## 6. DAG-Ordered Executor (`framework/executor_v4.py`)

- **Concurrency Control**: Bounded by `asyncio.Semaphore(max_concurrency)`.
- **Active Dispatch Deduplication**: Tracks in-flight edges in memory via `active_dispatches: Set[Tuple[str, str, str]]`. No `running` state written to SQLite.
- **Priority Scheduling**: Reflexive recovery edges (priority 0) > `todo urgent` (priority 1) > `todo` (priority 2), ordered by DAG topological tier.

---

## 7. Online Graph Mutation API & Server (`framework/server_v4.py`)

### 7.1 Online Mutation API
- `POST /api/sessions/{session_id}/graph/vertices`: Add/update vertex.
- `DELETE /api/sessions/{session_id}/graph/vertices/{name}`: Delete vertex.
- `POST /api/sessions/{session_id}/graph/edges`: Add/update edge (recalculates DAG tiers).
- `DELETE /api/sessions/{session_id}/graph/edges/{edge_id}`: Delete edge.
- `POST /api/sessions/{session_id}/graph/validate`: Check DAG validity.

### 7.2 Database Inspector API & Interactive Web Dashboard
- `GET /api/db/stats`: Global stats across sessions.
- `GET /api/db/sessions`: List distinct session IDs.
- `GET /api/db/sessions/{session_id}/vertices`: Query raw vertices table.
- `GET /api/db/sessions/{session_id}/staging`: Query raw staging table.
- `GET /dashboard` (or `GET /`): Interactive live web dashboard.
  - **Interactive Vertex Inspector**: Clicking any vertex row in the table loads its complete content, state, attributes, and execution count into an online form, permitting live editing and saving via the API.
  - **Interactive Edge Inspector**: Clicking any edge row loads its complete configuration, script path, connection pair, and JSON settings, permitting live reconfiguration or deletion.
  - **Session Switcher & Live SSE Stream**: Supports switching sessions, creating new sessions, and monitoring real-time workflow events.

---

## 8. SSEExecutor & Harness Tool Call Echo (`framework/sse_executor_v4.py`)

### 8.1 Session Routing
- Resolves incoming `session_id`. If absent, automatically generates a new session and instantiates an isolated `GraphV4`.

### 8.2 Subgraph Vertices
- Vertices with attribute `subgraph` load a child graph with session namespace `f"{session_id}::{vertex_name}"`.
- The child graph is registered in `SessionGraphManagerV4` and observable in the online API.

### 8.3 Harness Tool Call Echo Format
Responses are formatted as standard OpenAI / harness tool calls (`name: "echo"` with `arguments.info`):

#### Non-Streaming JSON Response:
```json
{
  "id": "call_v4_982ab71c",
  "type": "function",
  "function": {
    "name": "echo",
    "arguments": "{\"info\": {\"session_id\": \"sess_01\", \"success\": true, \"completed_edges\": [\"e1\", \"e2\"], \"vertex_states\": {\"v_end\": \"data ready\"}}}"
  }
}
```

#### Streaming SSE Chunk Format:
```text
data: {"id": "chatcmpl-912a", "object": "chat.completion.chunk", "created": 1741373900, "model": "vea-v4-sse-executor", "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_v4_...", "type": "function", "function": {"name": "echo", "arguments": "{\"info\": {\"event\": \"edge_completed\", \"edge_id\": \"e1\"}}"}}]}, "finish_reason": null}]}

data: [DONE]
```

---

## 9. Verification & Test Suite

All components are verified by an 18-test automated suite (`tests/test_v4_system.py` and `tests/test_v4_server.py`):
- SQLite CRUD & Staging attribution.
- Two-sided handshake enforcement and standalone CLI.
- Discrete manifest parsing and DAG cycle detection.
- Concurrency limiting and DAG tier ordering.
- Reflexive recovery loops and circuit breaker lockout.
- Online graph mutation API and multi-session isolation.
- SSEExecutor session routing, subgraph resolution, and tool call echo streaming.
