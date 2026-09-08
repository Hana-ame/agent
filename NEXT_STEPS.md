# 🚀 Vertex-Edge Agent Framework v4.0: Next Steps & Layered Specification

> **Status**: Layers 1–3 Complete & Verified (508 tests passing).  
> **Target**: Layer 4 (Executor & Fan-In), Layer 5 (Gateway & Security), and Layer 6 (Recursive Subgraphs & Distributed Workers).

---

## 1. Architectural Overview & Layered Hierarchy

The v4.0 architecture is decomposed into six strictly decoupled layers, tested and verified bottom-up:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 6: System E2E & Production                                       │
│          - Deep Recursive Subgraphs, Multi-Agent Workflows, Distributed │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 5: Server & HTTP Gateway (ServerV4, SSEExecutorV4)               │
│          - Path Traversal Hardening, Pydantic Schema, Dashboard UI     │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Executor & Scheduler (ExecutorV4)                              │
│          - Fan-In / Fan-Out Merge, Event-Driven Loop, Concurrency      │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Graph Topology (GraphV4)                          ✅ COMPLETED │
│          - Dynamic Mutation, DAG Tiering, Subgraph Joins, Dump/Load   │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Edge Runtime (EdgeV4, SenseNovaEdgeV4)             ✅ COMPLETED │
│          - Two-Sided Handshake, Staging, Prompt Fills, Reflexive Loop  │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Storage Layer (VertexStoreV4)                     ✅ COMPLETED │
│          - SQLite3 Key-Indexed Rows, CHECK Constraints, Staging Table  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Current Implementation Status Matrix

| Layer | Component | Test Coverage File | Status | Key Highlights |
|---|---|---|:---:|---|
| **Layer 1** | Storage (`vertex_v4.py`) | `tests/test_v4_layer1_store.py` (16 tests) | ✅ **Done** | SQLite `(session_id, name)` storage, staging attribution, atomic transactions, CHECK constraints. |
| **Layer 2** | Edge Runtime (`edge_v4.py`, `sensenova_edge_v4.py`) | `tests/test_v4_layer2_edge.py` (35 tests), `test_sensenova_live.py` (4 tests) | ✅ **Done** | Two-sided handshake, agent dispatch, JSON markdown fence stripping, reflexive self-healing with circuit breaker. |
| **Layer 3** | Graph Topology (`graph_v4.py`) | `tests/test_v4_layer3_graph.py` (30 tests) | ✅ **Done** | Kahn's DAG tiering, cyclic validation, dynamic vertex/edge insertion/deletion, subgraph joins, serialization/dump. |
| **Layer 4** | Executor & Scheduling (`executor_v4.py`) | `tests/test_v4_layer4_executor.py` (5 tests), `test_v4_system.py` (13 tests) | ✅ **Done** | Fan-in settlement barrier, `MergeStrategyV4` accumulation policies, reactive `asyncio.Event` scheduler wakeup. |
| **Layer 5** | Server & Gateway (`server_v4.py`, `sse_executor_v4.py`) | `tests/test_v4_layer5_server.py` (7 tests), `test_v4_server.py` (16 tests) | ✅ **Done** | Strict path traversal confinement (HTTP 400 + structured error code), Pydantic v2 schemas, extracted standalone HTML dashboard template. |
| **Layer 6** | System E2E & Production | `tests/test_v4_layer6_e2e.py` (7 tests) | ✅ **Done** | `BaseWorkerQueueV4` distributed queue protocol, recursive nested subgraphs, non-mock SenseNova live E2E pipeline & streaming. |

---

## 3. Detailed Specifications for Next Steps

### 3.1 Layer 4: Executor & Fan-In Accumulation

#### Problem
In graph architectures with fan-in patterns (multiple upstream edges delivering to a single destination vertex), independent executions risk race conditions and clobbering destination content. Furthermore, the scheduler currently relies on a 20ms sleep polling loop rather than reactive event notification.

#### Requirements
1. **Vertex Fan-In Merge Strategies (`MergeStrategyV4`)**:
   - Support configurable write behaviors when multiple edges target the same vertex:
     - `OVERWRITE` (default): Latest settled edge overwrites `content`.
     - `JSON_MERGE`: Parses existing vertex content and incoming output as JSON objects and merges keys (e.g. `{"edge1_res": ..., "edge2_res": ...}`).
     - `LIST_APPEND`: Appends each incoming edge output into a JSON list `[...]`.
     - `REDUCER_SCRIPT`: Invokes a Python reducer function `(previous_content, new_edge_result) -> updated_content`.
2. **Settlement Barrier & Readiness Coordination**:
   - Track expected incoming forward edges per vertex. Downstream vertex transitions from `todo` to `data ready` only when all participating incoming edges have completed or aborted.
3. **Event-Driven Scheduling**:
   - Replace the `await asyncio.sleep(0.02)` loop in `ExecutorV4._scheduler_loop()` with an `asyncio.Event` trigger tied to edge completion and state mutation callbacks.
4. **Dedicated Test Suite**:
   - Create `tests/test_v4_layer4_executor.py`:
     - Test semaphore boundary enforcement under heavy concurrency (e.g. 50 parallel tasks with `max_concurrency=4`).
     - Test priority preservation (`Tier -1` reflexive > `todo urgent` > `todo`).
     - Test fan-in accumulation under simultaneous edge completions.
     - Test task cancellation and exception shielding.

---

### 3.2 Layer 5: Server & HTTP Gateway Security

#### Problem
The current `/api/sse/execute` and manifest endpoints accept file paths without standardized boundary checking, and malformed JSON payloads can leak internal SQLite errors as raw HTTP 500 status codes.

#### Requirements
1. **Path Traversal Security Hardening**:
   - In `/api/sse/execute` and any endpoint accepting `manifest_path` or `dump_path`:
     - Resolve paths against `manifest_base_dir` / workspace root.
     - Reject paths that escape the boundary with **HTTP 400 Bad Request** (rather than 403 or unhandled exceptions), returning structured error JSON:
       ```json
       {"detail": "Invalid path: target is outside allowed directory", "error_code": "PATH_TRAVERSAL_REJECTED"}
       ```
     - Enforce `.json` extension whitelist for manifest loading.
2. **Pydantic Validation Models**:
   - Replace untyped `Dict[str, Any] = Body(...)` with explicit Pydantic v2 request/response schemas:
     - `VertexCreateRequest`: Validates `state` against `VertexStateV4` enum, attributes against `VertexAttributeV4` enum.
     - `EdgeCreateRequest`: Validates `edge_type` (`code`, `llm`, `sensenova`, `reflexive`), input/output vertex names, and schema constraints.
     - Maps validation errors to clean **HTTP 422 Unprocessable Entity** or **HTTP 400 Bad Request** before touching SQLite.
3. **Decouple Web Dashboard**:
   - Extract the 500+ lines of inline `DASHBOARD_HTML` in `server_v4.py` into a static template file (`framework/templates/dashboard.html`), served cleanly via FastAPI `HTMLResponse`.
4. **Dedicated Test Suite**:
   - Implement `tests/test_v4_layer5_server.py` covering:
     - Path traversal attack payloads (`../../etc/passwd`, absolute out-of-tree paths).
     - Validation error responses (HTTP 400/422 on invalid states, missing names, invalid edge types).
     - SSE streaming backpressure and client disconnect handling.

---

### 3.3 Layer 6: System E2E & Production Integration

#### Problem
Complex real-world workflows require nested workflows (graphs containing subgraphs, which in turn contain subgraphs) and the ability to distribute task execution across multiple nodes.

#### Requirements
1. **Deep Recursive Subgraphs**:
   - Extend `SSEExecutorV4.resolve_subgraph_vertices` to recursively resolve child subgraphs to arbitrary depth (hBclevel nesting), with hierarchical session IDs:
     `session_id::subgraph_1::nested_subgraph_2`
   - Propagate inputs and outputs seamlessly across all parent-child boundaries.
2. **Distributed Execution Interface (`WorkerQueueAdapter`)**:
   - Define an abstract execution adapter protocol for offloading edge runs to distributed workers (Redis Queue / Celery / HTTP RPC):
     ```python
     class BaseWorkerQueueV4(ABC):
         @abstractmethod
         async def submit_edge(self, task: EdgeTaskPayload) -> str: ...
         @abstractmethod
         async def poll_result(self, task_id: str) -> EdgeResultV4: ...
     ```
3. **Comprehensive E2E Test Suite**:
   - Create `tests/test_v4_layer6_e2e.py` verifying full pipeline execution combining:
     - SenseNova LLM edge -> Code parsing edge -> Reflexive error recovery -> Multi-level subgraph -> SSE event stream.

---

## 4. Layered Test Plan & File Layout

```
tests/
├── test_v4_layer1_store.py      # ✅ Complete (16 tests, 467 lines)
├── test_v4_layer2_edge.py       # ✅ Complete (35 tests, 1099 lines)
├── test_v4_layer3_graph.py      # ✅ Complete (30 tests, 979 lines)
├── test_v4_layer4_executor.py   # 🚀 NEXT STEP: Concurrency, priority, fan-in merge, event loops
├── test_v4_layer5_server.py     # 🚀 NEXT STEP: Security traversal, Pydantic validation, SSE gateway
├── test_v4_layer6_e2e.py        # 🚀 NEXT STEP: Recursive subgraphs, distributed runner, live workflows
└── test_sensenova_live.py       # ✅ Complete (4 live integration tests)
```

---

## 5. Execution Roadmap & Milestones

### Milestone 1: Layer 4 — Executor Hardening & Fan-In Accumulation
- [x] Implement `MergeStrategyV4` in `framework/vertex_v4.py` and `framework/edge_v4.py`.
- [x] Implement settlement barrier and multi-edge write coordination in `framework/executor_v4.py`.
- [x] Implement `asyncio.Event` notification to replace 20ms polling loop.
- [x] Build `tests/test_v4_layer4_executor.py` and verify all scenarios.

### Milestone 2: Layer 5 — Gateway Security & Validation
- [x] Add path sanitization with strict confinement checking and HTTP 400 responses in `server_v4.py`.
- [x] Add Pydantic v2 schemas for vertex/edge mutation endpoints to eliminate 500 status leaks.
- [x] Extract `DASHBOARD_HTML` to `framework/templates/dashboard.html`.
- [x] Build `tests/test_v4_layer5_server.py` and verify zero vulnerabilities.

### Milestone 3: Layer 6 — Recursive Subgraphs & Production Readiness
- [x] Support recursive hBcdepth nested subgraph resolution in `sse_executor_v4.py`.
- [x] Create `tests/test_v4_layer6_e2e.py` validating end-to-end multi-agent pipelines.
- [x] Update `README.md` and `ROADMAP.md` documentation to reflect full v4.0 delivery.
