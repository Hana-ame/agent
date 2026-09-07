# Handoff: V4 Vertex-Edge Framework

## 1. Executive Summary

This handoff documents the full review, audit, bug-fixing, and verification of the **V4 Vertex-Edge Framework** in accordance with [`v4-architecture-spec.md`](./v4-architecture-spec.md).

All **371 tests** across the repository (both legacy v1–v3 suites and the v4 system/server test suites) pass cleanly:
```bash
uv run --with pytest --with pytest-asyncio --with fastapi --with uvicorn --with beautifulsoup4 --with httpx --with aiohttp pytest tests/ -v
# Output: 371 passed, 2 warnings in 31.40s
```

---

## 2. Architecture & Design Principles

The V4 framework introduces a high-throughput, deterministic execution engine with the following foundational tenets:

1. **Persistent Vertex Storage**: Vertices are SQLite rows indexed by `(session_id, name)` with strict state transitions, rather than ephemeral in-memory objects.
2. **Strict Handshake Protocol**: Forward edges activate if and only if upstream is `data ready` and downstream is `todo` or `todo urgent`.
3. **Reflexive Self-Loop Recovery**: Errors transition downstream to `reject`. Reflexive edges (`input_vertex == output_vertex`) read diagnostics from `session_staging`, apply optional recovery scripts, and increment `processed_count` up to `max_retries`. If exceeded, vertices are locked to `forbidden`.
4. **No DB 'Running' States**: Transient in-flight execution states are maintained purely in memory via `active_dispatches: Set[Tuple[str, str, str]]`, guaranteeing resilience against crashes and preventing database locks.
5. **Topological Tier Scheduling**: Edges are sorted by priority (`reflexive` = 0, `todo urgent` = 1, `todo` = 2), then by topological DAG tier (`Tier -1`, `Tier 0`, `Tier k`).
6. **Online Graph Mutation & Server**: FastAPI server with isolated session graphs, real-time Server-Sent Events (SSE), harness tool-call echo compatibility, and interactive web dashboard.

---

## 3. Audited & Resolved Issues

The spec-vs-code audit revealed 5 critical runtime issues and 16 medium bugs/encapsulation leaks across all 6 v4 modules. All issues have been resolved:

### A. [`framework/vertex_v4.py`](./framework/vertex_v4.py)
- **State Check Constraint**: Added SQLite table constraint `CHECK (state IN ('data ready', 'idle', 'forbidden', 'todo', 'todo urgent', 'reject', 'pruning'))`.
- **Type Signature Fix**: Corrected `save_vertex()` type annotations and string conversion to prevent `VertexAttributeV4` attributes from corrupting the `state` column.
- **Store Encapsulation Methods**:
  - Added `delete_vertex(session_id, name)`
  - Added `list_sessions()`
  - Added `get_db_stats()`
- **Atomic Session Clearance**: Wrapped `clear_session(session_id)` in explicit `BEGIN ... COMMIT / ROLLBACK` transactions.
- **Context Manager**: Added `__enter__` and `__exit__` support for `VertexStoreV4`.

### B. [`framework/edge_v4.py`](./framework/edge_v4.py)
- **LLM Agent Dispatch**: Fixed `LLMEdgeV4.run()` to check for `process()`, `chat()`, `generate()`, and `callable()`, resolving crash when using default `HttpLLMAgent`.
- **JSON Content Cleanup**: Extracted and stripped markdown code fences (` ```json `) so pure JSON strings are saved to the database on validated vertices.
- **Reflexive Failure Handling**: `ReflexiveEdgeV4.run()` now stages error diagnostics and returns `success=False` if a custom recovery script raises an exception (preventing false-positive recovery reporting).
- **Code Cleanliness**: Removed unused import `os`.

### C. [`framework/graph_v4.py`](./framework/graph_v4.py)
- **Reflexive Precedence**: Evaluated `e_type == "reflexive" or in_v == out_v` before `"code"`, ensuring self-loop recovery edges are not misclassified as standard code edges.
- **Infinite Loop Defense**: Capped `compute_dag_tiers()` relaxation loop to $|V| + 1$ iterations, preventing server hangs on cyclic topologies.
- **Graph CRUD APIs**: Implemented `delete_vertex(name)`, `delete_edge(edge_id)`, and `get_edge(edge_id)`, automatically cleaning up incident edges and invalidating cached tiers.
- **Metadata Preservation**: Fixed `get_reflexive_edges()` to retain `script`, `trigger_state`, and `target_state`.
- **Database ID Sync**: `populate_store()` now syncs returned database row IDs back to in-memory `VertexRecordV4.id`.

### D. [`framework/executor_v4.py`](./framework/executor_v4.py)
- **Dispatch Race Elimination**: `active_dispatches.add(key)` is now invoked synchronously at task scheduling time rather than asynchronously inside the coroutine.
- **Overload Slice Guard**: Fixed `available_slots = max(0, self.max_concurrency - len(running_tasks))` to prevent Python negative slicing from launching duplicate tasks.
- **Graceful Cancellation**: Handled `asyncio.CancelledError` via `completed_task.cancelled()` check in the task wait loop.
- **Sink Detection Refinement**: Sink vertex detection in `_is_terminal()` ignores reflexive recovery self-loops, preventing premature termination or missed terminal states.
- **Reliable Seeding**: Moved initial vertex store seeding inside the scheduler `try` block to ensure events and queue sentinels are always emitted on failure.

### E. [`framework/server_v4.py`](./framework/server_v4.py)
- **Encapsulation Enforcement**: Replaced raw SQL queries in route handlers with `store.list_sessions()`, `store.delete_vertex()`, and `store.get_db_stats()`.
- **Cycle Validation before Tiering**: Edge additions validate DAG constraints before tier recomputation.
- **Edge Serialization**: Added `script`, `trigger_state`, `target_state`, and `max_retries` to edge API output, preventing the dashboard editor from overwriting scripts on save.
- **Dashboard Usability**:
  - Preserved `processed_count` during vertex inspector edits.
  - Fixed new session creation in the web dashboard.
  - Sanitized DOM rendering to eliminate XSS vulnerabilities.
- **Code Cleanliness**: Removed redundant `asyncio.create_task` + `await` and removed unused `self._lock`.

### F. [`framework/sse_executor_v4.py`](./framework/sse_executor_v4.py)
- **Active Subgraph Bridging**: Constructed functional non-reflexive bridge edges with proxy output vertices (`f"{v.name}__bridge_out"`), resolving executor deadlock.
- **Input Injection Helper**: Extracted duplicated input injection logic into `_inject_input_payload()`.
- **Stream Exception Shielding**: Added error handling around `executor.stream()`, emitting structured `ToolCallEcho` error chunks on failure.
- **Imports Cleaned**: Removed unused imports (`EdgeResultV4`, `ExecutionResultV4`, `VertexRecordV4`).

---

## 4. Repository File Map

```
framework/
├── vertex_v4.py        # SQLite storage engine, VertexRecordV4, StagingRecordV4, VertexStoreV4
├── edge_v4.py          # EdgeV4 base, CodeEdgeV4, LLMEdgeV4, ReflexiveEdgeV4, CLI runner
├── graph_v4.py         # GraphV4, DiscreteGraphLoaderV4, Kahn's cycle check, DAG tiering
├── executor_v4.py       # Concurrency-limited DAG-ordered scheduler, deduplication, priority
├── server_v4.py        # FastAPI app, SessionGraphManagerV4, REST API, dashboard UI
├── sse_executor_v4.py  # Session router, subgraph bridge execution, OpenAI tool call echo format
└── __init__.py         # Exports all v4 classes and utilities

tests/
├── test_v4_system.py   # 11 integration tests covering storage, edges, loader, executor, recovery
├── test_v4_server.py   # 7 API tests covering server, validation, inspector, SSE streaming
└── ...                 # 353 legacy and component tests
```

---

## 5. Verification & Testing

To run the complete test suite:
```bash
uv run --with pytest --with pytest-asyncio --with fastapi --with uvicorn --with beautifulsoup4 --with httpx --with aiohttp pytest tests/ -v
```

To run only the v4 test suites:
```bash
uv run --with pytest --with pytest-asyncio --with fastapi --with uvicorn --with beautifulsoup4 --with httpx --with aiohttp pytest tests/test_v4_system.py tests/test_v4_server.py -v
```

To launch the standalone server:
```bash
python3 -m framework.server_v4 --host 127.0.0.1 --port 8000 --db ./workflow.db
```

---

## 6. Recommendations for Next Iterations

1. **Extract Dashboard HTML/JS**: Move `DASHBOARD_HTML` (lines 248–765 in `server_v4.py`) into a dedicated static template file (e.g., `framework/static/dashboard.html`) to improve maintainability.
2. **Deep Recursive Subgraphs**: Expand `resolve_subgraph_vertices` to support arbitrary levels of nested subgraphs.
3. **Event-Driven Scheduler**: Replace the executor's 20ms polling interval with `asyncio.Event` triggers tied to edge completion and state transitions.
