# Vertex-Edge Agent Framework: Development Roadmap

> Each milestone records architectural evolution using the "Problem / Solution / Changes / Tests" paradigm.

---

## ✅ v1.0: Core Architecture (Completed)

### Feature 1: Event-Driven DAG Engine
- **Problem**: Lack of unified orchestration engine; vertices and edges managed state inconsistently.
- **Solution**: Declarative JSON topology with asynchronous event-loop scheduling via `Executor`.
- **Changes**: `Graph` (schema loading and validation), `Executor` (`run()` and non-blocking `stream()`).
- **Tests**: Validated in `tests/test_graph.py` and `tests/test_executor.py`; executed successfully across `examples/simple/`.

### Feature 2: Unified 5-Stage Edge Pipeline
- **Problem**: Edge routing and pipeline orchestration were detached, creating complex state boundaries.
- **Solution**: Consolidated orchestration into `Edge` with five distinct stages: Guard -> Pre-Process -> Compute -> Post-Process -> Deliver.
- **Changes**: `framework/edge.py` absorbed pipeline execution logic.
- **Tests**: Fully verified in `tests/test_edge.py` and `tests/test_improvements.py`.

### Feature 3: Standardized Messaging (EdgeSignal)
- **Problem**: Ad-hoc vertex-to-edge function calls created tight coupling.
- **Solution**: Standardized all state coordination on `handle_edge_signal` with `COMPLETED / ABORTED / FAILED` signals.
- **Changes**: `framework/vertex.py`, `framework/edge.py`.
- **Tests**: Verified state transitions and signal handling in `tests/test_vertex.py`.

### Feature 4: Conditional Routing and Dynamic Pruning
- **Problem**: Failed conditional edges left downstream nodes blocked indefinitely.
- **Solution**: Unmet guards trigger `ABORTED` signals with cascading pruning; settlement barrier ensures vertex activates only if at least one input edge succeeds.
- **Changes**: `Edge.condition` and executor settlement barrier logic; demonstrated in `examples/conditional_routing`.
- **Tests**: Verified that non-selected branches are pruned cleanly without deadlocks.

### Feature 5: Script Extension via Subclassing
- **Problem**: Earlier design attempted to load modules as procedural functions.
- **Solution**: `script: file.py:ClassName` format dynamically loads subclasses of `Vertex`, `Edge`, or `MapEdge`.
- **Changes**: `framework/utils/script_loader.py` with explicit class name lookup priority.
- **Tests**: Verified explicit class resolution in `tests/test_script_loader.py`.

### Feature 6: Concurrency Control
- **Problem**: Unbounded fan-out risked overwhelming LLM rate limits and sockets.
- **Solution**: Configurable semaphores per concurrency group (`llm`, `fetch`, `default`).
- **Changes**: `Executor.__init__(concurrency_config)`.
- **Tests**: Concurrency bounds verified in `tests/test_improvements.py`.

---

## ✅ v2.0: Application-Ready (Completed)

### Feature 1: Execution Retries and Self-Correction
- **Problem**: LLM schema failures or validation errors required feedback injection without polluting prompt state.
- **Solution**: `retry_policy` (max_retries, backoff, retry_on) combined with immutable `_base_prompt` isolation.
- **Changes**: `framework/edge.py`, `tests/test_retry_and_stream.py`.
- **Tests**: Verified that system feedback does not accumulate across successive retries.

### Feature 2: State Persistence and Checkpointing
- **Problem**: Long-running workflows required recovery upon failure or restart.
- **Solution**: `SQLiteStateStore` snapshots with `CheckpointedExecutor.resume()`.
- **Changes**: `framework/executor/checkpoint.py`, `framework/utils/store.py`.
- **Tests**: Snapshot and recovery validated in `tests/test_checkpoint.py`.

### Feature 3: Human-in-the-Loop (HITL) Approvals
- **Problem**: Sensitive operations require explicit user confirmation before downstream execution.
- **Solution**: `VertexState.PAUSED`, `pause_for_approval()`, `approve()`, and `require_approval` schema flags.
- **Changes**: `framework/vertex.py`, `examples/hitl_approval`.
- **Tests**: Verified pause and resume flow in test suite.

### Feature 4: Realtime Event Streaming
- **Problem**: Opaque background execution prevented real-time UI/CLI feedback.
- **Solution**: `executor.stream()` producing `GraphEvent` instances over an asynchronous queue.
- **Changes**: `framework/executor/base.py`, `examples/realtime_streaming`.
- **Tests**: Streaming event sequence verified in `tests/test_retry_and_stream.py`.

### Feature 5: Bounded Graph Cycles
- **Problem**: Cyclic feedback loops could cause infinite execution loops.
- **Solution**: Require `max_iterations > 0` on cycle back-edges with DFS cycle detection and runtime iteration bounds.
- **Changes**: Cycle validation in `framework/graph.py`.
- **Tests**: Cycle limits and `GraphCycleError` verified in `tests/test_loops.py`.

---

## ✅ v3.0: Enterprise-Grade (Completed)

| # | Feature | Status |
|:-:|---|---|
| 1 | `SubgraphVertex` nested subgraphs (input/output mappings, event bubbling) | ✅ Completed |
| 2 | `MemoryStore` global shared memory (TTL, namespaces, read/write hooks) | ✅ Completed |
| 3 | `TelemetryTracker` cost and latency profiling | ✅ Completed |
| 4 | Race Mode (`wait_policy: any` first-to-finish execution) | ✅ Completed |
| 5 | Dynamic topology generation (`LinearChain.build`) | ✅ Completed |
| 6 | `SchemaRegistry` Pydantic payload validation | ✅ Completed |
| 7 | Distributed multi-node execution workers | ✅ Completed |

---

## ✅ v4.0: Production-Ready Vertex-Edge Framework (Completed)

> Complete architectural rewrite: SQLite-backed persistent vertices, two-sided handshake protocol, topological tier scheduling, in-memory dispatch deduplication.

### Feature 1: SQLite-Backed Persistent Storage
- **Problem**: Legacy versions used in-memory Python dicts; process crash = total data loss.
- **Solution**: `VertexStoreV4` — dual-table SQLite schema (vertices + edges + staging) with WAL mode.
- **Changes**: `framework/vertex_v4.py`, `framework/store_v4.py`.
- **Tests**: `tests/test_v4_layer1_store.py`.

### Feature 2: Two-Sided Handshake Protocol
- **Problem**: Legacy edges used one-sided signaling with implicit state transitions.
- **Solution**: Upstream `data ready` → downstream `todo`/`todo urgent` → edge executes → upstream updates state.
- **Changes**: `framework/edge_v4.py` (`check_handshake`), `framework/executor_v4.py`.
- **Tests**: `tests/test_v4_layer2_edge.py`.

### Feature 3: Topological Tier Scheduling
- **Problem**: Legacy executor used simple counter polling with no tier awareness.
- **Solution**: `asyncio.Event`-driven scheduler with `compute_dag_tiers()` tier assignment.
- **Changes**: `framework/executor_v4.py`, `framework/graph_v4.py` (`compute_dag_tiers`).
- **Tests**: `tests/test_v4_layer3_graph.py`, `tests/test_v4_layer4_executor.py`.

### Feature 4: Reflexive Self-Loop Recovery
- **Problem**: Failed edges required manual restart; no automatic retry/recovery mechanism.
- **Solution**: `ReflexiveEdgeV4` — self-loop edges that catch `reject` state, reset to `todo urgent`, trigger automatic retry with retry limit and circuit breaker.
- **Changes**: `framework/edge_v4.py` (`ReflexiveEdgeV4`).
- **Tests**: `tests/test_v4_layer4_executor.py` (retry limit, circuit breaker).

### Feature 5: SenseNova 6.8 Flash Lite Integration
- **Problem**: No direct remote LLM integration; required manual agent configuration.
- **Solution**: `SensenovaEdgeV4` — zero-config edge with optional API key, prompt templating, JSON validation, SSE streaming.
- **Changes**: `framework/sensenova_edge_v4.py`.
- **Tests**: `tests/test_sensenova_edge_v4.py` (28 tests).

### Feature 6: Production Runtime Hardening
- **Problem**: 4 runtime issues identified in production scenario analysis (graph mutation races, fan-in deadlock, reentry stomp, restart blind spot).
- **Solution**: Session locks on all graph mutation routes, fan-in failure settlement barrier, in-flight task cancellation on reentry, dispatch lease tracking.
- **Changes**: `framework/server_v4.py`, `framework/executor_v4.py`.
- **Report**: `docs/archive/v4-runtime-issues-analysis.md`.

### Feature 7: Web Dashboard & SSE API
- **Problem**: No web interface or real-time event streaming.
- **Solution**: FastAPI REST API with `/api/sessions/*/run`, SSE event broadcast, interactive web dashboard.
- **Changes**: `framework/server_v4.py`, `framework/sse_executor_v4.py`.
- **Tests**: `tests/test_v4_server.py`, `tests/test_v4_layer5_server.py`.

### Upcoming: v5.0
- **Objective**: Multi-tenant session isolation with per-session concurrency budgets.
- **Approach**: Redis-backed state store adapter; session-scoped rate limiting middleware.
