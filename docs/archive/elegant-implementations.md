# Elegant Architectural Implementations

> This document tracks framework refactoring proposals using the "Problem / Solution / Changes / Tests" paradigm, noting current implementation status.
> Status: **#1 Implemented (MapEdge)**, **#2 Partially Implemented (edge_transform functional factory)**, **#7 Implemented (ExecutionContext / async agent lifecycle)**, with remaining items as optional enhancements.

---

## 1. Replacing Static Fan-Out with `MapEdge`

### Problem
Legacy iterations of fan-out workflows (like `hn_ai_report`) required declaring 10+ vertices and 17+ edges to express "for each item, fetch and summarize", creating verbose configurations that were difficult to scale.

### Solution
Introduce `MapEdge`: a single edge that executes a sub-pipeline over each list element concurrently, then fans-in results. Expressing "for each X do Y" becomes a unified concept.

### Changes
- `framework/edge.py`: Added `MapEdge` (pipeline steps, `max_concurrency` semaphore, and fan-in delivery).
- `examples/hn_ai_report/config.json`: Employs `ProcessStoriesMap` with `settings.pipeline`.
- `examples/s1_ai_report_map/config.json`: Employs identical `ProcessThreadsMap`.

### Verification
- Unit test suite validates concurrency bounds, step isolation, and fan-in aggregation.
- Live verification via `examples/hn_ai_report/demo.py` and `examples/s1_ai_report_map/demo.py`.

---

## 2. Composable Edge Stages: Functional Factories

### Problem
Requiring a dedicated subclass for simple transformations (e.g. `SelectEdge` returning `data[index]`) introduced unnecessary boilerplate.

### Solution
Provide `edge_transform(pre, post, guard)` functional factory to generate `Edge` subclasses from pure functions, allowing lightweight transformations and complex class-based edges to coexist cleanly.

### Changes
- `framework/edge.py`: Added `edge_transform()` factory generating dynamic `FunctionalEdge` types.

### Verification
- Tested in `tests/test_edge.py` confirming full parity with explicitly subclassed edges.

---

## 3. Declarative Vertex State Machine Descriptors

### Problem
Scattered state mutations across executor, signals, and checkpoints risked invalid state transitions without compile-time enforcement.

### Solution
Centralized state descriptor with an explicit transition table (`StateMachine.TRANSITIONS`), raising errors on illegal transitions while providing `force_state()` for reset and checkpoint restoration.

### Changes
- `framework/vertex.py`: Implemented declarative `StateMachine` descriptor and validated transitions.

### Verification
- Verified state validation and `force_state` recovery across unit tests.

---

## 4. Immutable Execution Context and Typed Slots

### Problem
Unrestricted dictionary mutations can cause subtle key collisions and runtime `KeyError` exceptions.

### Solution
Typed `Slot` declarations where edges declare explicit `produces` and `consumes` channels, enabling static graph compile-time validation.

### Status
- Optional enhancement: runtime checks currently manage slot safety; compile-time validation remains planned.

---

## 5. Functional Edge Transformations

### Problem
Subclassing overhead for trivial pure data transformations.

### Solution
Combine `edge_transform()` functional generation with declarative settings parameters.

### Changes
- `framework/edge.py`: `edge_transform` implementation. Security-sensitive inline code execution (e.g. `eval`) is avoided.

### Verification
- Unit tests verify factory-generated transformations execute safely.

---

## 6. Declarative Guard DSL

### Problem
Ad-hoc operator dispatch previously risked string injection vectors.

### Solution
Secure operator dictionary dispatch supporting standard comparisons (`>`, `<`, `==`, `!=`, `contains`, `matches`) without dynamic `eval()`.

### Changes
- `framework/edge.py`: Implemented explicit operator mapping from settings. Dynamic `eval()` completely eliminated.

### Verification
- Validated with unit tests covering all supported operators and cascading pruning behavior in conditional routing graphs.

---

## 7. Unified Resource Lifecycles

### Problem
Manual allocation and cleanup of HTTP clients, memory stores, and telemetry trackers risked socket and memory leaks during long runs.

### Solution
`ExecutionContext` asynchronous context manager: automatically initializes and disposes agents, stores, and telemetry. HTTP clients implement idempotent `close()` and `__aenter__/__aexit__`.

### Changes
- `framework/executor/base.py`: Added `ExecutionContext`.
- `framework/agents/_http_base.py`: Implemented async context manager and cache eviction.

### Verification
- Verified in `tests/test_agents.py` across normal and exceptional termination paths.

---

## Impact × Cost Assessment

| Proposal | Architecture Impact | Implementation Cost | Risk | Status |
|---|---|---|---|---|
| 1. `MapEdge` | High | Medium | Low | ✅ Implemented |
| 2. Functional Factories | Medium | Medium | Low | ✅ Implemented (`edge_transform`) |
| 3. State Machine Descriptor | Medium | Low | Medium | ✅ Implemented |
| 4. Typed Slots | High | Large | Medium | ⏳ Planned |
| 5. Functional Transformations | Low | Low | Low | ✅ Implemented |
| 6. Safe Guard DSL | Medium | Low | Low | ✅ Implemented |
| 7. Resource Lifecycle | Medium | Low | Low | ✅ Implemented |
