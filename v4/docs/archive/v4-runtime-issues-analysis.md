# Runtime Issues Analysis and Hardening Report

> **Date**: 2026-09-08  
> **Status**: Resolved  
> **Scope**: Boundary defect review and hardening in scheduling, dynamic graph mutation, concurrent re-entry, and process restart scenarios.

---

## 1. Overview

An in-depth review of the core codebase (`ExecutorV4`, `ServerV4`, and `GraphV4`) identified four potential runtime edge cases and hazards under high-concurrency, dynamic mutation, and service restarts.

**Current Status: All 4 issues resolved** (commit `625b9da`).

| Issue | Severity | Status |
|:---|:---:|:---:|
| 1. Dynamic Graph Mutation Without Session Lock | High | Resolved |
| 2. Fan-In Barrier Deadlock on Upstream Failure | Critical | Resolved |
| 3. Re-entry Race Condition on In-Flight Tasks | Medium | Resolved |
| 4. `active_dispatches` Blind Spot on Service Restart | Low | Resolved |

---

## 2. Issue Details and Fixes

### 1. Dynamic Graph Mutation Without Session Lock (Resolved)

#### Affected Code
- [`framework/server_v4.py`](../../framework/server_v4.py) (`SessionGraphManagerV4`)
- Mutation routes: `POST /graph/vertices`, `POST /graph/edges`, `DELETE /graph/vertices/{name}`, `POST /graph/subgraphs/splice`
- Execution route: `POST /api/sessions/{session_id}/run`

#### Hazard Scenario
- The `/run` route acquires `async with manager.get_session_lock(session_id):`, but graph mutation routes did not acquire the session lock.
- Consequence: Dynamic modifications during long executions triggered `RuntimeError: dictionary changed size during iteration`.

#### Solution
Added `async with manager.get_session_lock(session_id):` across all graph mutation routes in `server_v4.py`:
- `create_or_update_vertex`
- `delete_vertex`
- `create_or_update_edge`
- `delete_edge`
- `reconnect_edge_route`
- `splice_subgraph_route`
- `insert_subgraph_route`
- `add_subgraph_route`

---

### 2. Fan-In Barrier Deadlock on Upstream Failure (Resolved)

#### Affected Code
- [`framework/executor_v4.py`](../../framework/executor_v4.py) (`_execute_single_edge`)

#### Hazard Scenario
- The fan-in counter only incremented when `res.success == True`.
- Consequence: If edge `A->C` succeeded (count=1) but edge `B->C` failed, count remained at 1 indefinitely. Node `C` remained in `todo`, causing a permanent deadlock.

#### Solution
Introduced `_fan_in_failures` tracking and ensured `_fan_in_counts` increments on all completion branches:
```python
self._fan_in_counts[edge.output_vertex] += 1
self._fan_in_failures[edge.output_vertex] += 1

if self._fan_in_counts[edge.output_vertex] >= expected:
    if self._fan_in_failures[edge.output_vertex] > 0:
        # Fall back to REJECT to permit reflexive self-healing
        self.store.update_vertex_state(..., VertexStateV4.REJECT.value)
    else:
        # All edges succeeded -> DATA_READY
        self.store.update_vertex_state(..., VertexStateV4.DATA_READY.value)
```

Added `fan_in_failed` telemetry event emission for observability.

---

### 3. Re-entry Race Condition on In-Flight Tasks (Resolved)

#### Affected Code
- [`framework/server_v4.py`](../../framework/server_v4.py) (`reenter_vertex_route`)
- [`framework/executor_v4.py`](../../framework/executor_v4.py) (`cancel_downstream_tasks`)

#### Hazard Scenario
- `/graph/vertices/{name}/reenter` reset downstream nodes to `todo`, but did not cancel active in-flight coroutines.
- Consequence: Old tasks completing later wrote `update_vertex_content(state=DATA_READY)`, overwriting the re-entered state.

#### Solution
- **ExecutorV4**: Added `_running_tasks_by_output: Dict[str, Set[asyncio.Task]]` and `cancel_downstream_tasks(vertex_names: Set[str]) -> List[str]`.
- **ServerV4**: Added `_running_executors` registry in `SessionGraphManagerV4`. The re-entry route acquires the session lock, cancels active downstream tasks, and resets state.

---

### 4. `active_dispatches` Blind Spot on Service Restart (Resolved)

#### Affected Code
- [`framework/executor_v4.py`](../../framework/executor_v4.py) (`_dispatch_leases`, `recover_stale_leases`)

#### Hazard Scenario
- `active_dispatches` resided solely in memory, lost on process restarts.
- Consequence: A restarted scheduler seeing `data ready` upstream and `todo` downstream could re-dispatch edges prematurely.

#### Solution
- Added `_dispatch_leases: Dict[str, float]` recording `(edge_id -> expiry_timestamp)`.
- Leases are cleared upon task completion.
- Added `recover_stale_leases()` to identify expired dispatch leases on process restart.

---

## 3. Verification

All tests pass cleanly:
```
526 passed
```

| Dimension | Before Fix | After Fix |
|:---|:---:|:---:|
| Graph Mutation Concurrency | Unlocked | Fully locked via session lock |
| Fan-In Failure Handling | Deadlock | Graceful transition to REJECT |
| Re-entry In-flight State | Overwritten | Explicit task cancellation |
| Restart Recovery | State blind spot | Lease tracking enabled |
