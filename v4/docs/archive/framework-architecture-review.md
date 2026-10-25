# 🏗️ Vertex-Edge Agent Framework — Architecture Review

> Architectural Review Conclusion: The core execution model (Vertex state machine / Edge 5-stage pipeline / Executor asynchronous scheduling / message passing) is well-designed.
> This document records issues identified during architectural review along with their resolutions, organized by "Problem / Solution / Changes / Verification".

**Scope**: `framework/` (17 source files). **Status**: 355 tests passed ✅.

---

## Issue 1: `HttpLLMAgent` Retries on Fatal HTTP Errors

### Problem
tenacity retried on all `httpx.HTTPStatusError` exceptions. Non-transient client errors such as 400, 401, 403, and 404 (authentication or malformed parameters) were retried up to `max_retries` times, wasting quota and adding unnecessary latency.

### Solution
Raise `NonRetryableHTTPError` (a `ValueError` subclass) for non-retryable status codes (400, 401, 403, 404, etc.) so they bypass `retry_if_exception_type`. Only 429 and 5xx (`500, 502, 503, 504`) status codes trigger retry attempts.

### Changes
- `framework/agents/_http_base.py`: Added `NonRetryableHTTPError` and status code dispatch logic; defined `RETRYABLE_STATUS` for 5xx/429.

### Verification
- **Test Plan**: Assert 4xx errors fail immediately without retry; 5xx and 429 trigger retries up to the limit.
- **Method**: Injected mock HTTP 400 and 500 responses; asserted invocation counts.
- **Result**: HTTP 400 fails immediately on the first attempt; HTTP 500 retries up to the configured limit (`tests/test_agents.py`, commit `d64aab2`).

---

## Issue 2: `HttpLLMAgent` Never Closes `httpx.AsyncClient`

### Problem
`AsyncClient` was initialized in `__init__` without an explicit `close()` method, risking socket and file descriptor leaks in long-running services.

### Solution
Add asynchronous context management (`__aenter__` / `__aexit__`) and an idempotent `close()` method that drains and clears cached proxy clients.

### Changes
- `framework/agents/_http_base.py`: Implemented `__aenter__`, `__aexit__`, and `close()`; cached clients in `_proxied_clients` are cleared upon closing.

### Verification
- **Test Plan**: Verify explicit closing, idempotence, and regular/exceptional paths within async context managers.
- **Method**: Regression tests in `tests/test_agents.py`.
- **Result**: Passed (commit `d64aab2`).

---

## Issue 3: Duplicate Return in `HumanGateVertex.__repr__`

### Problem
`__repr__` contained two consecutive `return` statements where the second was unreachable dead code.

### Solution
Remove the redundant return statement.

### Changes
- `framework/executor/checkpoint.py`: Retained a single clean return statement in `__repr__`.

### Verification
- **Test Plan**: Verify `repr()` output format and execution.
- **Method**: `pytest tests/test_checkpoint.py`.
- **Result**: Passed; `__repr__` returns `f"HumanGateVertex(id={self.id}, state={self.state}, approval={self.approval_channel})"`.

---

## Issue 4: Dual Execution Paths in `Pipeline` and `Edge`

### Problem
Earlier designs separated `Pipeline` (5-stage orchestration) from `Edge` (topological routing), rebuilding pipeline objects on every execution and duplicating logic.

### Solution
Consolidate the orchestration pipeline directly into `Edge`; preserve `Pipeline` solely as a backward-compatible alias.

### Changes
- `framework/edge.py`: Absorbed guard checks, pre-processing, compute, retries, post-processing, schema validation, memory access, and telemetry.
- `framework/pipeline.py`: Re-exported `Pipeline = Edge` along with error classes, marked as `DEPRECATED`.

### Verification
- **Test Plan**: Verify `from framework.pipeline import Pipeline` works identically to `Edge`.
- **Method**: `pytest tests/test_improvements.py` (including legacy Pipeline regression tests).
- **Result**: Passed.

---

## Issue 5: `GraphBuilder.vertex()` Stored Script Under Wrong Key

### Problem
Legacy code wrote custom script references into `vc["pipeline"]`, whereas `Graph.from_dict()` read `vc["script"]`, causing custom vertex scripts to be silently dropped.

### Solution
Update `vertex()` to set `vc["script"] = script`.

### Changes
- `framework/builders/builder.py`: Fixed storage key to `vc["script"] = script`.

### Verification
- **Test Plan**: Ensure custom vertex scripts registered via builder are properly executed.
- **Method**: Construct graph using `GraphBuilder().vertex("x", script=...).build()`.
- **Result**: Passed (`tests/test_improvements.py`).

---

## Issue 6: Vestigial `agent` Parameter in `GraphBuilder.edge()`

### Problem
The `agent` argument in `edge()` populated `settings["agent"]`, but `Edge.__init__` no longer consumed this field (`self.agent = None`), silently ignoring it.

### Solution
Remove the parameter; retain `prompt` and `model` which are actively utilized.

### Changes
- `framework/builders/builder.py`: Removed `agent` parameter and associated dictionary assignments in `edge()`.
- `framework/edge.py`: Cleaned docstring to remove `agent` from parsed attributes.

### Verification
- **Test Plan**: Verify the builder no longer references `settings["agent"]`.
- **Method**: `grep "agent" framework/builders/builder.py`.
- **Result**: 0 occurrences. Framework-wide 0 reads of `settings["agent"]` (excluding the `--agent` CLI option in `opencode_agent_runner.py`). **355 tests passed**.

---

## Issue 7: Edge Prompt Accumulated Feedback Across Loop Iterations

### Problem
`retry_policy` mutated `self.prompt` in place, causing cyclic graphs to accumulate duplicate `[SYSTEM FEEDBACK]` blocks across multiple iterations.

### Solution
Freeze `self._base_prompt`; construct an ephemeral `active_prompt` per retry attempt; restore `self.prompt` upon step completion.

### Changes
- `framework/edge.py`: Implemented prompt restoration logic (commit `121ea9e`); added regression test in `tests/test_retry_and_stream.py`.

### Verification
- **Test Plan**: Ensure feedback blocks do not accumulate across retries or iterations.
- **Method**: Assert number of feedback blocks equals 1.
- **Result**: Passed.

---

## Issue 8: `SchemaMismatchError` Declared But Not Raised

### Problem
A dedicated `SchemaMismatchError` exception class was defined, but validation logic was raising a generic `ValueError`.

### Solution
Use `SchemaMismatchError` during graph compilation and validation.

### Changes
- `framework/graph.py`: Changed exception to `raise SchemaMismatchError(...)`.

### Verification
- **Test Plan**: Verify schema validation failures raise `SchemaMismatchError`.
- **Method**: `pytest tests/test_graph.py`.
- **Result**: Passed; `SchemaMismatchError` is consistently raised on mismatched port types.

---

## Issue 9: Connection Leak Risk in `SQLiteStateStore`

### Problem
Non-in-memory databases opened connections via `_connect()` without explicit closure, risking connection resource leakage in long-running processes.

### Solution
Implement `close()`, a `_closed` flag, and context management to guard against reuse after closing.

### Changes
- `framework/utils/store.py`: Added `close()`, `_closed` tracking, and `__exit__` context manager support.

### Verification
- **Test Plan**: Assert closed store rejects queries and duplicate `close()` calls are idempotent.
- **Method**: `pytest tests/test_checkpoint.py`.
- **Result**: Passed.

---

## Issue 10: Legacy `exec()` / `eval()` Arbitrary Execution Risk

### Problem
Older versions of `edge.py` utilized `eval()` for routing conditions and `exec()` for message transformations, presenting an arbitrary code execution vector.

### Solution
Completely removed dynamic string evaluation during refactoring; replaced with subclass overrides and functional `edge_transform` factories.

### Changes
- `framework/`: Removed all instances of `exec()` and `eval()` (0 occurrences remaining).

### Verification
- **Test Plan**: Ensure framework contains 0 dynamic execution calls.
- **Method**: `grep -rnE "\bexec\(|\beval\(" framework/`.
- **Result**: 0 occurrences.

---

## Architectural Notes & Conventions

| Topic | Resolution |
|---|---|
| Single `asyncio.Lock` in `Vertex._data_store` | High fan-in edges serialize; sufficient for current concurrency scales, documented for future optimization |
| Formal Error Hierarchy | `AbortPipeline`, `GuardAbortError`, `HookError`, `ComputeError` (`utils/errors.py`) provide clear failure boundaries |
| Executor Callbacks | Replaced monkey-patching with `ExecutorHooks` callback architecture |
| Per-Edge Timeouts | Fully supported via `settings["timeout"]` |
| Agent Streaming & Lifecycle | Implemented via `stream_process` and async context managers |

---

## Conclusion

The core architecture (actor-inspired state machines, 5-stage edge execution pipeline, bounded loops, checkpointing/HITL, subgraphs, global memory, telemetry, schemas, and race modes) is robust and production-ready. All identified issues have been resolved, and **355 tests passed**.
