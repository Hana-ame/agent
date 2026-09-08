# 🔁 Vertex-Edge Agent Framework — Review Round 2 Disposition Log

> Six issues identified during the second review round have all been resolved. This document records each issue following the "Problem / Solution / Changes / Verification" format.
> **Scope**: `framework/` (vertex, executor, agents, utils) + 2 examples + test suite. **Status**: 355 tests passed ✅.

---

## Issue 1: Mixed-Input Loop Silent Deadlock

### Problem
When a loop back-edge targeted a vertex that also received one-time non-back-edge inputs, the graph deadlocked. The loop re-entry branch in `Vertex.receive_signal` invoked `clear()` on `completed_incoming_edges`, retaining only the back-edge. As a result, one-time seed inputs were lost and never redelivered, leaving the vertex permanently IDLE in a `Deadlock`. Additionally, the loop branch counted the back-edge as a required input for the current round, meaning the vertex deadlocked on round one waiting for a back-edge that could only be produced after the vertex ran.

### Solution
- Treat loop back-edges strictly as "re-entry triggers" rather than prerequisites for the initial round.
- Do not clear settled non-back-edge inputs during re-entry (one-time seed data must persist).
- Standardize readiness determination: a vertex becomes READY when all **non-back-edge** incoming edges are settled (completed or aborted).

### Changes
- `framework/vertex.py`: Removed `clear()` and `_received_input_count=0` from the loop re-entry branch; updated to `completed_incoming_edges.add(edge_id)` and evaluated readiness across settled non-back-edge inputs; adjusted standard delivery readiness to evaluate the required non-back-edge list.
- `tests/test_loops.py`: Added `TestMixedInputLoop` (2 test cases).

### Verification
- **Test Plan**: Construct mixed topology `A(seed) -> X -> Y -> X(loop), X -> Z` with `max_iterations=3/5`; verify iteration completion, lack of deadlocks, and accurate iteration counts.
- **Method**:
  - `pytest tests/test_loops.py::TestMixedInputLoop -v`
  - Reproduction test on old logic confirmed deadlock behavior prior to fix.
- **Result**: 2 new test cases passed; `X.iteration_count == 3/5`, X/Y/Z all reach DONE, errors list is empty, and `Z` receives delivery on every round; `test_loops.py` all 13 tests passed.

---

## Issue 2: HITL Pause Reported as Execution Failure

### Problem
When executing a graph paused at a human approval gate (PAUSED state), `ExecutionResult` set `success=False` with `errors=[]`, and `summary()` displayed `FAILED ✗`, despite the store state being `awaiting_approval`. A normal pause state was erroneously interpreted by downstream callers as a workflow failure.

### Solution
Add a `paused` attribute to `ExecutionResult`. If the execution halts with any PAUSED vertices, set `paused=True`. Update `summary()` to display `PAUSED ⏸ (waiting for human approval)` instead of `FAILED`.

### Changes
- `framework/executor/base.py`: Added `self.paused = False` to `ExecutionResult.__init__`; set `self._result.paused = any(v.state == PAUSED)` at completion of `_run_internal`; excluded paused state from `success` failure logic; branched `summary()` title display when paused.
- `tests/test_checkpoint.py`: Added assertions to HITL test cases.

### Verification
- **Test Plan**: Execute a HITL graph containing a `require_approval` vertex; verify consistency between `paused`, `success`, `errors`, and `summary()`.
- **Method**:
  - `pytest tests/test_checkpoint.py::TestHumanGateVertex -v`
  - Inspected `result.summary()` after pause to verify title and error list.
- **Result**: `paused=True`, `success=False`, `errors==[]`, and summary contains `PAUSED` without `FAILED`; `TestHumanGateVertex` 7 tests passed; all checkpoint tests passed.

---

## Issue 3: Subprocess Agents Lacked Timeout and Leaked on Cancellation

### Problem
`PiAgentRunner` and `OpenCodeAgentRunner` used `proc.communicate()` without timeouts. If an enclosing task was cancelled (e.g. executor or edge timeout), child processes were not terminated, leaving orphaned CLI processes behind.

### Solution
- Wrap execution in `asyncio.wait_for` honoring `settings["timeout"]` when provided.
- Upon timeout or `CancelledError`, terminate the child process with `proc.kill()` and `await proc.wait()`, propagating a descriptive error.

### Changes
- `framework/agents/pi_agent_runner.py`, `framework/agents/opencode_agent_runner.py`: Wrapped `communicate()` in `wait_for`; added handlers for `TimeoutError` and `CancelledError` that cleanly kill and reap child processes.

### Verification
- **Test Plan**: Inject mock long-running commands (`sleep 300`) to test both the `settings["timeout"]` expiration and external cancellation paths; assert clean cancellation and zero leaked processes.
- **Method**:
  - Placed temporary dummy executable (`sleep 300`) on PATH; tested `PiAgentRunner.process(settings={"timeout": 1})` asserting `RuntimeError` containing "timed out"; tested `OpenCodeAgentRunner` cancellation after 0.3s asserting `CancelledError` propagation.
  - Verified process table with `ps -ef | grep '[s]leep 300'` to confirm zero residual processes.
- **Result**: Pi runner timed out after 1s with `Pi Agent CLI timed out after 1s, killed.`; OpenCode runner handled external cancellation cleanly; process count returned 0; all 355 tests passed.

---

## Issue 4: Stale References and Obsolete Terminology in Documentation

### Problem
Three outdated naming artifacts remained: `factory._build_from_dict` docstring listed `"proxy"` as a valid agent type (only `http` and `opencode` are supported); `examples/run.py` comments described legacy per-edge agent fallback semantics; `examples/self_correction/demo.py` docstring referenced the deprecated `EdgePipeline` name.

### Solution
Synchronize docstrings and comments with current implementations: remove `proxy` from factory docstrings, update `run.py` comments to reflect actual MockAgent fallback and script Edge mechanics, and update `demo.py` to reference `Edge`.

### Changes
- `framework/agents/factory.py`: Removed `|proxy` from two docstrings.
- `examples/run.py`: Updated comments to explain that standard Edges with prompt/model fall back to MockAgent, while real LLM and subprocess agents are self-contained in script Edges.
- `examples/self_correction/demo.py`: Replaced `EdgePipeline` with `Edge` and updated top module docstring.

### Verification
- **Test Plan**: Ensure `http|opencode|proxy` and `EdgePipeline` no longer appear across the repository.
- **Method**:
  - `grep -rn "http|opencode|proxy" framework/` -> 0 occurrences.
  - `grep -rn "EdgePipeline" examples/ framework/` -> 0 occurrences.
- **Result**: 0 matches for both searches; import succeeded; all 355 tests passed.

---

## Issue 5: Telemetry Pricing Disconnected From Active Models

### Problem
`DEFAULT_PRICING` only defined rates for proprietary tier models (Gemini, GPT, Claude). Free tier models like `hy3-free` defaulted to placeholder pricing ($1/$3 per 1M tokens), producing inaccurate cost metrics for zero-cost runs.

### Solution
Explicitly define `hy3-free` at $0.0 input and $0.0 output per 1M tokens; document fallback behavior for unlisted models.

### Changes
- `framework/utils/telemetry.py`: Added `"hy3-free": {"input": 0.0, "output": 0.0}` to `DEFAULT_PRICING` with explanatory comments.

### Verification
- **Test Plan**: Verify `calculate_cost` returns 0.0 for `hy3-free` and falls back to default rates for unknown models.
- **Method**:
  - Ran `python -c "from framework import calculate_cost; print(calculate_cost(1000, 500, 'hy3-free'))"` -> 0.0.
  - Verified fallback behavior on unknown model names.
- **Result**: hy3-free cost is 0.0; unknown models route to default placeholder pricing; `pytest tests/test_memory_and_telemetry.py -q` passed; 355 tests passed.

---

## Issue 6: Calling `run()` a Second Time on Executor Returned Stale Results Silently

### Problem
Calling `run()` a second time on an already completed `Executor` instance encountered terminal vertex states, causing `_loop` to exit immediately and return stale results with `success=True` without indicating that no execution occurred.

### Solution
Introduce a `_has_run` guard: calling `run()` or `stream()` more than once on the same Executor instance raises a `RuntimeError`, instructing callers to create a new Executor instance.

### Changes
- `framework/executor/base.py`: Added `self._has_run = False` to `Executor.__init__`; checked and set flag in `stream()` entry point.
- `tests/test_executor.py`: Added test case verifying re-execution raises an exception.

### Verification
- **Test Plan**: Run an executor to completion, call `run()` again, and assert `RuntimeError` is raised.
- **Method**: `pytest tests/test_executor.py::test_second_run_raises_instead_of_silent_stale -v`.
- **Result**: First run succeeded with `success=True`; second run raised `RuntimeError` matching "already been run"; all executor tests passed; all 355 tests passed.

---

## Conclusion
All 6 issues identified in Round 2 have been resolved and verified with regression tests. Test suite status: `pytest tests/` = **355 passed**.
