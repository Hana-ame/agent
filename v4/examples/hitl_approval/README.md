# HITL Approval — Human Approval Gates and SQLite Checkpoints

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates pausing at sensitive nodes for human approval and resuming execution from SQLite checkpoints. Framework v2.0 feature.

## Problem

Production workflows frequently require **human approval** prior to sensitive operations (e.g. financial transactions, publishing, high-risk shell commands), as well as the ability to recover from crashes in long-running jobs.

## Solution

- Vertices configured with `settings.require_approval: true` (or via `pause_for_approval()`) transition into the `PAUSED` state, suspending execution.
- `CheckpointedExecutor` persists state snapshots to `SQLiteStateStore` following each vertex settlement.
- The workflow resumes upon invoking `approve()`, or can be reloaded via `resume()` after process interruption.

## Changes

- `framework/vertex.py`: Implemented `PAUSED` state and `pause_for_approval()` / `approve()` methods.
- `framework/executor/checkpoint.py`: Implemented `CheckpointedExecutor` and `SQLiteStateStore`.
- `examples/hitl_approval/demo.py`: Demonstrates pause and approval workflow.

## Verification

- **Test Plan**: Verify execution halts at approval gates, snapshots are persisted, and workflows resume cleanly.
- **Method**:
  ```bash
  python examples/hitl_approval/demo.py
  ```
- **Result**: Sensitive vertex enters `PAUSED` state; execution completes following `approve()`; covered in `tests/test_checkpoint.py`.
