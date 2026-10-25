# Realtime Streaming — Non-blocking Event Stream Observability

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates real-time observability into graph state transitions during execution. Framework v2.0 feature.

## Problem

Graph execution is asynchronous, and users typically only inspect results upon completion. Debugging and monitoring require real-time visibility into vertex state transitions and edge triggers during runtime.

## Solution

The `executor.stream()` asynchronous generator yields `GraphEvent` instances from an internal `asyncio.Queue`, terminated by a sentinel. Consumers iterate over events in real-time via `async for event in executor.stream()`.

## Changes

- `framework/executor/base.py`: Implemented `stream()` generator and `GraphEvent` dataclass.
- `examples/realtime_streaming/demo.py`: Renders the event stream with ANSI colors.

## Verification

- **Test Plan**: Verify events are emitted in chronological order without blocking and terminate cleanly.
- **Method**:
  ```bash
  python examples/realtime_streaming/demo.py
  ```
- **Result**: Real-time printing of vertex state transitions (`IDLE` -> `READY` -> `AWAITING_EDGES` -> `DONE`) and edge invocations without stalling; covered in `tests/test_retry_and_stream.py`.
