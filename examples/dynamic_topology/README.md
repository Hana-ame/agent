# Dynamic Topology — Runtime Graph Growth

> Documented following the "Problem / Solution / Changes / Verification" format: dynamically generates worker vertices during graph execution. Framework v2.0 feature.

## Problem

Static topologies cannot represent workflows where task counts are determined at runtime, such as a manager vertex decomposing an incoming request into an arbitrary number of subtasks, each requiring an independent worker.

## Solution

A manager vertex issues tasks at runtime while the framework **dynamically creates and registers worker vertices and edges** during execution. Each subtask executes asynchronously across independent workers.

## Changes

- `examples/dynamic_topology/demo.py`: Demonstrates runtime vertex and edge registration.
- Leverages native `async def` hooks on `Edge` and runtime graph mutation.

## Verification

- **Test Plan**: Verify worker vertices are dynamically generated for each task item and processed concurrently to completion.
- **Method**:
  ```bash
  python examples/dynamic_topology/demo.py
  ```
- **Result**: Manager generates N tasks -> N worker vertices execute concurrently -> all payloads collect at sink; runtime graph expansion completes cleanly.
