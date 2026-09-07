# Simple Example — Minimal 3-Node Sequential Pipeline

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates the minimal runnable structure of the framework.

---

## Problem

New users need a minimal topology to understand the flow of data across `Vertex -> Edge -> Vertex` without being overwhelmed by advanced concepts like fan-out, conditional routing, or subgraphs.

## Solution

Build a 3-node sequential graph (`input -> processor -> output`), with one LLM step per edge:
- `input`: Injects initial data (`initial_data`); having no incoming edges, it is automatically marked `READY`.
- `processor` / `output`: Intermediate processing and sink nodes.
- `e1` / `e2`: Standard `Edge` instances that route data through the LLM (using the framework's default MockAgent with `[hy3-free]` prefix).

## Changes

- `examples/simple/config.json`: 3 vertices + 2 edges topology.
- No custom scripts required—entirely config-driven.

## Verification

- **Test Plan**: Verify root vertex is automatically READY, data flows sequentially across edges, and sink node finishes.
- **Method**:
  ```bash
  python examples/run.py examples/simple/config.json
  ```
- **Result**: `input` becomes READY -> `e1` runs MockAgent with `[hy3-free]` prefix -> `processor` receives data -> `e2` processes data -> `output` receives final payload -> graph transitions to `DONE`.

## Data Flow (Trace)

```
input(RDY) --e1--> processor(RDY) --e2--> output(RDY) --settle--> DONE
```
