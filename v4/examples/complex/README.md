# Complex Example — Multi-Source Fan-out / Fan-in with External Subclasses

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates concurrent multi-input ingestion, dependency joining, and external subclass integration.

---

## Problem

A representative medium-complexity example is needed to demonstrate three core capabilities:
1. **Multi-Source Concurrency**: Multiple input vertices providing initial payloads concurrently.
2. **Fan-out & Fan-in**: A single vertex broadcasting across multiple edges, with a downstream merge vertex awaiting all incoming edges.
3. **External Subclass Loading**: Transformation logic encapsulated in modular `.py` script files rather than embedded strings or top-level hooks.

## Solution

**Topology**:

```
input_a ─e1─▶ transform ─e3─▶ merge ─e5─▶ output
input_a ─e4─────────┬───────────▶ merge
input_b ─e2─────────▶ transform
```

- `input_a` / `input_b`: Dual source inputs.
- `transform`: Attached to `script: ../scripts/uppercase_handler.py` (`UpperVertex` subclass), transforming received strings to uppercase in `on_receive`.
- `merge`: Aggregation node (demands both `e3` and `e4` to settle before transitioning to READY via Settlement Barrier).
- `e3`: Attached to `script: ../scripts/prefix_handler.py` (`PrefixEdge` subclass), adding `[PRE]` prefix in `pre_process` and `[POST]` suffix in `post_process`.

## Changes

- `examples/complex/config.json`: 4 vertices + 5 edges topology; `script` fields referencing `../scripts/*.py`.
- `examples/scripts/uppercase_handler.py`: Subclasses `UpperVertex(Vertex)` implementing `on_receive` and `on_ready`.
- `examples/scripts/prefix_handler.py`: Subclasses `PrefixEdge(Edge)` implementing `pre_process` and `post_process`.
- All custom behavior is organized strictly as subclass overrides without top-level functions.

## Verification

- **Test Plan**: Multi-source concurrency, uppercase transformation, dual-input merge settlement, and prefix/suffix application.
- **Method**:
  ```bash
  python examples/run.py examples/complex/config.json
  ```
- **Result**:
  - `input_a` and `input_b` enter concurrently; `transform` receives both payloads and converts them to uppercase.
  - `merge` transitions to READY only after both `e4` (`input_a -> merge`) and `e3` (`transform -> merge`) arrive (barrier synchronization via `EdgeSignal`).
  - `e5` outputs text tagged with `[PRE]...[POST]`; `output` settles and the graph reaches DONE.

> This example demonstrates external subclasses as the recommended extension pattern; `custom_classes` is a simplified variant.
