# Simple Chain — Programmatic Topology Without JSON

> Documented following the "Problem / Solution / Changes / Verification" format: demonstrates creating an `A -> B -> C` pipeline programmatically without manual JSON definitions.

## Problem

Manually authoring `config.json` for a simple sequential pipeline can be verbose (metadata, vertices, edges, and settings). For common linear chains, a concise programmatic API is desirable.

## Solution

Provide `LinearChain.build(prompts: List[str]) -> Graph`, where the length of `prompts` determines the edge count, automatically constructing the linear `A -> B -> C...` topology and assigning prompts per edge.

## Changes

- `framework/builders/chain.py`: Implements `LinearChain.build(prompts)`.
- `examples/simple_chain/demo.py`: Invokes `LinearChain.build(["Step1", "Step2"])` and executes it with `Executor(graph)`.

## Verification

- **Test Plan**: Verify prompts automatically generate N+1 vertices and N edges.
- **Method**:
  ```bash
  python examples/simple_chain/demo.py
  ```
- **Result**: `A -> B -> C` graph constructed and executed successfully (covered in `tests/test_improvements.py`).
