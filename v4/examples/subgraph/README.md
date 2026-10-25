# Subgraph — Nested Subgraphs (Multi-Agent Teams)

> Documented following the "Problem / Solution / Changes / Verification" format: encapsulates an independent graph as a single node in a parent graph. Framework v3.0 feature.

## Problem

When orchestrating multi-agent teams, it is desirable to encapsulate an entire sub-team workflow as a reusable, isolated component within a parent graph, complete with input and output boundary mapping, rather than flattening all internal vertices into a global namespace.

## Solution

`SubgraphVertex`: Encapsulates an independent graph definition (`settings.graph_config` pointing to a subgraph configuration file or embedded dictionary). Boundaries are translated using `input_map` and `output_map`; child events bubble up as `subgraph_*`; checkpoint storage maintains isolated namespaces.

## Changes

- `framework/subgraph.py`: Implements `SubgraphVertex`.
- `examples/subgraph/demo.py`: Parent graph imports a `research_team.json` subgraph with boundary mappings.

## Verification

- **Test Plan**: Verify input/output mapping translation, child event bubbling, and data delivery across boundaries.
- **Method**:
  ```bash
  python examples/subgraph/demo.py
  ```
- **Result**: Parent graph invokes the subgraph team -> `input_map` injects payloads -> internal multi-agent graph coordinates -> `output_map` extracts results -> parent graph resumes; covered in `tests/test_subgraph.py`.
