# Examples

This directory contains **19 executable examples** (along with `scripts/` and `s1profile_collect/` as helper directories).
Each example contains a `README.md` following the "Problem / Solution / Changes / Verification" format detailing the problem addressed, design, code modifications, and verification results.

> **Two runners — pick by manifest version:**
>
> - **V4** (recommended): `python examples/run_v4.py <example>/config.json`, or the
>   installed console script `vea-run-v4 <config.json>` (identical entry point).
>   It loads the manifest, seeds SQLite and executes in one call; add
>   `--script-root DIR` (repeatable) for edge classes outside the repository.
> - **V1** (legacy, unchanged): `python examples/run.py <example>/config.json`,
>   or the directory's standalone `demo.py` / `run.py`.
>
> Each runner rejects the other stack's manifests with a message naming the correct
> one.

## Overview Index

| Example | Problem Addressed | Execution Command |
|---|---|---|
| `simple/` | Minimal 3-node sequential execution pipeline | `python examples/run.py examples/simple/config.json` |
| `complex/` | Multi-source fan-out / fan-in with external subclass scripts | `python examples/run.py examples/complex/config.json` |
| `conditional_routing/` | Guard conditional dispatch and cascade pruning without deadlock | `python examples/run.py examples/conditional_routing/config.json` |
| `custom_classes/` | Dynamic subclass loading for custom Vertex and Edge implementations | `python examples/run.py examples/custom_classes/config.json` |
| `real_llm/` | Real LLM endpoint configuration with transport proxies | `python examples/run.py examples/real_llm/config.json` |
| `real_pi/` | Subprocess delegation to local `pi` CLI agent | `python examples/run.py examples/real_pi/config.json` |
| `opencode_zen/` | Subprocess delegation to local `opencode` CLI agent | `python examples/opencode_zen/run.py` |
| `sensenova/` | Direct connection to SenseNova inference endpoints | `python examples/run.py examples/sensenova/config.json` |
| `realtime_streaming/` | Non-blocking event stream observability | `python examples/realtime_streaming/demo.py` |
| `self_correction/` | Business-level retry policies with self-correcting feedback loops | `python examples/self_correction/demo.py` |
| `hitl_approval/` | Human-in-the-loop (HITL) pause and SQLite checkpoint resumption | `python examples/hitl_approval/demo.py` |
| `subgraph/` | Nested subgraph execution with input/output boundary translation | `python examples/subgraph/demo.py` |
| `simple_chain/` | Programmatic graph construction via fluent builder without JSON | `python examples/simple_chain/demo.py` |
| `dynamic_topology/` | Dynamic runtime graph growth and edge creation | `python examples/dynamic_topology/demo.py` |
| `custom_edge/` | Zero-registration custom edge classes via `"type": "my_edges.py:MyEdge"` | `python examples/custom_edge/demo.py` |
| `race_mode/` | Race execution (first-to-finish wins, cancels laggards) | `python examples/race_mode/demo.py` |
| `hn_ai_report/` | Hacker News automated AI summary report (legacy MapEdge) | `python examples/hn_ai_report/demo.py` |
| `s1_ai_report/` | Direct multi-fetch AI report (8-way parallel fan-out) | `python examples/s1_ai_report/demo.py` |
| `s1_ai_report_map/` | Dynamic MapEdge-based AI report generator | `python examples/s1_ai_report_map/demo.py` |
| `finance_ai_report/` | Financial AI report with domain filtering (legacy MapEdge) | `python examples/finance_ai_report/demo.py` |

## V4 Production Pipelines & Architectural References

| V4 Example | Role & Capabilities Demonstrated | Execution Command |
|---|---|---|
| **`run_v4.py`** *(Generic runner)* | **Run any V4 manifest**: one-call load + SQLite seeding + execution, `--script-root` for edge classes outside the repo, `--db` for persistence, `--json` / `--print-events` output. Thin wrapper over `framework/run_v4.py` (`vea-run-v4`). | `python examples/run_v4.py <config.json>` |
| **`hn_v4/`** *(Canonical)* | **Recommended Full-Stack V4 Pipeline**: multi-branch Fan-In settlement barrier (`MergeStrategyV4.JSON_MERGE`), self-healing `ReflexiveEdgeV4` on network reject, SQLite edge metrics, and live SSE event streaming. | `python examples/hn_v4/run.py` |
| `hn_ai_report_v4/` | **Migration Study**: Comparative reference demonstrating how legacy v1 `MapEdge` fan-out translates into V4 `CodeEdgeV4` internal `asyncio.gather`. | `python examples/hn_ai_report_v4/demo.py` |
| `subgraph_v4/` | **Discrete Graph Architecture**: Discrete JSON loading, directory-level manifest ingestion, dynamic runtime subgraph splicing and insertion. | `python examples/subgraph_v4/run.py` |
| `sensenova_v4/` | **SenseNova LLM Integration**: Production SenseNova flash-lite / chat / generate edge orchestration. | `python examples/sensenova_v4/run.py` |
| `dynamic_tool_library/` | **Intent Router & Sandboxed Tools**: Declarative tool catalog with intent routing and SSE agent stepping. | `python examples/dynamic_tool_library/demo.py` |

Helper directories: `scripts/` (shared subclass scripts), `s1profile_collect/` (data collection helper).

---

## Issue: Example Count Discrepancy (16 -> 19)

### Problem
The README formerly stated "16 examples", omitting `s1_ai_report_map`, `sensenova`, and `finance_ai_report`. Additionally, `complex` and `custom_classes` descriptions referred to obsolete "module hooks" terminology.

### Solution
Update example counts to match the actual directory listing and modernize terminology to "subclass loading via `script`".

### Changes
- Updated table count to 19; added rows for `s1_ai_report_map`, `sensenova`, and `finance_ai_report`; updated descriptions for `complex` and `custom_classes`.

### Verification
- **Test Plan**: Assert table rows match the number of example subdirectories.
- **Method**: Compare `ls -d examples/*/` count (19 examples + 2 helper directories) with index table row count.
- **Result**: Exactly 19 example rows matching 19 directories.

---

## Issue: Parallel Execution Divergence in `real_llm` and `real_pi`

### Problem
Early versions of `real_llm` used raw `urllib` calls bypassing the framework agent model, while `real_pi` injected `PiAgentRunner` delegating to local CLI tools. Having multiple divergent agent integration patterns caused confusion.

### Solution
Standardize on the pattern where script Edges instantiate and hold their agent directly in `__init__`: `llm_edge.py:HttpLLMEdge`, `pi_edge.py:PiEdge`.

### Changes
- `real_llm/llm_edge.py`, `real_pi/pi_edge.py`: Initialized `self.agent = HttpLLMAgent` and `self.agent = PiAgentRunner` in `__init__`; removed framework injection assumptions.
- Synchronized documentation in `examples/README.md`.

### Verification
- **Test Plan**: Verify CLI and HTTP agents are held directly by script edges.
- **Method**: `grep -n "self.agent" examples/real_llm/llm_edge.py examples/real_pi/pi_edge.py examples/opencode_zen/zen_edge.py`.
- **Result**: All three instantiate their runner in `__init__`; tests pass (`tests/test_agents.py`).

---

## Running Examples

```bash
# General config-driven examples
python examples/run.py examples/simple/config.json
python examples/run.py examples/conditional_routing/config.json

# Examples with standalone demo scripts
python examples/realtime_streaming/demo.py
python examples/race_mode/demo.py
```

> For detailed issue breakdowns and architecture notes, consult the `README.md` in each example directory. Sample execution outputs are recorded in `report.md` files.
