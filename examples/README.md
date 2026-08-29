# Examples

This directory contains 16 independent example pipelines demonstrating the core capabilities of the `vertex-edge-agent` framework, from basic concepts to production-grade advanced applications.

All examples share a unified way of running. You can run the corresponding configuration file directly using `run.py` from the project root:

```bash
# Syntax
python examples/run.py examples/<example_directory>/config.json
```

### Examples Overview

| Example Directory | Key Features | Description |
| :--- | :--- | :--- |
| **`simple/`** | Linear Pipeline | Demonstrates a basic 3-node serial workflow: **Input -> Process -> Output**. It shows simple `Vertex` state transitions and `Edge` processing. |
| **`complex/`** | Complex DAG & Script Hooks | Demonstrates multi-path concurrent graph computation. Includes **Fan-out** (splitting single node data into multiple paths) and **Fan-in** (converging multiple paths and waiting for dependencies). It also shows how to use basic module-level hooks to pre-process/post-process data. |
| **`conditional_routing/`** | Conditional Routing | Demonstrates edge-based Guard interception. It shows threshold-based intent classification, dynamically pruning incorrect branches and triggering a cascading abort (`Cascading Abort`), allowing conditional concurrency without causing deadlocks. |
| **`custom_classes/`** | Native Subclassing | Moving away from traditional module hooks, this showcases the framework's OOP power. It demonstrates how the framework uses the `inspect` module to dynamically recognize and instantiate subclasses of `Vertex` and `Edge` defined in external scripts, facilitating clean data validation and processing logic injection. |
| **`real_llm/`** | Real API Integration | Demonstrates how subclassing `Edge` can completely bypass the built-in Mock testing setup. It directly makes asynchronous HTTP POST requests to a real cloud API (e.g. `opencode.ai`) and retrieves actual LLM responses. |
| **`opencode_zen/`** | **v3.0 Self-Throttling LLM Agents** | Fan-out graph calling OpenCode Zen directly (`OpenCodeAgent`, free + self-limited) and a self-hosted gateway (`ProxiedLLMAgent`, model aliasing) — both wired declaratively from `config.json`. |
| **`realtime_streaming/`** | **v2.0 Real-Time Event Streaming** | Demonstrates live ANSI-colored event streaming via `executor.stream()`, observing vertex state transitions and edge firings non-blockingly as they occur. |
| **`self_correction/`** | **v2.0 LLM Business Retry & Self-Correction** | Demonstrates `retry_policy` capturing post-process `KeyError`/`JSONDecodeError`, reflecting error context into the prompt (`[SYSTEM FEEDBACK: ...]`), and retrying with exponential backoff. |
| **`hitl_approval/`** | **v2.0 Human-in-the-Loop (HITL) Checkpoints** | Demonstrates pausing sensitive operations (`require_approval`), persisting snapshots to SQLite, and resuming the workflow upon human intervention via `gate.approve()`. |
| **`subgraph/`** | **v3.0 Hierarchical Nested Sub-Graphs** | Demonstrates modular multi-agent team delegation (`SubgraphVertex`) with input/output boundary mapping and real-time nested event bubbling. |
| **`simple_chain/`** | Programmatic Linear Topology | Builds `A->B->C` graphs with `LinearChain.build(prompts)` instead of hand-writing JSON — the shortest path to a working workflow. |
| **`dynamic_topology/`** | Async Hooks & Runtime Graph Growth | The Manager vertex emits a task list and the runtime spawns one worker vertex per item, showing that hooks and topology can grow mid-execution. |
| **`race_mode/`** | **v3.0 Race Mode (First-to-Finish)** | Fan-out fan-in graph where the sink wins on the first arriving response (`Executor(..., race_mode=True)`); the losers are cancelled cleanly. |
| **`real_pi/`** | Pi Agent CLI Integration | Same real-API story as `real_llm/` but delegates to the local `pi` CLI subprocess via `PiAgentRunner` — for shops that standardise on Pi. |
| **`hn_ai_report/`** | End-to-End AI Report (SubgraphVertex) | A multi-agent research-team graph that produces `report.md` from scratch; showcases nested sub-graph delegation at production scale. |
| **`s1_ai_report/`** | End-to-End AI Report (HTTP LLM) | Same S1 report pipeline as `hn_ai_report/` but with plain `HttpLLMAgent` edges — useful as a comparison baseline. |

### Running the Demos

```bash
# 1. Real-time event streaming with colored terminal output
python examples/realtime_streaming/demo.py

# 2. LLM error interception & automatic prompt self-correction
python examples/self_correction/demo.py

# 3. Human-in-the-loop approval and SQLite state resumption
python examples/hitl_approval/demo.py

# 4. Nested Sub-Graph (Agent Team) delegation
python examples/subgraph/demo.py

# 5. Race mode — first-to-finish wins, losers are cancelled
python examples/race_mode/demo.py

# 6. Dynamic topology — manager spawns workers at runtime
python examples/dynamic_topology/demo.py

# 7. Programmatic linear chain (no JSON)
python examples/simple_chain/demo.py

# 8. Real LLM via OpenCode Zen + Proxied gateway (self-throttling agents)
python examples/opencode_zen/run.py

# 9. End-to-end AI report via SubgraphVertex
python examples/hn_ai_report/demo.py

# 10. End-to-end AI report via HttpLLMAgent (comparison baseline)
python examples/s1_ai_report/demo.py
```

> [!TIP]
> Each example folder contains its own `README.md` and runnable scripts with code explanations. Feel free to check them out!
