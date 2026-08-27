# Examples

This directory contains five independent example pipelines demonstrating various core capabilities of the `vertex-edge-agent` framework, from basic concepts to production-grade advanced applications.

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
| **`real_llm/`** | Real API Integration | Demonstrates how subclassing `Edge` can completely bypass the built-in Mock testing setup. It directly uses `urllib` + `asyncio.to_thread` to make asynchronous HTTP POST requests to a real cloud API (e.g. `opencode.ai`) and retrieve actual LLM responses. |

> [!TIP]
> Each example folder contains its own `README.md`, complete with a Mermaid network topology diagram and detailed code explanations. Feel free to check them out!
