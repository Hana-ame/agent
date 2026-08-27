# Vertex-Edge Agent Framework

[English](README.md) | [中文文档](README_CN.md)

A **non-interactive**, data-driven, highly-scalable DAG (Directed Acyclic Graph) execution engine designed specifically for orchestrating and scheduling production-grade AI Agent pipelines.

## Unified Architecture

The framework adopts a highly unified **Actor / Message-Passing** model. There are no scattered method calls between Vertex (node) and Edge (edge); all interactions are completed solely through a single signal pipe, `handle_edge_signal`.

```
┌──────────┐    ┌───────────────────────────────────────────────┐    ┌──────────┐
│ Vertex A │───▶│                     Edge 1                    │───▶│ Vertex B │
│ (Source) │    │ Guard -> PreProcess -> Compute -> PostProcess │    │ (Sink)   │
└──────────┘    └───────────────────────────────────────────────┘    └──────────┘
```

### 1. Edge: Unified 5-Stage Pipeline

An `Edge` is no longer differentiated between ordinary or conditional edges, but unified into a standard 5-stage pipeline:
1. **Guard (Interception)**: Calls `evaluate_condition` for pre-validation (supporting JSON declarative rules or external Python scripts). If the condition is not met, an `ABORTED` signal is directly generated, triggering downstream cascading branch pruning to avoid deadlocks.
2. **Pre-Process (Preprocessing)**: Triggers the `pre_process` hook to process raw data.
3. **Compute (Computation)**: If Prompt and Model are configured, it computes via an LLM (PI Agent); if not, it acts as a transparent Pass-through edge, transmitting data directly.
4. **Post-Process (Postprocessing)**: Triggers the `post_process` hook to parse or format the result.
5. **Deliver (Delivery)**: Sends a `COMPLETED` signal to the target Vertex and writes the result.

### 2. Vertex: Unified 3-Stage Container

As a pure black-box state machine container, a `Vertex` goes through three lifecycles:
1. **Ingest**: When a `COMPLETED` signal is received from an edge, the `on_receive` interceptor/hook is triggered.
2. **Settle (Settlement/Barrier)**: Employs a dynamic Settlement Barrier Check. Real-time statistics are kept for `COMPLETED` and `ABORTED` signals. Once all incoming edges have resolved, it transitions to `READY` if at least one succeeded; if all failed, it transitions to `ABORTED`.
3. **Fuse (Fusion)**: Once settlement is complete, the engine triggers `prepare_outputs()` (the `on_ready` hook) to fuse scattered data into the states required by outgoing edges.

## Configuration Schema

Graph topology and execution rules are entirely JSON-driven, supporting declarative threshold control, script binding, and LLM configuration:

```jsonc
{
  "metadata": { "name": "...", "description": "..." },
  "vertices": [
    {
      "id": "v1",
      "settings": { /* Arbitrary config dictionary */ },
      "script": "path/to/vertex_script.py",      // Optional: Bind external extension script
      "initial_data": [                          // Optional: Initial injected data
        { "data_id": "text", "tags": ["en"], "value": "Hello" }
      ]
    }
  ],
  "edges": [
    {
      "id": "e1",
      "source": "v1",
      "destination": "v2",
      "data_id": "text",
      "tags": ["en"],
      "prompt": "Summarize this:",
      "model": "gemini-pro",
      "settings": {
        "threshold": 80,                         // Optional: Guard threshold config (declarative)
        "operator": ">="
      },
      "script": "path/to/edge_script.py"         // Optional: Bind external extension script
    }
  ]
}
```

## External Scripts

By configuring the `script` field, you can instantly upgrade ordinary nodes and edges with complex logic without modifying the core framework source code.

### Vertex Scripts

```python
def on_receive(data, data_id, tags, settings):
    """[Ingest Phase] Triggered when data arrives. Can transform data or raise exceptions to reject it."""
    if not valid(data):
        raise ValueError("rejected")
    return data.upper()

def on_ready(all_data, settings):
    """[Fuse Phase] Called when the node is ready, right before triggering downstream outgoing edges. Used to fuse multiple inputs into the final output."""
    return {("output_id", ("tag",)): merged_value}
```

### Edge Scripts

```python
def guard(data, settings):
    """[Guard Phase] Condition threshold. Returns False to prune the current branch. Also known as evaluate_condition."""
    return data.get("score", 0) >= 80

def pre_process(data, settings):
    """[Pre-process Phase] Transforms data before entering the LLM."""
    return f"[Please analyze the following content]\n{data}"

def post_process(data, settings):
    """[Post-process Phase] Parses the output of the LLM."""
    return data.strip()
```

```python
import asyncio
from framework import Graph, Executor, MockAgent

async def main():
    # 1. Parse graph configuration (supports DAGs, loops, and conditional branches)
    graph = Graph.from_json("config.json")
    
    # 2. Option A: Standard run
    result = await Executor(graph, MockAgent(), max_concurrency=8).run()
    print(result.summary())
    
    # 3. Option B: Real-time event streaming
    # executor = Executor(graph, MockAgent())
    # async for event in executor.stream():
    #     print(f"[{event.timestamp}] {event.event_type} - vertex={event.vertex_id}")

asyncio.run(main())
```

## Advanced Features (v2.0)

### 1. Business Logic Retry & Self-Correction
Edges can automatically catch domain errors in `post_process`, inject corrective feedback into the LLM prompt, and retry with exponential backoff:
```jsonc
{
  "id": "e_extract",
  "source": "v1",
  "destination": "v2",
  "prompt": "Extract valid JSON",
  "settings": {
    "retry_policy": {
      "max_retries": 3,
      "backoff_factor": 1.0,
      "retry_on": ["KeyError", "JSONDecodeError", "ValueError"]
    }
  }
}
```

### 2. State Checkpointing & Human-in-the-Loop (HITL)
Workflows can pause for approval either declaratively (`"require_approval": true`) or dynamically via `vertex.pause_for_approval()`. State can be snapshotted to SQLite via `SQLiteStateStore` and resumed via `CheckpointedExecutor.resume()`.

### 3. Stateful Loops & Cycles
Workflows support cyclic graph topologies for iterative refinement. Cycles are validated against back-edges configured with `max_iterations > 0` to prevent infinite loops.

### 4. Real-Time Non-Blocking Event Streaming
Observe graph execution live using `async for event in executor.stream()` emitting structured `GraphEvent` records without blocking core execution.


---

## ⚙️ Configuration Guide (Vertex & Edge)

The framework is driven by JSON configurations. You can define the topology in a `.json` file and load it via `Graph.from_json_file()`.

### 1. Vertex Configuration

Vertices are the state machine containers. Defined in the `vertices` array:

| Field | Type | Required | Default | Description & Options |
| :--- | :--- | :---: | :--- | :--- |
| **`id`** | `str` | **Yes** | - | Unique identifier (e.g., `"DataIngest"`). |
| **`type`** | `str` | No | `"vertex"` | `"vertex"` (standard node) or `"subgraph"` (nested subgraph). |
| **`initial_data`** | `list[dict]` | No | `[]` | Initial data injected into the node. Each dict must have `channel` and `value`. |
| **`script`** | `str` | No | `null` | Path to a Python script to inject `on_receive` or `on_ready` hooks. |
| **`settings`** | `dict` | No | `{}` | Advanced business logic settings. |

**Advanced `settings`:**
* `"require_approval"`: (`bool`) Set to `true` to enable Human-in-the-Loop (HITL), pausing execution and saving a snapshot.
* `"graph_config"`: (`str` / `dict`) **Required for `type="subgraph"`**. Path to the subgraph's `.json` configuration file.
* `"input_map"` / `"output_map"`: Port mapping redirects for nested subgraphs.

### 2. Edge Configuration

Edges are the 5-stage compute and routing pipelines. Defined in the `edges` array:

| Field | Type | Required | Default | Description & Options |
| :--- | :--- | :---: | :--- | :--- |
| **`id`** | `str` | **Yes** | - | Unique identifier (e.g., `"e_analyze"`). |
| **`source`** | `str` | **Yes** | - | Source Vertex ID. |
| **`destination`** | `str` | **Yes** | - | Destination Vertex ID. |
| **`channel`** | `str` | No | `"default"` | Channel name for data flow. |
| **`prompt`** | `str` | No | `""` | The prompt for the LLM. If empty, the edge is a **transparent pass-through**. |
| **`model`** | `str` | No | `"default"`| LLM model name (e.g., `"gemini-1.5-pro"`). |
| **`max_iterations`** | `int` | No | `0` | **Cycle bound**: Set `> 0` to mark as a back-edge, allowing `N` iterations. |
| **`script`** | `str` | No | `null` | Path to a Python script to inject `pre_process`/`post_process` hooks. |
| **`settings`** | `dict` | No | `{}` | Settings for guards, self-correction, and global memory. |

**Advanced `settings`:**
* **Conditional Routing (Guard)**: `"threshold"`, `"operator"` (e.g., `">="`), `"field"`. Triggers an `ABORTED` prune if conditions fail.
* **Self-Correction (`retry_policy`)**: E.g., `{"max_retries": 3, "retry_on": ["KeyError"]}`.
* **Global Memory**: `"memory_read"` (array of keys to read), `"memory_write"` (dict mapping output fields to global keys).

## Enterprise-Grade Features (v3.0)

### 1. Hierarchical Nested Sub-Graphs (`SubgraphVertex`)
Encapsulate multi-agent teams as single modular nodes in a parent graph. Features automatic `input_map`/`output_map` translation, namespaced checkpoint persistence, and event bubbling (`subgraph_*`).

### 2. Global Memory & Shared Context (`MemoryStore`)
A thread-safe key-value bus allowing distant nodes to read and write shared state without routing clutter:
```jsonc
{
  "id": "e_auth",
  "source": "Login",
  "destination": "Dashboard",
  "settings": {
    "memory_write": { "session_token": "global_session_id" },
    "memory_read": [ "user_permissions" ]
  }
}
```

### 3. Granular Telemetry & Cost Profiling (`TelemetryTracker`)
Automatically tracks prompt tokens, completion tokens, execution latency, and estimated dollar costs per edge and workflow-wide with built-in model pricing catalogs (OpenAI, Gemini, Claude).

## 💻 Official Examples

The `examples/` directory provides standalone, runnable demonstrations of the framework's core features. They serve as reference implementations for configuring Nodes, Edges, and the Execution API.

### 1. Available Examples & Capabilities

| Directory | Purpose / Feature Showcased | Command |
| :--- | :--- | :--- |
| **`realtime_streaming/`** | Demonstrates the non-blocking `executor.stream()`, capturing `GraphEvent` records and rendering ANSI colored logs. | `python examples/realtime_streaming/demo.py` |
| **`self_correction/`** | Simulates LLM formatting errors to trigger `retry_policy`. Injects error stack traces back into the Prompt for LLM self-healing. | `python examples/self_correction/demo.py` |
| **`hitl_approval/`** | Shows how `require_approval` pauses execution at sensitive nodes, saves SQLite state snapshots, and resumes via `approve()`. | `python examples/hitl_approval/demo.py` |
| **`subgraph/`** | Demonstrates hierarchical nesting. A parent graph imports a `research_team.json` subgraph, routing inputs/outputs via boundary mapping. | `python examples/subgraph/demo.py` |

### 2. Configuring an Example from Scratch

To build a custom multi-agent workflow from zero:

1. **Setup Directory Structure**:
   ```text
   my_agent/
   ├── config.json         # Required: Topology definition
   ├── run.py              # Required: Executor entrypoint
   └── hooks.py            # Optional: Custom logic scripts
   ```

2. **Define Topology (`config.json`)**:
   Declare source vertices, sink vertices, and connecting edges. To attach python hooks, set `"script": "hooks.py"`.

3. **Write Scripts (`hooks.py`)**:
   ```python
   def pre_process(data, settings):
       # Process edge data
       return data
   ```

4. **Write Entrypoint (`run.py`)**:
   ```python
   import asyncio
   from framework import Graph, Executor, HttpLLMAgent
   
   async def main():
       # Setup Agent (Requires API Key ENV vars)
       agent = HttpLLMAgent()
       
       graph = Graph.from_json_file("config.json")
       executor = Executor(graph, agents=agent)
       await executor.run()
   
   asyncio.run(main())
   ```

### 3. Precautions & Best Practices

1. **Path Resolution**: Paths defined in JSON (`"script"` or `"graph_config"`) are resolved relative to the **Current Working Directory (CWD)** where the script is executed. 
2. **Deadlock Prevention**: If an edge has a conditional guard (`threshold`), ensure there is a fallback edge, or that cascaded `ABORTED` signals are safely handled. Otherwise, downstream nodes may wait infinitely for data that will never arrive.
3. **Infinite Loop Protection**: Any edge that creates a topological cycle (a back-edge) **must** explicitly configure `"max_iterations": N`. Failure to do so will result in a `GraphCycleError` during initialization.
4. **LLM Agents**: Many examples use a `MockAgent` for predictable testing. For real-world usage, switch to `HttpLLMAgent` and provide necessary environment variables (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`).

## Tests

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

Currently contains **125 fully covered tests**, covering:
- Actor state machines & `EdgeSignal` unified message passing
- 5-stage `EdgePipeline` execution & error isolation
- Declarative threshold control, custom guards, and diamond branch pruning
- Bounded stateful cycles & loop-back iteration re-entry
- SQLite snapshot persistence, crash recovery, and HITL approval resumes
- Business-logic retry policies & self-correction prompt reflections
- Real-time sidecar event streaming and concurrency semaphore limits
- Hierarchical nested sub-graphs (`SubgraphVertex`) & event bubbling
- Global shared memory bus (`MemoryStore`), TTLs, and scoped namespaces
- Token usage tracking, latency benchmarking, and cost profiling
