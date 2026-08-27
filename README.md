# Vertex-Edge Agent Framework

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

## Examples

```bash
# Simple linear pipeline
python examples/run.py examples/simple/config.json

# Complex DAG (Supports Fan-out, Fan-in, external scripts)
python examples/run.py examples/complex/config.json

# Dynamic branch routing and conditional pruning (Guard & Routing)
python examples/run.py examples/conditional_routing/config.json

# Advanced Object-Oriented usage (Custom subclass overrides)
python examples/run.py examples/custom_classes/config.json
```
*Each example folder contains a dedicated `README.md` tutorial.*

## Tests

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

Currently contains **114 fully covered tests**, covering:
- Actor state machines & `EdgeSignal` unified message passing
- 5-stage `EdgePipeline` execution & error isolation
- Declarative threshold control, custom guards, and diamond branch pruning
- Bounded stateful cycles & loop-back iteration re-entry
- SQLite snapshot persistence, crash recovery, and HITL approval resumes
- Business-logic retry policies & self-correction prompt reflections
- Real-time sidecar event streaming and concurrency semaphore limits
