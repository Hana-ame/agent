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
      "script": "path/to/vertex_script.py",      // Optional: Path to a Vertex subclass
      "initial_data": [                          // Optional: Initial injected data
        { "channel": "text", "value": "Hello" }
      ]
    }
  ],
  "edges": [
    {
      "id": "e1",
      "source": "v1",
      "destination": "v2",
      "channel": "text",
      "settings": {
        "prompt": "Summarize this:",             // Computation layer inside settings
        "model": "gemini-pro",
        "threshold": 80,                         // Optional: Guard threshold config (declarative)
        "operator": ">="
      },
      "script": "path/to/edge_script.py"         // Optional: Path to an Edge subclass
    }
  ]
}
```

## External Scripts

By configuring the `script` field, you can instantly upgrade ordinary nodes and edges with complex logic without modifying the core framework source code.

The `script` field loads a Python file and instantiates a **subclass** of `Vertex` or `Edge`; the custom behaviour lives in the methods the subclass overrides (`on_receive`, `on_ready`, `pre_process`, `post_process`, …).

### Vertex Scripts

```python
# my_vertex.py — defines a Vertex subclass
from framework.vertex import Vertex

class UpperVertex(Vertex):
    """on_receive: uppercase strings; on_ready: combine all data into result channel."""

    def on_receive(self, data, channel, settings):
        if isinstance(data, str):
            return data.upper()
        return data

    def on_ready(self, all_data, settings):
        return {"result": " | ".join(str(v) for v in all_data.values())}
```

### Edge Scripts

```python
# my_edge.py — defines an Edge subclass
from framework.edge import Edge

class PrefixEdge(Edge):
    """pre_process / post_process as overridden methods."""

    def pre_process(self, data, settings):
        if isinstance(data, str):
            return f"{settings.get('prefix', '[PRE]')} {data}"
        return data

    def post_process(self, result, settings):
        if isinstance(result, str):
            return f"{result} {settings.get('suffix', '[POST]')}"
        return result
```

Reference the script in JSON with `"script": "my_vertex.py"` (auto-discovers the subclass), or `"script": "my_vertex.py:UpperVertex"` when the file contains multiple candidate classes.

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

## Agent Engines

The framework ships several swappable `BaseAgent` implementations. Pick one per graph — or let a script `Edge` subclass own one. The rest of the engine is agent-agnostic.

| Agent | Spec string | Default target | When to use |
| :--- | :--- | :--- | :--- |
| `MockAgent` | `"mock"` | — | Tests / dry runs. Echoes data with model metadata. |
| `HttpLLMAgent` | `"http"` | OpenCode Zen | Generic OpenAI-compatible endpoint; **unbounded** — the escape hatch when you drive your own concurrency. |
| `OpenCodeAgent` | `"opencode"` | `https://opencode.ai/zen/v1` | Free, key-less LLM via OpenCode Zen. **Self-throttled** for the free tier. |
| `PiAgentRunner` | `"pi"` | local `pi` CLI | Delegate to the installed Pi Agent CLI subprocess. |
| *custom* | `"path/to/script.py:ClassName"` | — | Subclass `BaseAgent` and load it. |

```python
from framework import Graph, Executor, OpenCodeAgent

# Free-tier Zen, self-limited to 3 concurrent calls / 20 per minute.
agent = OpenCodeAgent(max_concurrency=3, requests_per_minute=20.0)
executor = Executor(graph, agents=agent)
```

**Throttling knobs** (`OpenCodeAgent`):

* `max_concurrency` — `asyncio.Semaphore` bounding in-flight calls. A graph with 32 concurrent edges queues locally instead of opening 32 simultaneous connections.
* `requests_per_minute` — token-bucket budget charged **per attempt**, so retries count against the budget rather than re-entering an already-exhausted endpoint. `None` disables it.
* `queue_timeout` — fail fast with `ThrottleTimeoutError` (a `ComputeError`) instead of hanging the graph forever.

Both gates are agent-local, so each edge's `settings` still decides its own `prompt`/`model` — only *when* it gets to speak is coordinated.

```jsonc
// config.json — an edge loads its agent via `script: file:Class`
{ "id": "e_zen", "source": "v1", "destination": "v2",
  "settings": { "prompt": "Summarise.", "model": "hy3-free" },
  "script": "zen_edge.py:OpenCodeEdge" }
```

The edge script owns its agent in Python (e.g. `self.agent = OpenCodeAgentRunner()` in `__init__`); nothing is injected by the runner and no fallback default agent is used.

**Transport proxy** (HTTP 请求经代理出去): every HTTP agent (`HttpLLMAgent`, `OpenCodeAgent`) accepts a `proxy` URL — the HTTP(S)/SOCKS proxy the request *tunnels through* on its way to the endpoint.

```python
from framework import HttpLLMAgent

# Every HTTP request physically goes through corp-proxy:3128
agent = HttpLLMAgent(proxy="http://user:pass@corp-proxy:3128")
```

**在 graph.json 中设置代理（覆盖环境变量）:** the edge settings accept `proxy`, `https_proxy` or `HTTPS_PROXY` — same meaning — and the script edge forwards them to its agent. Setting it in the graph config **overrides** any `HTTP_PROXY` / `HTTPS_PROXY` from the environment, so a pipeline can pin its own egress proxy regardless of the shell it runs in:

```jsonc
{ "id": "e_real_llm", "source": "v1", "destination": "v2",
  "script": "llm_edge.py:HttpLLMEdge",
  "settings": { "prompt": "Summarise.", "model": "hy3-free",
                "https_proxy": "http://127.0.1.6:7890" } }
```

When the graph config leaves `proxy` unset, `trust_env=True` (default) lets httpx fall back to `HTTP_PROXY` / `HTTPS_PROXY` from the environment. Explicit config > environment.

## Advanced Features (v2.0)

### 1. Business Logic Retry & Self-Correction
Edges can automatically catch domain errors in `post_process`, inject corrective feedback into the LLM prompt, and retry with exponential backoff:
```jsonc
{
  "id": "e_extract",
  "source": "v1",
  "destination": "v2",
  "settings": {
    "prompt": "Extract valid JSON",
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

The framework is driven by JSON configurations. You can define the topology in a `.json` file and load it via `Graph.from_json()`.

### 1. Vertex Configuration

Vertices are the state machine containers. Defined in the `vertices` array:

| Field | Type | Required | Default | Description & Options |
| :--- | :--- | :---: | :--- | :--- |
| **`id`** | `str` | **Yes** | - | Unique identifier (e.g., `"DataIngest"`). |
| **`type`** | `str` | No | `"vertex"` | `"vertex"` (standard node) or `"subgraph"` (nested subgraph). |
| **`initial_data`** | `list[dict]` | No | `[]` | Initial data injected into the node. Each dict must have `channel` and `value`. |
| **`script`** | `str` | No | `null` | Path to a Python script defining a `Vertex` subclass (e.g. `my_vertex.py` or `my_vertex.py:ClassName`). |
| **`settings`** | `dict` | No | `{}` | Advanced business logic settings. |

**Advanced `settings`:**
* **Computation Layer**: `"prompt"` (instruction to the LLM), `"model"` (LLM model name, e.g. `"gemini-1.5-pro"`).
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
| **`max_iterations`** | `int` | No | `0` | **Cycle bound**: Set `> 0` to mark as a back-edge, allowing `N` iterations. |
| **`script`** | `str` | No | `null` | Path to a Python script defining an `Edge` subclass (e.g. `my_edge.py` or `my_edge.py:ClassName`). |
| **`settings`** | `dict` | No | `{}` | Contains `prompt`, `model`, and settings for guards, self-correction, and global memory. |

**Advanced `settings`:**
* **Computation Layer**: `"prompt"` (instruction to the LLM), `"model"` (LLM model name, e.g. `"gemini-1.5-pro"`).
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

### 4. Race Mode (First-to-Finish)
Support for `wait_policy: 'any'` on vertices to aggressively short-circuit execution. Once the first incoming edge satisfies the vertex, it immediately fires downstream routes and actively cancels all pending upstream `asyncio` tasks to minimize API costs and latency.

### 5. Async Hooks & Dynamic Topologies
Pipeline hooks (`pre_process`, `post_process`) natively support `async def` for I/O bound operations. The `LinearChain.build(prompts)` API enables rapid, programmatic construction of `A->B->C` topologies without JSON configuration.

### 6. Type-Safe Schema Validation
Pydantic integration via `SchemaRegistry` enforces data consistency across edges. It provides static graph compilation checks and runtime data validation, automatically routing `ValidationError`s to the LLM self-correction retry policy.

## 💻 Official Examples

The `examples/` directory provides 18 standalone, runnable demonstrations of the framework's core features. They serve as reference implementations for configuring Nodes, Edges, and the Execution API. See [`examples/README.md`](examples/README.md) for the full index; the highlights are:

| Directory | Purpose / Feature Showcased | Command |
| :--- | :--- | :--- |
| **`realtime_streaming/`** | Demonstrates the non-blocking `executor.stream()`, capturing `GraphEvent` records and rendering ANSI colored logs. | `python examples/realtime_streaming/demo.py` |
| **`self_correction/`** | Simulates LLM formatting errors to trigger `retry_policy`. Injects error stack traces back into the Prompt for LLM self-healing. | `python examples/self_correction/demo.py` |
| **`hitl_approval/`** | Shows how `require_approval` pauses execution at sensitive nodes, saves SQLite state snapshots, and resumes via `approve()`. | `python examples/hitl_approval/demo.py` |
| **`subgraph/`** | Demonstrates hierarchical nesting. A parent graph imports a `research_team.json` subgraph, routing inputs/outputs via boundary mapping. | `python examples/subgraph/demo.py` |
| **`opencode_zen/`** | **v3.0** Launches the local `opencode` CLI (`opencode run`) via `OpenCodeAgentRunner`; edge loaded by `script: zen_edge.py:OpenCodeEdge`, everything declared in the config. | `python examples/opencode_zen/run.py` |
| **`race_mode/`** | **v3.0** First-to-finish fan-in: the sink wins on the first response and cancels the losers cleanly. | `python examples/race_mode/demo.py` |
| **`dynamic_topology/`** | Async hooks and manager-driven runtime graph growth — one worker vertex per task the Manager emits. | `python examples/dynamic_topology/demo.py` |
| **`simple_chain/`** | Programmatic `LinearChain.build(prompts)` for the shortest path to a working `A->B->C` graph (no JSON). | `python examples/simple_chain/demo.py` |
| **`real_pi/`** | Real-LLM flow that delegates to the local `pi` CLI subprocess via `PiAgentRunner` (Pi-stdlib counterpart of `real_llm/`). | `python examples/run.py examples/real_pi/config.json` |
| **`real_llm/`** | Real-LLM call (`hy3-free`) through a transport `https_proxy` pinned in the edge settings, overriding env proxies — edge loaded by `script: llm_edge.py:HttpLLMEdge` owning `HttpLLMAgent`. | `python examples/run.py examples/real_llm/config.json` |
| **`hn_ai_report/`** | End-to-end S1 AI report graph on `SubgraphVertex` delegation. | `python examples/hn_ai_report/demo.py` |
| **`s1_ai_report/`** | Same S1 report graph on plain `HttpLLMAgent` — a comparison baseline against `hn_ai_report/`. | `python examples/s1_ai_report/demo.py` |

### 2. Configuring an Example from Scratch

To build a custom multi-agent workflow from zero:

1. **Setup Directory Structure**:
   ```text
   my_agent/
   ├── config.json         # Required: Topology definition
   ├── run.py              # Required: Executor entrypoint
   └── my_nodes.py         # Optional: Custom Vertex/Edge subclasses
   ```

2. **Define Topology (`config.json`)**:
   Declare source vertices, sink vertices, and connecting edges. To attach custom logic, set `"script": "my_nodes.py"` (or `"my_nodes.py:ClassName"`) on the node.

3. **Write Subclasses (`my_nodes.py`)**:
   ```python
   from framework.edge import Edge

   class MyEdge(Edge):
       def pre_process(self, data, settings):
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
       
       graph = Graph.from_json("config.json")
       executor = Executor(graph, agents=agent)
       await executor.run()
   
   asyncio.run(main())
   ```

### 3. Precautions & Best Practices

1. **Path Resolution**: Paths defined in JSON (`"script"` or `"graph_config"`) are resolved relative to the **directory of the config file** that references them. 
2. **Deadlock Prevention**: If an edge has a conditional guard (`threshold`), ensure there is a fallback edge, or that cascaded `ABORTED` signals are safely handled. Otherwise, downstream nodes may wait infinitely for data that will never arrive.
3. **Infinite Loop Protection**: Any edge that creates a topological cycle (a back-edge) **must** explicitly configure `"max_iterations": N`. Failure to do so will result in a `GraphCycleError` during initialization.
4. **LLM Agents**: Many examples use a `MockAgent` for predictable testing. For real-world usage, use `OpenCodeAgent` (free OpenCode Zen, self-throttled) or `HttpLLMAgent` (generic, unbounded) — and provide the necessary environment variables (e.g. `OPENAI_API_KEY`).

## Tests

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

Currently contains **333 fully covered tests**, covering:
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
- Race Mode cancellation (`wait_policy: 'any'`), async hooks, and `LinearChain.build`
- Static and Runtime Pydantic Schema Validation

## Testing Custom Edge Subclasses

Framework provides a test template for you to write tests against your own Edge subclasses.

### Quick Start

```bash
# Copy template
cp tests/test_edge_template.py tests/test_my_edge.py

# Edit test_my_edge.py, fill in your Edge class and test data

# Run
pytest tests/test_my_edge.py -v
```

### Template Structure

| Test Class | Purpose |
|------------|---------|
| `TestCondition` | Test `condition()` guard logic |
| `TestHooks` | Test `pre_process` / `post_process` transforms |
| `TestExecution` | End-to-end execution tests |
| `TestSettingsCombinations` | Different settings combinations |
| `TestResetAndRepr` | Reset and repr behavior |

### Core Helper Functions

```python
from tests.test_edge_template import make_edge, make_source_vertex, make_dest_vertex, echo_agent

# Create Edge instance
edge = make_edge(MyCustomEdge, channel="score", settings={"threshold": 80})

# Create source Vertex with data
src = make_source_vertex(90, channel="score")

# Create destination Vertex
dst = make_dest_vertex(incoming_edges=["e1"])

# Execute and verify
result = await edge.execute(src, dst, echo_agent())
assert edge.completed is True
```

### Writing Your Tests

1. Replace `TODO` comments with your Edge class
2. Define `INPUT_SCENARIOS` for test data
3. Define `SETTINGS_SCENARIOS` for configurations
4. Implement assertions in test methods
