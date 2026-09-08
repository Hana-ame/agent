# Vertex-Edge Agent Framework Guide

Vertex-Edge Agent Framework is a data-driven agent orchestration system where **vertices** store states and data while **edges** encapsulate logic, transforms, and model inference.

> 📖 **架构与技术全景指南**：详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 以及开发者指南 [agent.md](agent.md)。

---

## 1. Installation & Setup

- **Python Requirement**: `Python 3.12+`

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install editable package
pip install -e .

# Optional: extra deps used only by the bundled example graphs
pip install -e ".[examples]"
```

Once installed, two console scripts are available:

- `vea-server` — the V4 API server (same entry point as `python3 -m framework.server_v4`)
- `vea-run-v4` — run a V4 manifest end to end (same entry point as
  `python3 -m framework.run_v4`):

  ```bash
  vea-run-v4 examples/custom_edge/config.json --session demo
  ```

  `python3 examples/run.py <config.json>` is the **V1** runner and is unchanged;
  it refuses V4 manifests and `vea-run-v4` refuses V1 ones, each pointing at the
  other.

---

## 2. All Execution Modes & Running Guide (运行方式全景指南)

Vertex-Edge Agent Framework supports multiple execution paradigms: standalone CLI scripts, headless programmatic Python calls, request-driven agent harness clocks, and OpenAI-compatible HTTP services.

### 2.1 Command-Line Example Pipelines (命令行一键运行示例)

| Task / Scenario | Command | Key Architecture Features Demonstrated |
| :--- | :--- | :--- |
| **Hacker News V4 Full Pipeline** | `python3 examples/hn_v4/run.py` | Multi-branch Fan-In (`JSON_MERGE`), ReflexiveEdge self-healing, real-time streaming, SQLite edge metrics, declarative ToolEdge |
| **Hacker News AI Report V4** | `python3 examples/hn_ai_report_v4/demo.py --limit 5` | V4 migration of MapEdge fan-out, async batch story summarization, Markdown report |
| **Nested Subgraphs & Discrete Paths** | `python3 examples/subgraph_v4/run.py` | Hierarchical execution (`parent -> child -> parent`), discrete `.json` vertex/edge loading |
| **SenseNova Remote LLM Pipeline** | `python3 examples/sensenova_v4/demo.py` | Remote LLM inference with two-sided handshake and state transitions |
| **OpenCode & Proxied Agent** | `python3 examples/opencode_zen/proxy_demo.py` | Dynamic proxy agent, OpenCode integration, token tracking |
| **Generic JSON Config Runner** | `python3 examples/run.py <path/to/config.json>` | Generic engine runner for legacy and discrete pipeline definitions |
| **V4 Manifest Runner** | `vea-run-v4 <path/to/graph.json> [--session ID] [--db DB] [--script-root DIR] [--json]` | Load + seed SQLite + execute in one call; V4 manifests only (`examples/run.py` stays V1) |
| **Runtime Topology Mutation** | `python3 examples/dynamic_topology/demo.py` | Dynamic vertex insertion and runtime graph modification |
| **Custom Edge Extension** | `python3 examples/custom_edge/demo.py` | Zero-registration custom edge classes referenced as `"type": "my_edges.py:MyEdge"` |
| **Self-Correction Loop** | `python3 examples/self_correction/demo.py` | Reflexive error recovery and multi-pass correction |
| **Real-time Streaming Pipeline** | `python3 examples/realtime_streaming/demo.py` | Token/chunk streaming through execution nodes |
| **Human-in-the-Loop Approval** | `python3 examples/hitl_approval/demo.py` | Interactive pause-and-resume execution flow |
| **Speculative Race Mode** | `python3 examples/race_mode/demo.py` | Parallel competitive path execution (first-to-finish wins) |

---

### 2.2 Headless Python SDK Execution Modes (代码内调用模式)

You can invoke workflows directly within your Python applications using 4 execution modes:

#### 1. Complete Workflow Execution (`ExecutorV4.run()`)
Runs the graph concurrently to completion according to DAG topological order:
```python
from framework import GraphV4, VertexStoreV4, ExecutorV4

store = VertexStoreV4(":memory:")
executor = ExecutorV4(graph=graph, store=store, max_concurrency=4)
result = await executor.run()
print(f"Success: {result.success}, Completed Edges: {result.completed_edges}")
```

#### 2. Real-Time Event Streaming (`ExecutorV4.stream()`)
Streams lifecycle events (`edge_started`, `edge_completed`, `edge_failed`) asynchronously for live telemetry or UI progress bars:
```python
async for event in executor.stream():
    if event.event_type == "edge_started":
        print(f"▶ Started Edge: {event.edge_id} ({event.payload['input']} ➔ {event.payload['output']})")
    elif event.event_type == "edge_completed":
        print(f"✔ Completed Edge: {event.edge_id}")
```

#### 3. Single-Edge Hop-by-Hop Stepping (`ExecutorV4.step()`)
Advances the graph by exactly one eligible edge, enabling step-debugging without DAG tier batching:
```python
step_res = await executor.step()
print(f"Executed edge: {step_res.completed_edges}, Remaining: {not step_res.success}")
```

#### 4. Agent Harness Request-Clock Stepping (`HttpHarnessExecutorV4.step()`)
Drives execution via passive external request pulses, emitting OpenAI standard `tool_calls` when a `ToolEdgeV4` is encountered:
```python
from framework import HttpHarnessExecutorV4

http_exec = HttpHarnessExecutorV4(store=store)
messages = [{"role": "user", "content": "Start workflow"}]

while True:
    resp = await http_exec.step(session_id, graph, messages, model="default")
    finish_reason = resp["choices"][0]["finish_reason"]
    msg = resp["choices"][0]["message"]
    messages.append(msg)

    if finish_reason == "stop":
        print("Completed! Final answer:", msg["content"])
        break
    elif finish_reason == "tool_calls":
        # External harness executes command in its sandbox:
        tool_call = msg["tool_calls"][0]
        sandbox_output = "command execution result"
        messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": sandbox_output})
```

---

### 2.3 Starting the API Server & Dashboard (启动 HTTP 服务端)

> 🔒 **Security defaults (changed in this revision)**
> - The server binds **`127.0.0.1`** by default. Binding any other host **requires** an API key.
> - Set `VEA_API_KEY` (or pass `--api-key`) to require `X-API-Key` / `Authorization: Bearer`
>   on every `/api/*` and `/v1/*` route. `/dashboard` stays reachable so the console can load;
>   the browser prompts for the key once and stores it in `localStorage.vea_api_key`.
> - CORS is **disabled** unless you pass `--cors-origin`.
> - Client-supplied manifest paths are confined to the repository root (override with
>   `--manifest-base-dir`); the `vea_manifest_path` request field was removed.
> - Inline `lambda` scripts are disabled by default; trusted local configs must opt in with
>   `settings.allow_inline_script = true`. The HTTP API never allows them.
> - Client-supplied edge script specs (`"my_edge.py:MyEdge"`) may only load from the repository
>   root plus `VEA_SCRIPT_ROOTS` (`os.pathsep`-separated). Pass `--script-root DIR` (repeatable)
>   to replace that set — recommended for a deployed server, so only your own edge directory
>   is loadable.

#### Standard Service Launch (loopback only, default port 11434)
```bash
# Launch unified V4 server (supports OpenAI API + Dashboard + Graph Mutation)
python3 -m framework.server_v4 --port 11434
```

#### Launch with an API key (required for any non-loopback bind)
```bash
export VEA_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
python3 -m framework.server_v4 --host 0.0.0.0 --port 11434 --api-key "$VEA_API_KEY"
```

#### Launch with SQLite Database Persistence
```bash
# Persist all sessions, vertices, staging, and edge metrics to SQLite file
python3 -m framework.server_v4 --port 11434 --db agent_data.db
```

#### Launch with a custom edge directory (zero-registration edges)
```bash
# Only scripts under ./my_edges may be referenced by "type": "my_edge.py:MyEdge"
python3 -m framework.server_v4 --port 11434 --script-root ./my_edges
```

#### Production Launch via Uvicorn
```bash
# VEA_API_KEY must be exported — uvicorn bypasses the CLI's bind guard.
VEA_API_KEY=... uvicorn framework.server_v4:app --host 127.0.0.1 --port 11434 --workers 1
```

#### Legacy V1 Server
```bash
uvicorn framework.serve.app:app --host 127.0.0.1 --port 8000
```

---

### 2.4 Client API Invocations & Agent Harness (客户端调用与接入)

Once the server is running on `http://localhost:11434`, you can interact with it using any standard client:

#### 1. Official OpenAI Python SDK (Non-Streaming)
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Run AI digest pipeline"}],
)
print(response.choices[0].message.content)
```

#### 2. Official OpenAI Python SDK (Streaming)
```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Analyze repository"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### 3. Standard `curl` CLI Requests
```bash
# List available models
curl http://localhost:11434/v1/models

# Standard chat completion
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Execute task"}]}'

# Server-Sent Events (SSE) graph execution
curl -N -X POST http://localhost:11434/api/sse/execute \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_sess", "input": "Hello V4"}'
```

#### 4. Web Console & Visual Dashboard
Open in browser:
```text
http://localhost:11434/dashboard
```
Visualizes active DAG topologies, vertex states (`DATA_READY`, `TODO`, `REJECT`), SQLite database tables, and live edge metrics.

#### 5. Edge Performance Benchmarking REST APIs
```bash
# Query edge metrics summary (total executions, error rate, latency breakdown)
curl http://localhost:11434/api/metrics/summary

# Query edge metrics for a specific session
curl http://localhost:11434/api/sessions/{session_id}/metrics

# Database table statistics
curl http://localhost:11434/api/db/stats
```

#### 6. Online Graph Mutation REST APIs
```bash
# Add or update vertex
curl -X POST http://localhost:11434/api/sessions/{session_id}/vertices \
  -H "Content-Type: application/json" \
  -d '{"name": "new_node", "content": "data", "state": "data ready"}'

# Dynamically connect edge
curl -X POST http://localhost:11434/api/sessions/{session_id}/edges \
  -H "Content-Type: application/json" \
  -d '{"id": "e_new", "input_vertex": "src", "output_vertex": "dst", "type": "code"}'

# Replay/reenter vertex with automatic downstream invalidation
curl -X POST http://localhost:11434/api/sessions/{session_id}/vertices/{vertex_name}/reenter \
  -H "Content-Type: application/json" \
  -d '{"content": "revised input"}'
```

---

### 2.5 Automated Testing (运行自动化测试)

```bash
# Run complete test suite (551 tests, 100% pass rate)
pytest -q

# Run edge metrics telemetry benchmarks
pytest tests/test_v4_edge_metrics.py -v

# Run OpenAI API compatibility tests
pytest tests/test_openai_v4_endpoints.py -v

# Run Agent Harness multi-turn interaction tests
pytest tests/test_harness_http_executor.py -v
```

---

## 3. Quick Start & Graph Authoring Guide

### Approach A: Orchestration via JSON Manifest (Recommended)

#### Step 1: Write execution script (`workflow/trans.py`)

```python
def to_uppercase(content: str, settings: dict, staging: dict) -> str:
    """Convert input content to uppercase."""
    return content.upper()

def add_signature(content: str, settings: dict, staging: dict) -> str:
    """Append author signature to content."""
    author = settings.get("author", "VEA")
    return f"{content}\n-- Processed by {author}"
```

#### Step 2: Define graph manifest (`workflow/graph.json`)

```json
{
  "version": "4.0",
  "session_id": "demo_pipeline",
  "metadata": {"name": "Text Processing Pipeline"},
  "vertices": [
    {"name": "input_node", "content": "hello world", "attributes": ["start"], "state": "data ready"},
    {"name": "upper_node", "content": "", "state": "todo"},
    {"name": "output_node", "content": "", "attributes": ["end"], "state": "todo"}
  ],
  "edges": [
    {
      "id": "e_upper",
      "input_vertex": "input_node",
      "output_vertex": "upper_node",
      "script": "workflow/trans.py:to_uppercase"
    },
    {
      "id": "e_sign",
      "input_vertex": "upper_node",
      "output_vertex": "output_node",
      "script": "workflow/trans.py:add_signature",
      "settings": {"author": "VertexEdgeAgent"}
    }
  ]
}
```

#### Step 3: Load and execute

```python
import asyncio
from framework import VertexStoreV4, DiscreteGraphLoaderV4, ExecutorV4

async def main():
    store = VertexStoreV4(":memory:")
    graph = DiscreteGraphLoaderV4.load_from_manifest("workflow/graph.json")
    DiscreteGraphLoaderV4.populate_store(graph, store)

    executor = ExecutorV4(graph=graph, store=store, max_concurrency=2)
    result = await executor.run()

    print("Success:", result.success)
    final_node = store.get_vertex(graph.session_id, "output_node")
    print("Output:\n", final_node.content)

if __name__ == "__main__":
    asyncio.run(main())
```

The same manifest can be run without any boilerplate — the runner does the load,
the store seeding and the execution in a single call:

```python
from framework import run_from_manifest

result = run_from_manifest("workflow/graph.json", session_id="demo")
print(result.success, result.vertex_contents["output_node"])
```

Outside the repository, pass `--script-root DIR` (repeatable) to `vea-run-v4` so a
manifest can reference your own edge classes by script path.

---

### Approach B: Programmatic Orchestration via Python API

```python
import asyncio
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, CodeEdgeV4
)

async def main():
    store = VertexStoreV4(":memory:")
    session_id = "code_sess"
    graph = GraphV4(session_id=session_id)

    # 1. Add vertices
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="src",
        content="Antigravity",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="dst",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    # 2. Add transformation edge
    def reverse_text(data, settings, staging):
        return data[::-1]

    graph.add_edge(CodeEdgeV4("edge_rev", "src", "dst", script=reverse_text))
    graph.validate()

    # 3. Run executor
    executor = ExecutorV4(graph=graph, store=store)
    result = await executor.run()

    print("Result:", store.get_vertex(session_id, "dst").content)  # Output: ytivargitnA

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Approach C: Load via Discrete JSON File Paths or Directories

`graph.add_vertex()` and `graph.add_edge()` accept direct `.json` paths or directory paths:

```python
from framework import GraphV4

graph = GraphV4(session_id="path_loading_session")

# 1. Load vertices directly via individual JSON files
graph.add_vertex("examples/subgraph_v4/parent_in.json")
graph.add_vertex("examples/subgraph_v4/parent_subgraph.json")
graph.add_vertex("examples/subgraph_v4/parent_out.json")

# 2. Load edges directly via individual JSON files
graph.add_edge("examples/subgraph_v4/e_start_to_subgraph.json")
graph.add_edge("examples/subgraph_v4/e_subgraph_to_output.json")

# 3. Batch load all vertices and edges from a directory
dir_graph = GraphV4.from_directory("examples/subgraph_v4/discrete_dir_demo", session_id="dir_sess")
```

---

## 4. Remote Model Pipeline (SenseNova)

Built-in support for remote model inference (optionally set credentials via `SENSENOVA_API_KEY`):

```python
import asyncio
import json
from framework import (
    VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4,
    GraphV4, ExecutorV4, SenseNovaEdgeV4, CodeEdgeV4
)

async def main():
    store = VertexStoreV4(":memory:")
    session_id = "llm_sess"
    graph = GraphV4(session_id=session_id)

    # 1. Prompt vertex
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="prompt_node",
        content="Calculate 15 + 25. Output strictly JSON: {\"result\": 40}",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value
    ))

    # 2. Model result vertex
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="model_node",
        content="",
        attributes=[VertexAttributeV4.JSON.value],
        state=VertexStateV4.TODO.value
    ))

    # 3. Final target vertex
    graph.add_vertex(VertexRecordV4(
        id=0, session_id=session_id, name="final_node",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value
    ))

    # 4. Connect model edge and post-processing code edge
    graph.add_edge(SenseNovaEdgeV4(
        edge_id="e_llm",
        input_vertex="prompt_node",
        output_vertex="model_node",
        settings={"temperature": 0.1}
    ))

    def parse_result(content, settings, staging):
        data = json.loads(content)
        return f"Computed result: {data['result']}"

    graph.add_edge(CodeEdgeV4("e_calc", "model_node", "final_node", script=parse_result))
    graph.validate()

    executor = ExecutorV4(graph=graph, store=store)
    res = await executor.run()

    print("Final content:", store.get_vertex(session_id, "final_node").content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Core Architectural Features

### 1. Fan-In Merge Strategies

When multiple upstream edges converge on a single vertex, use `MergeStrategyV4` to specify data merging behavior:

```python
from framework import MergeStrategyV4

# Overwrite (default)
store.apply_merge_strategy(sess, "target", incoming, strategy=MergeStrategyV4.OVERWRITE)

# Merge JSON dictionaries: {**existing, **incoming}
store.apply_merge_strategy(sess, "target", '{"b": 2}', strategy=MergeStrategyV4.JSON_MERGE)

# Append to JSON list: [item1, item2, ...]
store.apply_merge_strategy(sess, "target", '"new_item"', strategy=MergeStrategyV4.LIST_APPEND)

# Custom reducer function
def custom_reducer(existing: str, incoming: str) -> str:
    return f"{existing},{incoming}".strip(",")

store.apply_merge_strategy(sess, "target", "item", strategy=MergeStrategyV4.REDUCER_SCRIPT, reducer_fn=custom_reducer)
```

---

### 2. Error Recovery and Circuit Breaking (ReflexiveEdge)

When a vertex execution fails, it enters `REJECT`. A reflexive edge triggers self-loop recovery:

```python
from framework import ReflexiveEdgeV4, VertexStateV4

def recovery_fn(old_content, settings, staging):
    return f"{old_content} (retry with simplified query)"

reflexive_edge = ReflexiveEdgeV4(
    edge_id="e_retry",
    vertex_name="worker_node",
    trigger_state=VertexStateV4.REJECT.value,
    target_state=VertexStateV4.TODO_URGENT.value,
    max_retries=3,
    script=recovery_fn
)
```

If retry count exceeds `max_retries`, the vertex is locked into `FORBIDDEN`.

---

### 3. Nested Subgraphs

Tag a vertex with the `subgraph` attribute and supply the child graph manifest or directory path in `content`:

```json
{
  "name": "subgraph_box",
  "content": {
    "subgraph_manifest": "child/child_graph.json"
  },
  "attributes": ["subgraph"],
  "state": "todo"
}
```

During execution, `SSEExecutorV4` resolves the nested graph and attaches execution bridge edges:
- Data from the parent vertex routes to the child vertex with the `start` attribute.
- Internal child edges execute in topological DAG order.
- Data from the child vertex with the `end` attribute returns to the parent vertex.

Runnable example: [examples/subgraph_v4/](examples/subgraph_v4/):
```bash
python3 examples/subgraph_v4/run.py
```

---

### 4. Tool Call Edges for Agent Harness (`ToolEdgeV4` & `LLMToolEdgeV4`)

Define edges that yield OpenAI-compatible tool calls to an external execution sandbox (e.g. bash, git, python):

```python
from framework import ToolEdgeV4, LLMToolEdgeV4

# 1. Declarative static tool call
tool_edge = ToolEdgeV4(
    edge_id="e_run_sandbox",
    input_vertex="prompt_vertex",
    output_vertex="result_vertex",
    tool_name="bash",
    arguments_template='{"command": "cat {input}"}',
)

# 2. Dynamic LLM-driven tool call
llm_tool_edge = LLMToolEdgeV4(
    edge_id="e_llm_tool",
    input_vertex="query_node",
    output_vertex="action_node",
    tools=[{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute bash commands in the workspace",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    }],
    prompt_template="Analyze workspace for {input}"
)
```

When connected to an Agent Harness via `/v1/chat/completions`, these edges return `finish_reason: "tool_calls"`. When the harness posts back `role: "tool"` with the command output, the vertex transitions to `data ready` and continues.

---

### 5. Edge-Level Benchmarking & Metrics (`edge_metrics`)

Every edge execution automatically records execution duration (ms), prompt/completion tokens, costs, success status, and error traces in SQLite:

```python
from framework import VertexStoreV4

store = VertexStoreV4("agent_data.db")

# 1. List individual edge metrics
metrics = store.list_edge_metrics(session_id="sess_1")
for m in metrics:
    print(f"[{m.edge_id}] {m.edge_type}: {m.execution_time_ms:.1f}ms, {m.total_tokens} tokens")

# 2. Get aggregate session or global summary
summary = store.get_edge_metrics_summary(session_id="sess_1")
print(f"Total Executions: {summary['total_executions']}, Error Rate: {summary['error_rate']:.2%}")
print(f"Average Latency: {summary['avg_execution_time_ms']:.1f}ms, Total Tokens: {summary['total_tokens']}")
```

---

## 6. Server & Web Dashboard Deep Dive

### 1. Launch API Server (Default Port: 11434)

```bash
# Launch server using default port 11434 (Ollama-compatible standard port)
python3 -m framework.server_v4 --port 11434

# Or using uvicorn directly
uvicorn framework.server_v4:app --host 0.0.0.0 --port 11434
```

### 2. Native OpenAI-Compatible API (`/v1/chat/completions`)

The V4 server natively implements the OpenAI API protocol, enabling seamless integration with Agent Harnesses, LangChain, or standard OpenAI client libraries:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Analyze repository files"}],
)

# Tool calls emitted by ToolEdgeV4 / LLMToolEdgeV4 are yielded directly:
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    print("Execute in Sandbox:", tool_call.function.name, tool_call.function.arguments)
```

### 3. Web Dashboard

Navigate to `http://localhost:11434/dashboard` in your browser:
- Real-time visualization of DAG topology and vertex state highlights.
- Live inspection of vertex contents, SQLite tables, and edge execution metrics.

### 4. REST & Metrics APIs

- **Session Edge Performance Metrics**:
  ```bash
  # Query performance records and timing for a session
  curl http://localhost:11434/api/sessions/{session_id}/metrics
  ```
- **Global Benchmarking Summary**:
  ```bash
  # Aggregated latency, token counts, and error rates across all sessions
  curl http://localhost:11434/api/metrics/summary
  ```
- **Database Statistics**:
  ```bash
  curl http://localhost:11434/api/db/stats
  ```
- **Online Graph Mutation**:
  - `POST /api/sessions/{session_id}/vertices` — Dynamically add or update a vertex.
  - `POST /api/sessions/{session_id}/edges` — Dynamically connect a new edge.
  - `POST /api/sessions/{session_id}/edges/{edge_id}/reconnect` — Reconnect edge endpoints.
  - `POST /api/sessions/{session_id}/vertices/{vertex_name}/reenter` — Replay/reenter a vertex with downstream reset.
  - `POST /api/db/sessions/{session_id}/clear` — Atomically purge session vertices, staging, and metrics.

---

## 7. Testing & Quality Assurance

The framework includes a comprehensive test suite covering unit tests, SQLite persistence, concurrency semaphores, security defenses, Agent Harness HTTP execution, and performance benchmarks:

```bash
# Run full test suite (551 tests, 100% pass rate)
pytest

# Run specific V4 tests
pytest tests/test_v4_edge_metrics.py tests/test_openai_v4_endpoints.py tests/test_harness_http_executor.py -v
```

---

## 8. Documentation & Architecture Reference

- **System Handoff & Technical Specification**: See [docs/HANDOFF.md](docs/HANDOFF.md).
- **Documentation Index & Archives**: See [docs/README.md](docs/README.md).
- **Historical Reports & Architecture Specifications**: Located in [docs/archive/](docs/archive/).
