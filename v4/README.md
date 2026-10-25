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

The core engine (`GraphV4`, `ExecutorV4`, edges, store, runner) imports without the
web stack: `fastapi`/`uvicorn` are loaded only when you actually use the server or
`SSEExecutorV4` (`create_v4_server`, `SessionGraphManagerV4`, `SSEExecutorV4` and
`ToolCallEcho` are resolved lazily). So the library can be embedded in a process
that already has its own web framework.

---

## Quick Start (快速上手)

Four things you'll probably want to do, in the order they come up. Every command
below is copy-pasteable from a fresh clone.

### 0. Install

```bash
pip install -e ".[examples]"      # editable install + the bundled example graphs' deps
vea-run-v4 --help                 # verify both console scripts are on PATH
```

`pip install -e .` alone is enough if you don't need the example graphs (the core
engine has no FastAPI dependency).

### 1. Run a whole graph — `vea-run-v4`

A **manifest** is a JSON file describing a graph: vertices with their initial
`content`, edges with their `type`, a `session`, and the `edge_ids` to start with.

```bash
vea-run-v4 examples/custom_edge/config.json --session demo
```

```
session:     demo
success:     True
elapsed:     0.010s
completed:   e_count, e_upper, e_passthrough
vertex data:
  v_in: the quick brown fox jumps
  v_count: 5 words
  v_upper: 5 WORDS
  v_out: 5 WORDS
```

| Flag | Meaning |
| :--- | :--- |
| `--session ID` | Session id. Falls back to the manifest's `"session"` key, then the filename stem. |
| `--db graph.db` | Persist vertices and edge metrics to SQLite instead of in-memory. |
| `--script-root DIR` | Restrict where `"my_edges.py:MyEdge"` scripts may be loaded from (repeatable). |
| `--json` | Print the result as JSON: edge ids, outputs, failure reasons. |
| `--print-events` / `--quiet` | Turn the live event log on or off. |
| `--concurrency N`, `--timeout S` | Execution bounds. |

`python3 examples/run.py <config.json>` is the **V1** runner and is frozen as-is;
it refuses V4 manifests and `vea-run-v4` refuses V1 ones, each pointing at the
other.

### 2. Write your own edge — zero registration

1. Subclass `EdgeV4` in any script:

   ```python
   # my_edges.py
   from framework import EdgeV4
   from framework.edges.base import EdgeResultV4
   from framework.vertex_v4 import VertexStateV4


   class UpperCaseEdge(EdgeV4):
       async def run(self, session_id, store, agent=None, auto_transition=True, **kwargs):
           ok, reason, in_v, out_v = self.check_handshake(session_id, store)
           if not ok:
               return EdgeResultV4(edge_id=self.id, success=False, skipped=True, reason=reason)
           out = (in_v.content or "").upper()
           store.apply_merge_strategy(
               session_id=session_id, name=self.output_vertex, incoming_content=out
           )
           if auto_transition:
               store.update_vertex_state(
                   session_id, self.output_vertex, VertexStateV4.DATA_READY.value
               )
           return EdgeResultV4(edge_id=self.id, success=True, output=out)
   ```

2. Reference it as `"type": "my_edges.py:UpperCaseEdge"` — in a manifest, or via
   the single-edge CLI:

   ```bash
   python3 -m framework.edges.cli \
       --dir ./my_edges \
       --type my_edges.py:UpperCaseEdge \
       --session s1 \
       --input v_in --output v_out \
       --id e1 \
       --seed "hello world"
   # {"success": true, "output": "HELLO WORLD", "edge_id": "e1", "skipped": false, ...}
   ```

   `--seed` writes initial content into the input vertex (the CLI only has vertex
   *names*, so something must put content in the store first).

3. Done. There is no type whitelist and no registry to edit: any `script.py:ClassName`
   that imports and defines an `EdgeV4` subclass works. Built-in edge types:
   `code`, `llm`, `tool`, `sensenova` (plus aliases `code_edge`, `llm_chat`,
   `llm_generate`, `llm_process`, `llm_tool_call`, `reflexive`, `sensenova_edge`,
   `tool_call`). A server instance reports its live list at `GET /api/edge-types`.

### 3. Call it from Python

```python
from framework.run_v4 import run_from_manifest

result = run_from_manifest(
    "examples/custom_edge/config.json",
    session_id="demo",        # None = the manifest's "session", then the filename stem
)
print(result.success, result.completed_edges, result.errors)
```

`run_from_manifest` returns an `ExecutionResultV4`: `success`, `completed_edges`,
`edge_results`, `vertex_states`, `vertex_contents`, `errors`, `execution_time`,
`session_id`. Inside a running event loop (e.g. an ASGI request handler) use
`await run_manifest_async(...)` instead — the sync wrapper refuses nested loops.

Lower level, `load_v4_manifest` just loads: it returns a `GraphV4` with `.edges`
(a dict of edge id → `EdgeV4`) and `.vertices` (name → `VertexRecordV4`).

```python
from framework import ExecutorV4, VertexStoreV4
from framework.run_v4 import load_v4_manifest

graph = load_v4_manifest("examples/custom_edge/config.json")
print(graph.edges, graph.vertices)

store = VertexStoreV4(":memory:")
# seed the store from graph.vertices, then ExecutorV4(graph, store).run()
```

### 4. Serve it over HTTP and open the dashboard

```bash
vea-server --port 11434          # default: host 127.0.0.1, port 11434
# → http://127.0.0.1:11434/dashboard
```

The server also exposes an OpenAI-compatible API at `/v1` and the graph/DB API at
`/api/*`. Binding any non-loopback host requires `VEA_API_KEY` (or `--api-key`);
`--script-root DIR` limits which edge scripts a client may reference. See
§2.3 for the full security defaults.

### Naming cheat sheet: CLI flags ↔ config JSON keys

The single-edge CLI (`python3 -m framework.edges.cli`) accepts the same values
either as flags or as keys in a `--config` JSON file; a flag always wins.

| Flag | Config key | Sets |
| :--- | :--- | :--- |
| `--id` | `"id"` | Edge id (`--edge-id` is a deprecated alias) |
| `--type` | `"type"` | `"code"` / `"my_edges.py:MyEdge"` (`"edge_type"` alias accepted) |
| `--input` / `--output` | `"input_vertex"` / `"output_vertex"` | Vertex names |
| `--session` | `"session"` | Session id (`"session_id"` is a deprecated alias) |
| `--seed` | `"seed"` | Initial content for the input vertex (`--seed-input` / `"seed_input"` deprecated) |
| `--script` | `"script"` | Inline source or `.py` path for `code` / `recovery` edges |
| `--model` | `"model"` | Model for `llm` edges |
| `--settings` | `"settings"` | JSON object, merged over the config's own `settings` |
| `--db` | `"db"` | SQLite path (`:memory:` by default) |
| `--mock` | `"mock"` | Use `MockAgentV4` — no API key, offline |
| `--dir` | `"dir"` | Where `script.py:ClassName` is resolved from |
| `--config` | — | JSON file holding all of the above |

The last three (`--db`, `--mock`, `--dir`) and `--session` / `--seed` are
**runner** parameters: consumed by the CLI itself to create the store, seed
content and resolve paths. They are not read by `EdgeV4.from_config`, which is
why they only appear in a single-edge `--config`, never in a graph manifest.

### 5. Config reference

#### V4 manifest — the full shape

```json
{
  "version": "4.0",
  "metadata": { "name": "my_graph", "description": "what this graph does" },
  "session": "demo",
  "vertices": [
    { "name": "v_in",  "content": "the quick brown fox jumps",
      "state": "data ready", "attributes": ["start"] },
    { "name": "v_out", "content": "", "state": "todo", "attributes": ["end"] }
  ],
  "edges": [
    { "id": "e1", "type": "code", "input_vertex": "v_in", "output_vertex": "v_out",
      "script": "my_code.py", "settings": { "merge_strategy": "json_merge" } }
  ]
}
```

| Key | Required | Meaning |
| :--- | :--- | :--- |
| `version` | no | `"4.0"`. `vea-run-v4` decides a file is V1 when `version` does **not** start with `4` *and* every vertex lacks both `name` and `state`. Only unambiguous V1 manifests are refused, so a version-less V4 file still loads. |
| `metadata` | no | Free-form dict; `metadata.name` becomes the graph name (`"subgraph"` if absent). |
| `session` | no | Session id. Resolution order: `--session` > this key > filename stem (`"default_session"` when loading from a dict). `"session_id"` is a deprecated alias. |
| `vertices` | yes | One entry per vertex: `name` (unique), `content`, `state`, `attributes`, `processed_count`. |
| `edges` | yes | One entry per edge — each is passed straight to `EdgeV4.from_config` (see below). |
| `subgraph` | no | A directory to load as a subgraph (child manifest + per-node `.json` files). |

#### Edge config — the keys `EdgeV4.from_config` reads

| Key | Meaning |
| :--- | :--- |
| `id` | Edge id (also `--id`) |
| `type` | `"code"` / `"my_edges.py:MyEdge"`. `"edge_type"` is a lenient alias. |
| `input_vertex`, `output_vertex` | Vertex names the edge consumes and produces |
| `script` | `.py` path (or inline source for `code` edges when allowed) |
| `model` | Model name for `llm` edges |
| `settings` | Object merged with the edge's own settings (CLI `--settings` wins) |
| `concurrency_limit`, `concurrency_group`, `priority`, `timeout` | Scheduling hints, top-level or inside `settings` |

#### Vertex states, attributes, merge strategies

| States (`state`) | `idle`, `data ready`, `todo`, `todo urgent`, `forbidden`, `reject`, `pruning` |
| :--- | :--- |
| Attributes (`attributes`) | `start`, `end`, `llm prompt`, `llm result`, `json`, `plain text`, `subgraph`, `active`, `inactive`, `orphan` |
| Merge strategies (`settings.merge_strategy`) | `overwrite` (default), `json_merge` (shallow object merge), `list_append`, `reducer_script` (pairs with `reducer_script`) |

#### Built-in edge types

| Type | Purpose | `settings` keys |
| :--- | :--- | :--- |
| `code` (alias `code_edge`) | Run a Python script/function over the input content | `allow_inline_script`, `merge_strategy`, `reducer_script` |
| `llm` (aliases `llm_chat`, `llm_generate`, `llm_process`, `llm_tool`, `llm_tool_call`, `llm_callable`, `llm_edge`) | Call a model through the agent protocol; the alias fixes the invocation mode | `model`, `prompt`, `temperature`, `agent_mode`, `merge_strategy`, `reducer_script` |
| `tool` (alias `tool_call`) | Invoke a tool and merge its result | `args`, `arguments`, `arguments_template`, `agent_mode`, `merge_strategy` |
| `sensenova` (alias `sensenova_edge`) | Remote SenseNova inference with two-sided handshake | `merge_strategy` |
| `reflexive`, `recovery` | Self-correction / retry edges | `merge_strategy` |

`script` is a **top-level** edge key, not a settings key — with
`"settings": {"script": "..."}` the code edge silently ignores it and passes the
input through. Relative paths resolve against the manifest's directory, and a
script exposes any one of `execute`, `process`, `run` or `transform` as its entry
point.

`GET /api/edge-types` returns the registered types (`types`), plus
`script_spec_supported: true` and whether script roots are configured
(`script_roots_configured`). A script spec is **free text** in the graph editor,
not part of that list.

#### Single-edge `--config` JSON

The single-edge CLI reads the same edge keys plus the runner parameters, so a
config file is just a JSON version of the flags:

```json
{
  "session": "s1",
  "id": "e1",
  "type": "my_edges.py:UpperCaseEdge",
  "input_vertex": "v_in",
  "output_vertex": "v_out",
  "seed": "hello world",
  "dir": "./my_edges",
  "db": "graph.db",
  "mock": false,
  "settings": { "merge_strategy": "overwrite" }
}
```

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

You can invoke workflows directly within your Python applications using 4 execution
modes. The snippets below assume you already have a `graph` and a seeded `store` —
build them with `load_v4_manifest` + `store.save_vertex` (Quick Start §3), or let
`run_from_manifest` do both in one call.

#### 1. Complete Workflow Execution (`ExecutorV4.run()`)
Runs the graph concurrently to completion according to DAG topological order:
```python
from framework import ExecutorV4

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
# Full suite — 723 tests; the 6 "live" ones need a real model API key, so they're deselected
pytest tests/ -q -m "not live"

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
# Full suite: 723 passed, 0 failed (the 6 "live" tests need a real model API key)
pytest tests/ -q -m "not live"

# Include the live tests (requires network + a real model API key)
pytest tests/ -m live

# Run specific V4 tests
pytest tests/test_v4_edge_metrics.py tests/test_openai_v4_endpoints.py tests/test_harness_http_executor.py -v
```

---

## 8. Documentation & Architecture Reference

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — full architecture guide: the
  vertex-as-state / edge-as-computation model, the four layers, DAG scheduling,
  fan-in settlement, the state/colour model, and the security defaults.
- **[docs/REDESIGN_RFC.md](docs/REDESIGN_RFC.md)** — what changed in this revision
  and why, migration notes, and the deliberately-unchanged list (V1 is frozen, not
  retired).
- **[docs/CODE_REVIEW_2026-09-08.md](docs/CODE_REVIEW_2026-09-08.md)** — the review
  this revision answers. Its findings are kept as the original record; the banner at
  the top carries the current test count and what is still open.
- **[agent.md](agent.md)** — developer guide: module map, conventions, where things live.
- **[docs/DOC_STANDARDS.md](docs/DOC_STANDARDS.md)** — pinned formatting and
  authoring rules. The mechanical subset is enforced by
  [`scripts/check_docs.py`](scripts/check_docs.py) in CI; the rest (commands
  actually run, API claims checked against code) is a review obligation.
- **[docs/README.md](docs/README.md)** — documentation index.
- **[docs/archive/](docs/archive/)** — historical specs, review logs, milestone plans
  and one-off reports (the pre-refactor review report, the pre-hardening V4.0 handoff,
  the legacy V1 usage guide, ...). Records, not references: when they conflict with
  the current docs, the current docs win.
