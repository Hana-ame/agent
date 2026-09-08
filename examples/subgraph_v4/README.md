# Subgraph & Discrete Configuration Loading Guide

This example demonstrates how to define and load vertices and edges via individual JSON configuration files or directories in the Vertex-Edge Agent Framework v4, as well as how to construct and execute nested subgraphs.

---

## 1. Quick Run

Run the following command from the repository root:

```bash
python3 examples/subgraph_v4/run.py
```

---

## 2. Directory Structure

```
examples/subgraph_v4/
├── parent_in.json                # Parent graph start vertex configuration
├── parent_subgraph.json          # Subgraph container vertex (references child_graph.json)
├── parent_out.json               # Parent graph end vertex configuration
├── e_start_to_subgraph.json      # Parent edge (input -> subgraph)
├── e_subgraph_to_output.json     # Parent edge (subgraph -> output)
├── parent_graph.json             # Parent master manifest
├── run.py                        # Executable demo script
├── child/                        # Subgraph configuration and transform scripts
│   ├── child_start.json          # Child start vertex
│   ├── child_middle.json         # Child intermediate vertex
│   ├── child_end.json            # Child end vertex
│   ├── child_edge1.json          # Child edge 1 (executes cleaner.py)
│   ├── child_edge2.json          # Child edge 2 (executes enricher.py)
│   ├── child_graph.json          # Child subgraph manifest
│   ├── cleaner.py                # Text normalization script
│   └── enricher.py               # Metadata enrichment script
└── discrete_dir_demo/            # Directory batch loading demo
    ├── vertices/                 # Vertex JSON definitions
    │   ├── v1_in.json
    │   └── v2_out.json
    └── edges/                    # Edge JSON definitions
        └── e1_dir.json
```

---

## 3. Usage Patterns

### Pattern 1: Load via Individual JSON File Paths

`graph.add_vertex()` and `graph.add_edge()` accept direct `.json` file paths:

```python
from framework.graph_v4 import GraphV4

graph = GraphV4(session_id="my_session")

# Load vertices by JSON path
graph.add_vertex("examples/subgraph_v4/parent_in.json")
graph.add_vertex("examples/subgraph_v4/parent_subgraph.json")
graph.add_vertex("examples/subgraph_v4/parent_out.json")

# Load edges by JSON path
graph.add_edge("examples/subgraph_v4/e_start_to_subgraph.json")
graph.add_edge("examples/subgraph_v4/e_subgraph_to_output.json")
```

#### Vertex JSON format example (`parent_in.json`):
```json
{
  "name": "session_input",
  "content": "hello subgraph world",
  "attributes": ["start"],
  "state": "data_ready"
}
```

#### Edge JSON format example (`e_start_to_subgraph.json`):
```json
{
  "id": "e_in_to_subgraph",
  "type": "code",
  "input_vertex": "session_input",
  "output_vertex": "enrichment_box"
}
```

---

### Pattern 2: Batch Load All Vertices and Edges from a Directory

Load entire directories containing vertex and edge definitions:

```python
from framework.graph_v4 import GraphV4

# 1. Construct a new graph directly from a directory
graph = GraphV4.from_directory("examples/subgraph_v4/discrete_dir_demo", session_id="my_session")

# 2. Merge directory contents into an existing graph
existing_graph = GraphV4(session_id="existing_session")
existing_graph.load_directory("examples/subgraph_v4/discrete_dir_demo")

# 3. Pass specific component subdirectories to add_vertex / add_edge
existing_graph.add_vertex("examples/subgraph_v4/discrete_dir_demo/vertices")
existing_graph.add_edge("examples/subgraph_v4/discrete_dir_demo/edges")
```

Supported directory layouts:
- Structured subdirectories: `vertices/` (or `vertex/`, `nodes/`) and `edges/` (or `edge/`).
- Flat directory: automatically detects vertices and edges based on JSON fields.
- Manifest directory: contains `graph.json` or `manifest.json`.

---

### Pattern 3: Nested Subgraphs

Define a vertex with attribute `subgraph` and specify its child manifest or directory in `content`:

#### Subgraph vertex configuration (`parent_subgraph.json`):
```json
{
  "name": "enrichment_box",
  "attributes": ["subgraph"],
  "state": "todo",
  "content": {
    "subgraph_manifest": "child/child_graph.json"
  }
}
```

When executed via `SSEExecutorV4`, the framework resolves child subgraphs and constructs bridge edges:

```python
from framework.graph_v4 import DiscreteGraphLoaderV4
from framework.server_v4 import SessionGraphManagerV4
from framework.sse_executor_v4 import SSEExecutorV4
from framework.vertex_v4 import VertexStoreV4

store = VertexStoreV4(":memory:")
manager = SessionGraphManagerV4(store)

executor = SSEExecutorV4(
    manager=manager,
    store=store,
    default_manifest="examples/subgraph_v4/parent_graph.json",
)

result = await executor.execute_harness_call()
```
