#!/usr/bin/env python3
"""Dynamic Tool Library & Intent Router Demo (Plan A).

Demonstrates:
1. Tool Library: A catalog of modular tool subgraphs (code_analyzer, finance_calculator, data_extractor).
2. Intent Router Vertex: Analyzes incoming user task.
3. Dynamic Subgraph Splicing: On-the-fly loads the selected tool subgraph from disk and wires it into the pipeline.
4. Execution & Audit Snapshots: Full graph execution with automatic step-by-step history snapshot capture.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.graph_v4 import GraphV4, DiscreteGraphLoaderV4
from framework.vertex_v4 import VertexStoreV4, VertexStateV4
from framework.executor_v4 import ExecutorV4
from framework.snapshot_v4 import GraphSnapshotManagerV4
from examples.dynamic_tool_library.tool_scripts import classify_intent


TOOL_CATALOG_DIR = Path(__file__).parent / "tools"


async def run_pipeline_with_dynamic_tool(task_query: str, session_id: str) -> Dict[str, Any]:
    """Dynamically route task_query to appropriate tool subgraph, splice it, and run."""
    print("\n" + "=" * 76)
    print(f"📥 [User Task Request] ({session_id})")
    print(f"   \"{task_query}\"")
    print("=" * 76)

    # 1. Initialize SQLite store and base Graph
    store = VertexStoreV4(":memory:")
    graph = GraphV4(session_id=session_id, name="dynamic_router_graph")

    # 2. Add base parent vertices
    v_start = graph.add_vertex(
        "v_user_query",
        content=task_query,
        state=VertexStateV4.DATA_READY.value,
        attributes=["start"]
    )
    v_final = graph.add_vertex(
        "v_final_output",
        content="",
        state=VertexStateV4.TODO.value,
        attributes=["end"]
    )

    # 3. Router logic: Determine which tool subgraph to activate
    tool_name = classify_intent(task_query)
    tool_manifest_path = TOOL_CATALOG_DIR / f"{tool_name}.json"
    
    if not tool_manifest_path.exists():
        raise FileNotFoundError(f"Requested tool '{tool_name}' not found in catalog: {tool_manifest_path}")

    print(f"🧭 [Intent Router] Classified Task ➔ Category: '{tool_name}'")
    print(f"📦 [Tool Library] Splicing Tool Subgraph from '{tool_manifest_path.name}' on-the-fly...")

    # 4. Dynamically load and splice the tool subgraph into the main graph
    tool_subgraph = DiscreteGraphLoaderV4.load_from_manifest(tool_manifest_path)
    
    # Connect parent input -> tool sub_in, and tool sub_out -> parent output
    splice_info = graph.insert_subgraph(
        subgraph=tool_subgraph,
        incoming_bindings={"v_user_query": "sub_in"},
        outgoing_bindings={"sub_out": "v_final_output"},
        name_prefix=tool_name,
    )

    print(f"🔗 [Topology Spliced] Added Vertices: {splice_info['inserted_vertices']}")
    print(f"                      Added Edges:    {splice_info['inserted_edges']}")

    # 5. Populate SQLite store with newly expanded graph
    DiscreteGraphLoaderV4.populate_store(graph, store)

    # 6. Execute with complete Graph Snapshots enabled
    snap_dir = Path(__file__).parent / "snapshots"
    executor = ExecutorV4(
        graph=graph,
        store=store,
        snapshot_dir=snap_dir,
        max_concurrency=4,
    )

    print(f"🚀 [Executor] Starting DAG execution with automated history snapshots...")
    result = await executor.run()

    if not result.success:
        print(f"❌ Execution failed: {result.errors}")
        return {}

    final_content = result.vertex_contents.get("v_final_output", "")
    print(f"✅ [Execution Finished] Success in {result.execution_time * 1000:.2f} ms")
    print("\n" + "-" * 76)
    print(final_content)
    print("-" * 76)

    # 7. Inspect the complete graph history snapshots
    mgr = GraphSnapshotManagerV4(base_dir=snap_dir)
    snapshots = mgr.list_snapshots(session_id)
    print(f"\n📸 [Graph Snapshots Captured] Total: {len(snapshots)} steps")
    for s in snapshots:
        print(f"   • Step {s['step']}: trigger='{s['trigger']}' | nodes={s['vertex_count']}, edges={s['edge_count']} -> {s['filename']}")

    return {
        "tool_used": tool_name,
        "output": final_content,
        "snapshots_count": len(snapshots),
    }


async def main():
    test_tasks = [
        ("def calculate_tax(gross): eval('2 + 2')\n    return gross * 0.2", "sess_code_audit"),
        ("Company reported revenue of 5000000 and total operating costs of 3200000 this quarter.", "sess_finance_eval"),
        ("Please contact founder alice@startup.io or lead engineer bob@deeptech.ai for product demos.", "sess_data_extract"),
    ]

    for query, session_id in test_tasks:
        await run_pipeline_with_dynamic_tool(query, session_id)

    print("\n" + "=" * 76)
    print("🎉 All dynamic tool library routing tasks completed successfully!")
    print("=" * 76)


if __name__ == "__main__":
    asyncio.run(main())
