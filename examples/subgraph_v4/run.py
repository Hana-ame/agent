#!/usr/bin/env python3
"""V4 Subgraph & Discrete JSON Path Configuration Example.

Demonstrates:
1. Defining vertices and edges via individual JSON configuration files.
2. Programmatically adding vertices and edges using JSON file paths.
3. Loading all vertices and edges directly from a directory.
4. Hierarchical execution: parent graph -> child subgraph -> child edge transforms -> parent output.

Usage:
    python3 examples/subgraph_v4/run.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from framework.graph_v4 import DiscreteGraphLoaderV4, GraphV4
from framework.server_v4 import SessionGraphManagerV4
from framework.sse_executor_v4 import SSEExecutorV4
from framework.vertex_v4 import VertexStateV4, VertexStoreV4

DEMO_DIR = Path(__file__).resolve().parent


async def run_method_1_discrete_paths() -> None:
    """Method 1: Loading vertices and edges by specifying individual JSON file paths."""
    print("\n" + "=" * 60)
    print("[Method 1] Load Vertices and Edges via Discrete JSON Paths")
    print("=" * 60)

    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    session_id = "subgraph_discrete_paths_session"

    graph = manager.get_or_create_graph(session_id)

    # 1. Add vertices directly by specifying JSON file paths
    graph.add_vertex(DEMO_DIR / "parent_in.json")
    graph.add_vertex(DEMO_DIR / "parent_subgraph.json")
    graph.add_vertex(DEMO_DIR / "parent_out.json")

    # 2. Add edges directly by specifying JSON file paths
    graph.add_edge(DEMO_DIR / "e_start_to_subgraph.json")
    graph.add_edge(DEMO_DIR / "e_subgraph_to_output.json")

    print(f"  Registered Vertices: {list(graph.vertices.keys())}")
    print(f"  Registered Edges:    {list(graph.edges.keys())}")

    # Seed the graph into SQLite Store
    DiscreteGraphLoaderV4.populate_store(graph, store)

    # Execute workflow
    executor = SSEExecutorV4(manager=manager, store=store)
    result = await executor.execute_harness_call(session_id=session_id)
    info = json.loads(result["function"]["arguments"])["info"]

    print("\n  Execution Completed:")
    print(f"    - Success:         {info['success']}")
    print(f"    - Completed Edges: {info['completed_edges']}")
    print("    - Final Node Output (final_output):")
    final_output = info["vertex_contents"].get("final_output", "")
    print(f"      {final_output}")


async def run_method_2_directory_loading() -> None:
    """Method 2: Loading an entire directory containing vertices and edges."""
    print("\n" + "=" * 60)
    print("[Method 2] Load All Vertices and Edges from Directory")
    print("=" * 60)

    # Load all components directly from directory
    dir_path = DEMO_DIR / "discrete_dir_demo"
    dir_graph = GraphV4.from_directory(dir_path, session_id="dir_loading_session")

    print(f"  Loaded from directory [{dir_path.name}]:")
    print(f"    - Vertices: {list(dir_graph.vertices.keys())}")
    print(f"    - Edges:    {list(dir_graph.edges.keys())}")


async def run_method_3_manifest_loading() -> None:
    """Method 3: Loading parent graph with subgraph via master manifest."""
    print("\n" + "=" * 60)
    print("[Method 3] Execute Full Subgraph Pipeline via Master Manifest")
    print("=" * 60)

    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    manifest_path = DEMO_DIR / "parent_graph.json"

    executor = SSEExecutorV4(manager=manager, store=store, default_manifest=manifest_path)
    result = await executor.execute_harness_call()
    info = json.loads(result["function"]["arguments"])["info"]

    print("\n  Execution Completed:")
    print(f"    - Session ID:       {info['session_id']}")
    print(f"    - Success:          {info['success']}")
    print(f"    - Completed Edges:  {info['completed_edges']}")
    print("    - Vertex Final Contents:")
    for name, content in info["vertex_contents"].items():
        print(f"      * {name}: {content}")


async def main() -> None:
    await run_method_1_discrete_paths()
    await run_method_2_directory_loading()
    await run_method_3_manifest_loading()
    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
