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
    print("【方法 1】通过指定单独的 JSON 配置文件路径加载 Vertex 与 Edge")
    print("=" * 60)

    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    session_id = "subgraph_discrete_paths_session"

    graph = manager.get_or_create_graph(session_id)

    # 1. 直接指定 JSON 路径添加 Vertex
    graph.add_vertex(DEMO_DIR / "parent_in.json")
    graph.add_vertex(DEMO_DIR / "parent_subgraph.json")
    graph.add_vertex(DEMO_DIR / "parent_out.json")

    # 2. 直接指定 JSON 路径添加 Edge
    graph.add_edge(DEMO_DIR / "e_start_to_subgraph.json")
    graph.add_edge(DEMO_DIR / "e_subgraph_to_output.json")

    print(f"  已注册 Vertex: {list(graph.vertices.keys())}")
    print(f"  已注册 Edge:   {list(graph.edges.keys())}")

    # 将图写入 SQLite Store
    DiscreteGraphLoaderV4.populate_store(graph, store)

    # 执行流程
    executor = SSEExecutorV4(manager=manager, store=store)
    result = await executor.execute_harness_call(session_id=session_id)
    info = json.loads(result["function"]["arguments"])["info"]

    print("\n  执行完成:")
    print(f"    - 执行成功: {info['success']}")
    print(f"    - 完成的边: {info['completed_edges']}")
    print("    - 最终节点输出 (final_output):")
    final_output = info["vertex_contents"].get("final_output", "")
    print(f"      {final_output}")


async def run_method_2_directory_loading() -> None:
    """Method 2: Loading an entire directory containing vertices and edges."""
    print("\n" + "=" * 60)
    print("【方法 2】通过指定文件夹一键加载其中的所有 Vertex 与 Edge")
    print("=" * 60)

    # 从 discrete_dir_demo 目录一键加载
    dir_path = DEMO_DIR / "discrete_dir_demo"
    dir_graph = GraphV4.from_directory(dir_path, session_id="dir_loading_session")

    print(f"  从目录 [{dir_path.name}] 加载成功:")
    print(f"    - 包含 Vertex: {list(dir_graph.vertices.keys())}")
    print(f"    - 包含 Edge:   {list(dir_graph.edges.keys())}")


async def run_method_3_manifest_loading() -> None:
    """Method 3: Loading parent graph with subgraph via master manifest."""
    print("\n" + "=" * 60)
    print("【方法 3】通过 Master Manifest JSON 配置文件执行完整 Subgraph 流程")
    print("=" * 60)

    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    manifest_path = DEMO_DIR / "parent_graph.json"

    executor = SSEExecutorV4(manager=manager, store=store, default_manifest=manifest_path)
    result = await executor.execute_harness_call()
    info = json.loads(result["function"]["arguments"])["info"]

    print("\n  执行完成:")
    print(f"    - Session ID: {info['session_id']}")
    print(f"    - 执行成功:   {info['success']}")
    print(f"    - 完成的边:   {info['completed_edges']}")
    print("    - 节点最终数据:")
    for name, content in info["vertex_contents"].items():
        print(f"      * {name}: {content}")


async def main() -> None:
    await run_method_1_discrete_paths()
    await run_method_2_directory_loading()
    await run_method_3_manifest_loading()
    print("\n" + "=" * 60)
    print("全部示例执行完毕。")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
