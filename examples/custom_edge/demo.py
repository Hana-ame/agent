#!/usr/bin/env python3
"""Run the custom-edge example.

The graph JSON references user classes by script path
(``"type": "my_edges.py:WordCountEdge"``); nothing in ``framework/`` was modified
or registered.

Usage:
    python3 examples/custom_edge/demo.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.executor_v4 import ExecutorV4
from framework.graph_v4 import DiscreteGraphLoaderV4
from framework.vertex_v4 import VertexStoreV4

DEMO_DIR = Path(__file__).resolve().parent


async def main() -> None:
    store = VertexStoreV4(":memory:")
    graph = DiscreteGraphLoaderV4.load_from_manifest(
        str(DEMO_DIR / "config.json"),
        override_session_id="custom_edge_demo",
    )

    print("Loaded edges:")
    for edge_id, edge in graph.edges.items():
        print(f"  {edge_id}: {type(edge).__name__}  (type={edge.type})")

    DiscreteGraphLoaderV4.populate_store(graph, store)
    executor = ExecutorV4(graph=graph, store=store, max_concurrency=2, timeout=10.0)
    result = await executor.run()

    print("\nSuccess:        ", result.success)
    print("Completed edges:", result.completed_edges)
    print("Final output:   ", result.vertex_contents.get("v_out"))
    print("\nSerialized edge (round-trips back to the same classes):")
    print(json.dumps(graph.edges["e_count"].to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
