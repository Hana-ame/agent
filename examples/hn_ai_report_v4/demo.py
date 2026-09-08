#!/usr/bin/env python3
"""HN AI Report V4 — Standalone execution entry point.

Migrates the legacy MapEdge-based HN AI Report to the V4 vertex-edge framework.

Legacy (v1):  MapEdge fan-out — ProcessStoriesMap spawns per-story sub-edges
V4 (this):    Single CodeEdgeV4 — summarize_all() loops internally with asyncio.gather

Usage:
    python examples/hn_ai_report_v4/demo.py
    python examples/hn_ai_report_v4/demo.py --concurrency 3
    python examples/hn_ai_report_v4/demo.py --limit 10

Environment (optional):
    SENSENOVA_API_KEY  — API key for SenseNova LLM
    HN_PROXY           — HTTP proxy for HN API access
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Bootstrap repository root
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.vertex_v4 import VertexStateV4, VertexStoreV4
from framework.graph_v4 import GraphV4
from framework.edge_v4 import CodeEdgeV4, LLMEdgeV4
from framework.executor_v4 import ExecutorV4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hn_ai_report_v4")

CONFIG_PATH = Path(__file__).parent / "config.json"


def build_graph_from_config(
    config_path: Path,
    override: Optional[dict] = None,
) -> tuple[GraphV4, VertexStoreV4]:
    """Load config.json and build a V4 GraphV4 with a VertexStoreV4.

    Returns (graph, store). Caller must close store when done.
    """
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if override:
        cfg.update(override)

    store = VertexStoreV4(":memory:")
    session_id = "hn_ai_v4"

    # --- Create vertices ---
    for v in cfg["vertices"]:
        store.save_vertex(
            session_id=session_id,
            name=v["name"],
            content=v.get("content", ""),
            attributes=v.get("attributes", []),
            state=v.get("state", VertexStateV4.IDLE.value),
        )

    # --- Create graph ---
    graph = GraphV4(session_id=session_id)
    graph = GraphV4.load_from_store(store, session_id, name=cfg["metadata"]["name"])

    # --- Create edges ---
    for e in cfg["edges"]:
        edge_type = e.get("type", "code")
        settings = e.get("settings", {})
        concurrency_group = e.get("concurrency_group")
        concurrency_limit = e.get("concurrency_limit")

        if edge_type == "code":
            edge = CodeEdgeV4(
                edge_id=e["id"],
                input_vertex=e["input_vertex"],
                output_vertex=e["output_vertex"],
                script=e.get("script"),
                settings=settings,
                concurrency_group=concurrency_group,
                concurrency_limit=concurrency_limit,
            )
        elif edge_type == "llm":
            edge = LLMEdgeV4(
                edge_id=e["id"],
                input_vertex=e["input_vertex"],
                output_vertex=e["output_vertex"],
                model=settings.pop("model", "sensenova-6.8-flash-lite"),
                prompt_template=settings.pop("prompt", "{input}"),
                settings=settings,
                concurrency_group=concurrency_group,
                concurrency_limit=concurrency_limit,
            )
        else:
            raise ValueError(f"Unknown edge type: {edge_type}")

        graph.add_edge(edge)

    return graph, store


async def run() -> str:
    """Execute the V4 HN AI Report pipeline and return the final report."""
    parser = argparse.ArgumentParser(description="HN AI Report V4 Pipeline")
    parser.add_argument("--limit", type=int, default=30, help="Max HN stories to fetch")
    parser.add_argument("--concurrency", type=int, default=1, help="Max parallel summaries")
    parser.add_argument("--proxy", default=os.environ.get("HN_PROXY"), help="HTTP proxy for HN API")
    args = parser.parse_args()

    logger.info("Building graph from %s", CONFIG_PATH)
    graph, store = build_graph_from_config(CONFIG_PATH, override={
        "metadata": {"name": "HN AI Report V4"},
    })

    # Apply overrides to edge settings
    for edge in graph.edges.values():
        if hasattr(edge, "settings"):
            if args.limit:
                edge.settings.setdefault("limit", args.limit)
            if args.concurrency:
                edge.settings.setdefault("max_conc", args.concurrency)
            if args.proxy:
                edge.settings.setdefault("proxy", args.proxy)

    # Resolve API key from environment
    api_key = os.environ.get("SENSENOVA_API_KEY", "")
    for edge in graph.edges.values():
        if hasattr(edge, "settings"):
            edge.settings.setdefault("api_key", api_key)

    logger.info("Starting V4 executor (timeout=900s, max_concurrency=4)")
    executor = ExecutorV4(
        graph=graph,
        store=store,
        timeout=900,
        max_concurrency=4,
    )

    try:
        result = await executor.run()
    finally:
        store.close()

    if not result.success:
        logger.error("Pipeline failed: %s", result.errors)
        return ""

    report_text = result.vertex_contents.get("v_report", "")

    logger.info("Report generated (%d chars)", len(report_text))
    return report_text


async def main():
    report = await run()
    if report:
        output_path = Path(__file__).parent / "report.md"
        output_path.write_text(report, encoding="utf-8")
        logger.info("Saved to %s", output_path)
        # Print to stdout for piping
        print("\n" + "=" * 72)
        print("HN AI Report V4 — Pipeline Output")
        print("=" * 72 + "\n")
        print(report)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
