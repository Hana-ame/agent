#!/usr/bin/env python3
"""OpenCode Zen + Proxied LLM example.

Runs ``config.json`` — a fan-out graph where one edge calls OpenCode Zen
directly (free, no key, self-throttled) and the other routes through a
self-hosted proxy/gateway with model aliasing.

Both agents are wired **declaratively** from ``config.json`` via the per-edge
``settings.agent`` block, so this entrypoint stays agent-agnostic:

    python examples/opencode_zen/run.py

The OpenCode edge works out of the box. The proxy edge needs a gateway
listening at ``http://localhost:8000/v1`` (e.g. LiteLLM / one-api) — set
``LLM_PROXY_BASE_URL`` / ``LLM_PROXY_API_KEY`` to point elsewhere.
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from framework import Executor, Graph


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


async def main() -> None:
    setup_logging()
    config = os.path.join(os.path.dirname(__file__), "config.json")
    graph = Graph.from_json(config)

    # agents=None → each edge uses the agent declared in its own settings.
    executor = Executor(graph, agents=None, max_concurrency=4)

    result = await executor.run()
    print("\n" + result.summary())

    for vid in ("zen_out", "proxy_out"):
        data = await graph.vertices[vid].fetch_data()
        print(f"\n--- {vid} ---\n{data}")


if __name__ == "__main__":
    asyncio.run(main())
