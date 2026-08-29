#!/usr/bin/env python3
"""OpenCode CLI Agent example.

Runs ``config.json`` — a single edge loaded from
``script: zen_edge.py:OpenCodeEdge`` that launches the local ``opencode``
CLI to process the prompt.

Everything is declared in the config (edge class + prompt + model). The
runner only loads the graph and executes it — no agent is injected and no
fallback default agent is used.

    python examples/opencode_zen/run.py
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

    # The edge owns its agent (OpenCodeAgentRunner) via config's `script`.
    executor = Executor(graph)

    result = await executor.run()
    print("\n" + result.summary())

    data = await graph.vertices["zen_out"].fetch_data()
    print(f"\n--- zen_out ---\n{data}")


if __name__ == "__main__":
    asyncio.run(main())
