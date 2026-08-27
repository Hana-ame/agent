#!/usr/bin/env python3
"""Run a vertex-edge framework example pipeline.

Usage:
    python examples/run.py <path_to_config.json>
"""

import asyncio
import logging
import os
import sys

# Allow running from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from framework import Graph, Executor, MockPIAgent

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

async def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    setup_logging()
    logger = logging.getLogger("example_runner")

    logger.info("Loading graph from %s", config_path)
    graph = Graph.from_json(config_path)

    # Use MockPIAgent — replace with ExternalPIAgent() when pi_agent is installed
    agent = MockPIAgent()
    executor = Executor(graph, pi_agent=agent, max_concurrency=4, timeout=30)

    result = await executor.run()
    print("\n" + result.summary())
    return result

if __name__ == "__main__":
    asyncio.run(main())
