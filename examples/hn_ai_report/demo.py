import logging
logging.basicConfig(level=logging.INFO)
import asyncio
import os
import sys
# Add framework root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from framework.graph import Graph
from framework.executor.base import Executor
from framework.executor.context import ExecutionContext
from framework.agents import MockAgent
async def main():
    print("Loading graph from config.json...")
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    g = Graph.from_json(config_path)
    agent = MockAgent(
    )
    print("Executing graph with MockAgent...")
    async with ExecutionContext(agents=agent) as ctx:
        executor = Executor(
            g,
            context=ctx,
            max_concurrency=10,
            concurrency_config={"llm": 1, "fetch": 10, "default": 10},
        )
        await executor.run()
    report_path = os.path.join(os.path.dirname(__file__), "report.md")
    if os.path.exists(report_path):
        print("\n=== SUCCESS: GENERATED REPORT ===")
        with open(report_path, "r", encoding="utf-8") as f:
            print(f.read()[:500] + "\n... (truncated)")
    else:
        print("\nERROR: Report file was not generated.")
if __name__ == "__main__":
    asyncio.run(main())
