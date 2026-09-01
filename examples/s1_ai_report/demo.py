import logging
logging.basicConfig(level=logging.INFO)
import asyncio
import os
import sys

# Add framework root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from framework.graph import Graph
from framework.executor.base import Executor
from framework.agents import HttpLLMAgent

async def main():
    print("Loading graph from config.json...")
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    g = Graph.from_json(config_path)

    agent = HttpLLMAgent(
        api_key=os.environ.get("LLM_API_KEY", "public"),
        base_url=os.environ.get("LLM_BASE_URL", "https://opencode.ai/zen/v1/chat/completions"),
    )

    print("Executing graph with HttpLLMAgent...")
    try:
        executor = Executor(
            g,
            agents=agent,
            max_concurrency=10,
            concurrency_config={"llm": 1, "fetch": 10, "default": 10},
        )
        await executor.run()
    finally:
        await agent.close()

    report_path = os.path.join(os.path.dirname(__file__), "report.md")
    if os.path.exists(report_path):
        print("\n=== SUCCESS: GENERATED REPORT ===")
        with open(report_path, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("\nERROR: Report file was not generated.")

if __name__ == "__main__":
    asyncio.run(main())
