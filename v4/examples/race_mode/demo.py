import asyncio
import sys
import os
import time
import logging
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from framework import Graph, Vertex, Edge, Executor, MockAgent

async def main():
    # We create a Fan-out Fan-in graph.
    # Start -> QuickAgent, Start -> SlowAgent
    # QuickAgent -> End, SlowAgent -> End
    
    config = {
        "vertices": [
            {"id": "Start", "initial_data": [{"channel": "query", "value": "Write a poem"}]},
            {"id": "QuickAgent"},
            {"id": "SlowAgent"},
            {"id": "End", "settings": {"wait_policy": "any"}} # RACE MODE ENABLED!
        ],
        "edges": [
            {"id": "e_start_quick", "source": "Start", "destination": "QuickAgent", "channel": "query"},
            {"id": "e_start_slow", "source": "Start", "destination": "SlowAgent", "channel": "query"},
            {"id": "e_quick_end", "source": "QuickAgent", "destination": "End", "settings": {"prompt": "quick_task"}, "channel": "quick_result"},
            {"id": "e_slow_end", "source": "SlowAgent", "destination": "End", "settings": {"prompt": "slow_task"}, "channel": "slow_result"},
        ]
    }
    
    graph = Graph.from_dict(config)

    # Let's mock an agent that sleeps based on the prompt
    async def race_agent_hook(data, prompt, model, settings):
        if prompt == "quick_task":
            print(f"\n[Agent] QuickAgent starting... (will take 1s)")
            await asyncio.sleep(1)
            print(f"[Agent] QuickAgent finished!")
            return "I am the quick response!"
        elif prompt == "slow_task":
            print(f"\n[Agent] SlowAgent starting... (will take 5s)")
            await asyncio.sleep(5)
            print(f"[Agent] SlowAgent finished! (This should not print if cancelled!)")
            return "I am the slow response!"
        return data

    agent = MockAgent(response_fn=race_agent_hook)
    
    executor = Executor(graph, agents=agent)
    
    start_time = time.time()
    await executor.run()
    elapsed = time.time() - start_time
    
    print("\n--- Race Results ---")
    print(f"Total execution time: {elapsed:.2f}s (Should be ~1s, not 5s!)")
    
    data = await graph.vertices["End"].get_all_data()
    print(f"Final Data at End Vertex: {data}")

if __name__ == "__main__":
    asyncio.run(main())
