import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from framework import Executor, MockAgent
from framework.builders.chain import LinearChain

async def main():
    # Instead of writing a complex JSON with explicit Vertices and Edges,
    # just provide a list of prompts for a simple linear workflow!
    prompts = [
        "Translate the input to French",
        "Summarize the translation into one sentence"
    ]
    
    graph = LinearChain.build(prompts)
    
    # Inject initial data into the first node (Node_0)
    await graph.vertices["Node_0"].set_data("default", "Hello world, the weather is very nice today.")
    
    # Run the executor
    agent = MockAgent(response_fn=lambda d, p, m, s: f"[{p}] -> Processed({d})")
    executor = Executor(graph, agents=agent)
    
    print("Running Simple Linear Chain...")
    await executor.run()
    
    # Fetch result from the last node
    last_node = f"Node_{len(prompts)}"
    result = await graph.vertices[last_node].fetch_data("default")
    
    print("\nFinal Result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
