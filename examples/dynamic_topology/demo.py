import asyncio
import sys
import os
import json
import logging
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from framework import Graph, Vertex, Edge, Executor, MockAgent

# 1. Define the async hook for the Edge
async def post_process(data, settings):
    # This is a post_process hook on the Manager's edge.
    # The LLM (Manager) has output a list of tasks.
    tasks = data if isinstance(data, list) else []
    print(f"\n[Dynamic Orchestrator] Manager determined {len(tasks)} sub-tasks. Spawning dynamic graph on the fly...")
    
    # Build a dynamic graph using a dict config
    config = {
        "vertices": [
            {"id": "Collector"}
        ],
        "edges": []
    }
    
    for i, task in enumerate(tasks):
        v_name = f"Worker_{i}"
        e_name = f"e_work_{i}"
        
        # Add a dynamic source node for this task
        config["vertices"].append({
            "id": v_name,
            "initial_data": [{"channel": f"result_{i}", "value": task}]
        })
        
        # Add an edge from the worker to the collector
        config["edges"].append({
            "id": e_name,
            "source": v_name,
            "destination": "Collector",
            "channel": f"result_{i}",
            "prompt": f"Execute task: {task}"
        })
        
    dynamic_graph = Graph.from_dict(config)
    
    # We will use an inline on_ready hook on the Collector to merge the results
    def merge_results(store, sets):
        return {"final_report": list(store.values())}
    dynamic_graph.vertices["Collector"].on_ready = merge_results
    
    # Execute the dynamic graph using a nested Executor
    print("[Dynamic Orchestrator] Executing dynamic sub-graph asynchronously...")
    # We use a MockAgent that just does some fake work based on the prompt
    worker_agent = MockAgent(response_fn=lambda d, p, m, s: f"Result of [{list(d.values())[0] if isinstance(d, dict) else d}]")
    sub_executor = Executor(dynamic_graph, agents=worker_agent)
    
    await sub_executor.run()
    
    # Hoist the result from the dynamic sink back to the parent data stream
    final_data = await dynamic_graph.vertices["Collector"].fetch_data("final_report")
    print(f"[Dynamic Orchestrator] Sub-graph finished! Final merged data: {final_data}\n")
    return final_data


async def main():
    # We create a simple parent graph using from_dict
    parent_config = {
        "vertices": [
            {"id": "Manager", "initial_data": [{"channel": "default", "value": "Analyze market"}]},
            {"id": "FinalSink"}
        ],
        "edges": [
            {
                "id": "e_manage",
                "source": "Manager",
                "destination": "FinalSink",
                "prompt": "Break down into tasks"
            }
        ]
    }
    
    parent_graph = Graph.from_dict(parent_config)

    # The MockAgent for Manager will output a list of 3 tasks
    manager_agent = MockAgent(
        response_fn=lambda d, p, m, s: ["Task 1: Search SEO", "Task 2: Analyze Competitors", "Task 3: Write Copy"]
    )
    
    # IMPORTANT: Attach the async hook to the edge
    parent_graph.edges["e_manage"].post_process = post_process

    executor = Executor(parent_graph, agents=manager_agent)
    await executor.run()
    
    res = await parent_graph.vertices["FinalSink"].fetch_data("default")
    print(f"Parent graph execution complete. Final result in FinalSink:\n{json.dumps(res, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    asyncio.run(main())
