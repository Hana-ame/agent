#!/usr/bin/env python3
"""Example: Nested Sub-Graphs (Hierarchical Agent Teams).

Demonstrates how a parent workflow ("EditorialOffice") delegates a complex task
to a composite "ResearchDepartment" (a nested 2-agent subgraph: WebSearcher -> FactChecker),
with automatic data boundary mapping and hierarchical event stream bubbling.

Run:
    python examples/subgraph/demo.py
"""

import asyncio
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from framework import Graph, Executor, MockAgent

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def main():
    print(f"\n{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}{CYAN}   v3.0 Demo: Nested Sub-Graphs (Hierarchical Agent Teams) {RESET}")
    print(f"{BOLD}{CYAN}============================================================{RESET}\n")

    # 1. Inner Subgraph: The Research Department Team (WebSearcher -> FactChecker)
    inner_research_subgraph = {
        "metadata": {"name": "ResearchDepartment_Subgraph"},
        "vertices": [
            {"id": "WebSearcher"},
            {"id": "FactChecker"},
        ],
        "edges": [
            {
                "id": "e_search_to_check",
                "source": "WebSearcher",
                "destination": "FactChecker",
                "channel": "query",
                "prompt": "Find verified raw facts and statistics for query",
            }
        ],
    }

    # 2. Parent Graph: Editorial Pipeline (EditorInChief -> ResearchDepartment -> LeadWriter)
    parent_config = {
        "metadata": {"name": "Editorial_Production_Graph"},
        "vertices": [
            {
                "id": "EditorInChief",
                "initial_data": [
                    {
                        "channel": "assignment",
                        "value": "Breakthroughs in Fusion Energy 2026",
                    }
                ],
            },
            {
                "id": "ResearchDepartment",
                "type": "subgraph",  # Instantiated as SubgraphVertex
                "settings": {
                    "graph_config": inner_research_subgraph,
                    # Map parent input channel 'assignment' -> Inner vertex 'WebSearcher' channel 'query'
                    "input_map": {"assignment": "WebSearcher.query"},
                    # Map inner vertex 'FactChecker' channel 'query' -> Parent output channel 'verified_facts'
                    "output_map": {"FactChecker.query": "verified_facts"},
                },
            },
            {
                "id": "LeadWriter",
            },
        ],
        "edges": [
            {
                "id": "e_assign_research",
                "source": "EditorInChief",
                "destination": "ResearchDepartment",
                "channel": "assignment",
                "prompt": "Assign investigation topic",
            },
            {
                "id": "e_deliver_facts_to_writer",
                "source": "ResearchDepartment",
                "destination": "LeadWriter",
                "channel": "verified_facts",
                "prompt": "Draft complete feature article using facts",
            },
        ],
    }

    # Simulate realistic agent responses with small async delays
    async def simulated_llm(data, prompt, model, settings):
        await asyncio.sleep(0.3)
        if "Find verified raw facts" in prompt:
            return f"[Verified Facts on '{data}': Net energy gain achieved 1.8x, Q > 1]"
        elif "Draft complete feature article" in prompt:
            return f"📰 [ARTICLE PUBLISHED]: Breakthrough discoveries confirm {data}"
        return f"[Processed: {data}]"

    parent_graph = Graph.from_dict(parent_config)
    agent = MockAgent(response_fn=simulated_llm)
    executor = Executor(parent_graph, agents=agent, max_concurrency=4)

    print(f"{BOLD}Streaming Live Hierarchical Execution Events:{RESET}\n")

    # Stream parent and bubbled inner events
    async for event in executor.stream():
        ts = event.timestamp.split("T")[1].replace("Z", "")
        
        # Check if event is bubbled from nested subgraph
        if event.event_type.startswith("subgraph_"):
            inner_type = event.event_type.replace("subgraph_", "")
            print(f"  [{ts}] {MAGENTA}↳ [SUBGRAPH EVENT]{RESET} {BOLD}{inner_type}{RESET} on {YELLOW}{event.vertex_id}{RESET}")
        else:
            if event.event_type == "workflow_started":
                print(f"[{ts}] {BOLD}{GREEN}▶ PARENT WORKFLOW STARTED{RESET}")
            elif event.event_type == "vertex_state_changed":
                state = event.payload.get("state")
                print(f"[{ts}] {BLUE}● Node [{event.vertex_id}]{RESET} state: {BOLD}{state.upper()}{RESET}")
            elif event.event_type == "edge_started":
                src = event.payload.get("source")
                dst = event.payload.get("destination")
                print(f"[{ts}] {CYAN}⚡ Edge [{event.edge_id}]{RESET} ({src} ──▶ {dst})")
            elif event.event_type == "edge_completed":
                print(f"[{ts}] {GREEN}✔ Edge [{event.edge_id}]{RESET} finished delivery")
            elif event.event_type == "workflow_finished":
                print(f"\n[{ts}] {BOLD}{GREEN}■ PARENT WORKFLOW FINISHED{RESET} (success={event.payload.get('success')})")

    # Print final output at the LeadWriter sink node
    print(f"\n{BOLD}Final Output at LeadWriter Node:{RESET}")
    writer_data = await parent_graph.vertices["LeadWriter"].get_all_data()
    for k, v in writer_data.items():
        print(f"  • Channel '{k}': {GREEN}{v}{RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
