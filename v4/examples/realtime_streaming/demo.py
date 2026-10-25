#!/usr/bin/env python3
"""Example 1: Real-Time Event Streaming with ANSI Color Output.

Demonstrates how to consume `executor.stream()` to get live, non-blocking
event observability during graph execution (e.g. for WebSockets or UI dashboards).

Run:
    python examples/realtime_streaming/demo.py
"""

import asyncio
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from framework import Graph, Executor, MockAgent

# ANSI Color Codes for beautiful terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def main():
    print(f"\n{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}{CYAN}   v2.0 Demo: Real-Time Non-Blocking Event Streaming   {RESET}")
    print(f"{BOLD}{CYAN}============================================================{RESET}\n")

    # Multi-step fan-out/fan-in pipeline
    config = {
        "vertices": [
            {"id": "DataIngest", "initial_data": [{"channel": "topic", "value": "Quantum Computing"}]},
            {"id": "Analyzer_A"},
            {"id": "Analyzer_B"},
            {"id": "Aggregator"},
        ],
        "edges": [
            {
                "id": "e_ingest_to_a",
                "source": "DataIngest",
                "destination": "Analyzer_A",
                "channel": "topic",
                "settings": {"prompt": "Analyze market trends for {topic}"}
            },
            {
                "id": "e_ingest_to_b",
                "source": "DataIngest",
                "destination": "Analyzer_B",
                "channel": "topic",
                "settings": {"prompt": "Analyze technical challenges for {topic}"}
            },
            {
                "id": "e_a_to_agg",
                "source": "Analyzer_A",
                "destination": "Aggregator",
                "channel": "topic",
                "settings": {"prompt": "Summarize market analysis"}
            },
            {
                "id": "e_b_to_agg",
                "source": "Analyzer_B",
                "destination": "Aggregator",
                "channel": "topic",
                "settings": {"prompt": "Summarize tech analysis"}
            },
        ],
    }

    # Simulate slight async delay in mock agent to show concurrent streaming
    async def simulated_llm(data, prompt, model, settings):
        await asyncio.sleep(0.3)
        return f"[Result for: {prompt[:30]}...]"

    graph = Graph.from_dict(config)
    agent = MockAgent(response_fn=simulated_llm)
    executor = Executor(graph, agents=agent, max_concurrency=4)

    # Stream events live
    async for event in executor.stream():
        ts = event.timestamp.split("T")[1].replace("Z", "")
        
        if event.event_type == "workflow_started":
            print(f"[{ts}] {BOLD}{GREEN}▶ WORKFLOW STARTED{RESET} (concurrency={event.payload.get('concurrency')})")

        elif event.event_type == "vertex_state_changed":
            state = event.payload.get("state")
            color = YELLOW if state == "awaiting_edges" else (GREEN if state == "done" else MAGENTA)
            print(f"[{ts}] {color}● Vertex [{event.vertex_id}]{RESET} state transitioned to {BOLD}{state.upper()}{RESET}")

        elif event.event_type == "edge_started":
            src = event.payload.get("source")
            dst = event.payload.get("destination")
            print(f"[{ts}] {BLUE}⚡ Edge [{event.edge_id}]{RESET} firing: {src} ──▶ {dst}")

        elif event.event_type == "edge_completed":
            print(f"[{ts}] {GREEN}✔ Edge [{event.edge_id}]{RESET} completed payload delivery")

        elif event.event_type == "edge_aborted":
            print(f"[{ts}] {MAGENTA}✖ Edge [{event.edge_id}]{RESET} aborted: {event.payload.get('reason')}")

        elif event.event_type == "workflow_finished":
            success = event.payload.get("success")
            time_taken = event.payload.get("execution_time")
            status_text = f"{GREEN}SUCCESS{RESET}" if success else "\033[91mFAILED\033[0m"
            print(f"\n[{ts}] {BOLD}■ WORKFLOW FINISHED:{RESET} {status_text} in {time_taken:.3f}s\n")

    # Print final result collected at sink
    agg_data = await graph.vertices["Aggregator"].get_all_data()
    print(f"{BOLD}Final Data at Aggregator Node:{RESET}")
    for k, v in agg_data.items():
        print(f"  • {k}: {v}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
