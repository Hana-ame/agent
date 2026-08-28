#!/usr/bin/env python3
"""Example 2: LLM Output Error Self-Correction via Retry Policy.

Demonstrates how `EdgePipeline` intercepts business logic exceptions (e.g. JSON schema errors,
missing keys in post-process), automatically reflects the error back into the prompt
(`[SYSTEM FEEDBACK: ...]`), and retries with exponential backoff until success.

Run:
    python examples/self_correction/demo.py
"""

import asyncio
import json
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from framework import Graph, Executor, MockAgent

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def main():
    print(f"\n{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}{CYAN}   v2.0 Demo: LLM Error Interception & Self-Correction    {RESET}")
    print(f"{BOLD}{CYAN}============================================================{RESET}\n")

    attempt_counter = {"count": 0}

    # Simulate an LLM that makes mistakes before getting it right
    def mock_flaky_llm(data, prompt, model, settings):
        attempt_counter["count"] += 1
        n = attempt_counter["count"]
        print(f"  {BOLD}Attempt #{n}:{RESET} LLM received prompt:")
        
        # Display the prompt (noting if system feedback is injected)
        if "[SYSTEM FEEDBACK:" in prompt:
            feedback_part = prompt.split("[SYSTEM FEEDBACK:")[1]
            print(f"    ↳ {YELLOW}[SYSTEM FEEDBACK INJECTED]:{RESET} {feedback_part.strip()}")
        else:
            print(f"    ↳ Initial Prompt: \"{prompt}\"")

        if n == 1:
            print(f"    ↳ {RED}LLM Output:{RESET} 'I am an AI and I like apples' (Invalid JSON)")
            return "I am an AI and I like apples"
        elif n == 2:
            print(f"    ↳ {RED}LLM Output:{RESET} '{{\"wrong_key\": 42}}' (Missing 'extracted_entities' key)")
            return '{"wrong_key": 42}'
        else:
            print(f"    ↳ {GREEN}LLM Output:{RESET} '{{\"extracted_entities\": [\"Apple\", \"Google\"]}}' (Valid!)")
            return '{"extracted_entities": ["Apple", "Google"]}'

    # Post-process hook expecting strict JSON with "extracted_entities"
    def strict_post_process(result, settings):
        # May raise json.JSONDecodeError or KeyError
        data = json.loads(result)
        if "extracted_entities" not in data:
            raise KeyError(f"Missing required key 'extracted_entities', got keys: {list(data.keys())}")
        return data["extracted_entities"]

    config = {
        "vertices": [
            {"id": "RawInput", "initial_data": [{"channel": "article", "value": "Big tech report"}]},
            {"id": "StructuredOutput"},
        ],
        "edges": [
            {
                "id": "e_extract",
                "source": "RawInput",
                "destination": "StructuredOutput",
                "channel": "article",
                "settings": {
                    "prompt": "Extract company entities in strict JSON format: {\"extracted_entities\": [...]}",
                    "retry_policy": {
                        "max_retries": 3,
                        "backoff_factor": 0.2,
                        "retry_on": ["JSONDecodeError", "KeyError", "ValueError"],
                    }
                },
            }
        ],
    }

    graph = Graph.from_dict(config)
    
    # Inject our post-process hook via Edge subclass
    from framework.edge import Edge
    class ExtractEdge(Edge):
        def post_process(self, result, settings):
            return strict_post_process(result, settings)

    old_edge = graph.edges["e_extract"]
    graph.edges["e_extract"] = ExtractEdge(
        edge_id=old_edge.id, source_id=old_edge.source_id,
        destination_id=old_edge.destination_id, channel=old_edge.channel,
        settings=old_edge.settings, concurrency_type=old_edge.concurrency_type,
        max_iterations=old_edge.max_iterations
    )

    agent = MockAgent(response_fn=mock_flaky_llm)
    executor = Executor(graph, agents=agent)

    print(f"{BOLD}Starting execution with retry_policy (max_retries=3)...{RESET}\n")
    result = await executor.run()

    print(f"\n{BOLD}Result:{RESET} {'✔ SUCCESS' if result.success else '✖ FAILED'}")
    output_data = await graph.vertices["StructuredOutput"].get_all_data()
    print(f"Final extracted data in 'StructuredOutput' node: {GREEN}{output_data}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
