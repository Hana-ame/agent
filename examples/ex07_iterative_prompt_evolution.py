"""Example 07: Iterative Prompt Evolution & Multi-Generation DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 07: Iterative Prompt Evolution DAG ===")
    dag = DAG("prompt_evolution_dag", models="models.json", llm_client=MockLLMClient())

    # Gen 0 -> 1: Candidate prompt generator
    dag.add_edge(
        input_tags=["task_goal"],
        processor=lambda g: f"Prompt_v1: Summarize concisely with bullet points. [Goal: {g}]",
        output_tags=["prompt_v1"],
        layer=0,
    )

    # Gen 1 -> 2: Critic evaluation & score
    dag.add_edge(
        input_tags=["prompt_v1"],
        processor=lambda p: f"Critique: Prompt lacks structured JSON output constraint. Score: 6/10. [Base: {p}]",
        output_tags=["prompt_critique"],
        layer=1,
    )

    # Gen 2 -> 3: Evolutionary Mutation & Optimization
    dag.add_edge(
        input_tags=["prompt_v1", "prompt_critique"],
        processor=lambda bundle: f"Optimized_Prompt_v2: Output strict JSON with 'bullets' list. [Refined from {bundle[:40]}]",
        output_tags=["optimal_prompt"],
        layer=2,
    )

    dag.add_content("提炼文章核心论点并支持下游解析", tags=["task_goal"], gen=0)

    dag.run(until_tag="optimal_prompt", max_steps=5)
    final_prompt = dag.get_result("optimal_prompt")
    print(f"Optimal Prompt: {final_prompt}")
    assert "Optimized_Prompt_v2" in final_prompt
    assert dag.contents[-1].gen >= 3
    print("Example 07 Passed Successfully!")


if __name__ == "__main__":
    main()
