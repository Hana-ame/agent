"""Example 09: Multi-Agent Adversarial Debate & Neutral Judge DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 09: Multi-Agent Debate & Judge DAG ===")
    dag = DAG("debate_dag", models="models.json", llm_client=MockLLMClient())

    # Agent Pro (Advocate): arguments in favor
    dag.add_edge(
        input_tags=["motion"],
        processor=lambda m: f"AgentPro: Remote work boosts productivity by 35% and cuts office overhead. [Motion: {m}]",
        output_tags=["pro_arguments"],
        layer=0,
    )

    # Agent Con (Critic): counter-arguments
    dag.add_edge(
        input_tags=["motion"],
        processor=lambda m: f"AgentCon: Remote work weakens team cohesion and increases onboarding friction. [Motion: {m}]",
        output_tags=["con_arguments"],
        layer=0,
    )

    # Neutral Judge (Convergence): synthesize trade-offs into balanced policy
    dag.add_edge(
        input_tags=["pro_arguments", "con_arguments"],
        processor=lambda deb: f"NeutralJudgeRuling: Adopt a 3+2 hybrid model to balance productivity and culture. [Debate: {deb[:60]}]",
        output_tags=["final_ruling"],
        layer=1,
    )

    dag.add_content("科技企业是否应永久全面实行远程办公？", tags=["motion"], gen=0)

    dag.run(until_tag="final_ruling", max_steps=5)
    ruling = dag.get_result("final_ruling")
    print(f"Judge Ruling: {ruling[:100]}...")
    assert "NeutralJudgeRuling" in ruling
    print("Example 09 Passed Successfully!")


if __name__ == "__main__":
    main()
