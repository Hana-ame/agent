"""Example 03: Competitor Intelligence Map-Reduce DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 03: Map-Reduce Market Research DAG ===")
    dag = DAG("market_research_dag", models="models.json", llm_client=MockLLMClient())

    # Branch 1 (Competitor A analysis)
    dag.add_edge(
        input_tags=["sector_overview"],
        processor=lambda x: "CompetitorA: Leading cloud provider with 40% enterprise share.",
        output_tags=["comp_a_intel"],
        layer=0,
    )

    # Branch 2 (Competitor B analysis)
    dag.add_edge(
        input_tags=["sector_overview"],
        processor=lambda x: "CompetitorB: Open-source challenger growing at 80% YoY.",
        output_tags=["comp_b_intel"],
        layer=0,
    )

    # Branch 3 (Regulatory / Policy analysis)
    dag.add_edge(
        input_tags=["sector_overview"],
        processor=lambda x: "PolicyLandscape: Stricter data sovereignty requirements expected.",
        output_tags=["policy_intel"],
        layer=0,
    )

    # Map-Reduce Convergence
    dag.add_edge(
        input_tags=["comp_a_intel", "comp_b_intel", "policy_intel"],
        processor=lambda x: f"StrategicSWOT: Focus on hybrid-cloud sovereignty niche. Inputs: {x}",
        output_tags=["swot_matrix"],
        layer=1,
    )

    dag.add_content("2026年全球智能体云基础设施行业报告", tags=["sector_overview"], gen=0)

    dag.run(until_tag="swot_matrix", max_steps=6)
    res = dag.get_result("swot_matrix")
    print(f"SWOT Result: {res[:100]}... | Steps: {dag.step_count}")
    assert "StrategicSWOT" in res
    print("Example 03 Passed Successfully!")


if __name__ == "__main__":
    main()
