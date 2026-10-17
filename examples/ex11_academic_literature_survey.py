"""Example 11: Academic Paper Literature Survey DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 11: Academic Literature Survey DAG ===")
    dag = DAG("literature_dag", models="models.json", llm_client=MockLLMClient())

    # Edge 0: Method extraction
    dag.add_edge(
        input_tags=["paper_abstract"],
        processor=lambda ab: f"Methodology: Transformer with linear attention O(N) complexity from {ab[:30]}",
        output_tags=["method_summary"],
        layer=0,
    )

    # Edge 1: Benchmark and empirical results
    dag.add_edge(
        input_tags=["paper_abstract"],
        processor=lambda ab: f"EmpiricalResults: Achieves 84.5% on ImageNet-1K with 2x speedup from {ab[:30]}",
        output_tags=["benchmark_results"],
        layer=0,
    )

    # Edge 2: Research gaps & future directions
    dag.add_edge(
        input_tags=["method_summary", "benchmark_results"],
        processor=lambda x: f"ResearchGapSynthesis: High throughput on short contexts, but degradation on 128k+ long contexts. {x[:60]}",
        output_tags=["research_survey_digest"],
        layer=1,
    )

    dag.add_content("We present LinearFormer, a linear complexity attention mechanism for vision-language tasks...", tags=["paper_abstract"], gen=0)

    dag.run(until_tag="research_survey_digest", max_steps=5)
    digest = dag.get_result("research_survey_digest")
    print(f"Survey Digest: {digest[:100]}...")
    assert "ResearchGapSynthesis" in digest
    print("Example 11 Passed Successfully!")


if __name__ == "__main__":
    main()
