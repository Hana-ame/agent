"""Example 01: Multi-Quarter Financial Report Analysis DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 01: Financial Report Analysis DAG ===")
    dag = DAG("financial_dag", models="models.json", llm_client=MockLLMClient())

    # Layer 0: Extract revenue & cost
    dag.add_edge(
        input_tags=["raw_financial_text"],
        processor=lambda x: f"Extracted: Revenue=$50M, Cost=$32M, GrossProfit=$18M from {x}",
        output_tags=["financial_metrics"],
        layer=0,
    )

    # Layer 1: Growth trend & margins
    dag.add_edge(
        input_tags=["financial_metrics"],
        processor=lambda x: f"FinancialAnalysis(GrossMargin=36%, YoYGrowth=+22% based on {x})",
        output_tags=["margin_analysis"],
        layer=1,
    )

    # Layer 2: Final Executive Report
    dag.add_edge(
        input_tags=["margin_analysis"],
        processor=lambda x: f"ExecutiveReport: Company maintains solid profit margins. {x}",
        output_tags=["final_cfo_report"],
        layer=2,
    )

    dag.add_content("Q3财报原始数据：总收入5000万美元，营业成本3200万美元", tags=["raw_financial_text"], gen=0)

    dag.run(until_tag="final_cfo_report", max_steps=5)
    res = dag.get_result("final_cfo_report")
    print(f"Result: {res[:100]}... | Steps: {dag.step_count}")
    assert "ExecutiveReport" in res
    assert dag.contents[-1].gen >= 2
    print("Example 01 Passed Successfully!")


if __name__ == "__main__":
    main()
