"""Example 04: Tool-Augmented DAG using `tools.json` Calculator."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 04: Tool-Augmented Calculator DAG ===")
    # Loads tools.json into dedicated DAG
    dag = DAG("calculator_dag", models="models.json", tools="tools.json", llm_client=MockLLMClient())

    # Verify tool is mounted
    assert "calculator" in dag.tools
    assert "search_web" in dag.tools

    # Edge 0: Extract numeric formula from query
    dag.add_edge(
        input_tags=["financial_query"],
        processor=lambda q: "(52000000 - 32000000) / 52000000 * 100",
        output_tags=["math_expression"],
        layer=0,
    )

    # Edge 1: Invoke DAG tool `calculator` to compute exact margin
    def _compute_via_tool(expr: str) -> str:
        calc_result = dag.call_tool("calculator", expression=expr.strip())
        return f"GrossMarginPercentage: {calc_result:.2f}%"

    dag.add_edge(
        input_tags=["math_expression"],
        processor=_compute_via_tool,
        output_tags=["exact_margin_result"],
        layer=1,
    )

    dag.add_content("总营收5200万美元，营业成本3200万美元，请计算精准毛利率", tags=["financial_query"], gen=0)

    dag.run(until_tag="exact_margin_result", max_steps=5)
    margin_res = dag.get_result("exact_margin_result")
    print(f"Tool Calculation Result: {margin_res}")
    assert "38.46%" in margin_res
    print("Example 04 Passed Successfully!")


if __name__ == "__main__":
    main()
