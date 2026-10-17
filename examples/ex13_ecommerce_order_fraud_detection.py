"""Example 13: E-Commerce Order Fraud & Risk Decision DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 13: E-Commerce Fraud Detection DAG ===")
    dag = DAG("fraud_detection_dag", models="models.json", llm_client=MockLLMClient())

    # Branch 1: Device and IP Geolocation risk
    dag.add_edge(
        input_tags=["order_raw"],
        processor=lambda o: "GeoRisk: High risk (Order IP in Tokyo, shipping address in New York)",
        output_tags=["geo_risk_score"],
        layer=0,
    )

    # Branch 2: Card velocity & transaction amount risk
    dag.add_edge(
        input_tags=["order_raw"],
        processor=lambda o: "VelocityRisk: 4 transactions in last 2 minutes, Amount=$4200",
        output_tags=["velocity_risk_score"],
        layer=0,
    )

    # Convergence: Combined Decision
    dag.add_edge(
        input_tags=["geo_risk_score", "velocity_risk_score"],
        processor=lambda s: f"ACTION: BLOCK_TRANSACTION. Suspicious cross-border high velocity. [{s[:50]}]",
        output_tags=["fraud_action"],
        layer=1,
    )

    dag.add_content("Order#8842: Card=4111xxxx Amount=$4200 IP=203.0.113.1", tags=["order_raw"], gen=0)

    dag.run(until_tag="fraud_action", max_steps=5)
    action = dag.get_result("fraud_action")
    print(f"Risk Decision: {action}")
    assert "BLOCK_TRANSACTION" in action
    print("Example 13 Passed Successfully!")


if __name__ == "__main__":
    main()
