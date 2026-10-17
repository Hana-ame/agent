"""Example 08: Meta-DAG Runtime Hot-Patching of Business Workflows."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient, transform_dag_with_meta_dag


def main():
    print("=== Running Example 08: Meta-DAG Runtime Hot-Patching DAG ===")
    
    # Base online DAG with an outdated pipeline
    base_dag = DAG("online_payment_dag", models="models.json", llm_client=MockLLMClient())
    base_dag.add_edge(
        input_tags=["payment_req"],
        processor=lambda req: f"ChargeCard({req})",
        output_tags=["payment_status"],
        layer=0,
    )
    base_dag.add_content("Order#1002: Amount=$2500", tags=["payment_req"], gen=0)

    # Prompt instruction: Hot-patch a 3D Secure / OTP verification edge in front!
    patch_prompt = (
        "添加边: [payment_req] -> (laptop-6.8: 风险评分与OTP二次验证) -> [otp_verified] layer=0; "
        "添加边: [otp_verified] -> (sensenova-6.8: 安全扣款入账) -> [settled_receipt] layer=1"
    )

    # Meta-DAG parses instruction and produces hot-patched DAG
    patched_dag = transform_dag_with_meta_dag(base_dag, patch_prompt, llm_client=MockLLMClient())
    print(f"Original edges: {len(base_dag.edges)} | Patched edges: {len(patched_dag.edges)}")
    assert len(patched_dag.edges) == 3

    # Execute patched workflow
    patched_dag.run(until_tag="settled_receipt", max_steps=5)
    receipt = patched_dag.get_result("settled_receipt")
    print(f"Hot-patched Settlement Output: {receipt[:80]}...")
    assert receipt is not None
    print("Example 08 Passed Successfully!")


if __name__ == "__main__":
    main()
