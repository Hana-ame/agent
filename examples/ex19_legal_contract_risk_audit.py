"""Example 19: Commercial Contract Legal Compliance & Risk Audit DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 19: Legal Contract Risk Audit DAG ===")
    dag = DAG("contract_audit_dag", models="models.json", llm_client=MockLLMClient())

    # Branch 1: Exclusivity & Non-compete Clause Analysis
    dag.add_edge(
        input_tags=["contract_clause"],
        processor=lambda cl: "ExclusivityRisk: Clause 4.2 imposes a 3-year global non-compete. High antitrust liability.",
        output_tags=["exclusivity_audit"],
        layer=0,
    )

    # Branch 2: Liability & Indemnification Cap
    dag.add_edge(
        input_tags=["contract_clause"],
        processor=lambda cl: "LiabilityRisk: Clause 9.1 has uncapped indirect damages. Should cap at 12 months fees.",
        output_tags=["liability_audit"],
        layer=0,
    )

    # Convergence: General Counsel Redline Advice
    dag.add_edge(
        input_tags=["exclusivity_audit", "liability_audit"],
        processor=lambda audits: f"GeneralCounselRedline: 1) Narrow non-compete to direct competitors for 1 year. 2) Cap liability at 1x annual contract value. Audits: {audits[:50]}",
        output_tags=["contract_redline_summary"],
        layer=1,
    )

    dag.add_content("Master Services Agreement Section 4 & 9 (Confidential Draft)...", tags=["contract_clause"], gen=0)

    dag.run(until_tag="contract_redline_summary", max_steps=5)
    redline = dag.get_result("contract_redline_summary")
    print(f"Contract Redline: {redline[:100]}...")
    assert "GeneralCounselRedline" in redline
    print("Example 19 Passed Successfully!")


if __name__ == "__main__":
    main()
