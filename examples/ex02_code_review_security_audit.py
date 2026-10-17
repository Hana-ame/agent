"""Example 02: Code Review & Security Audit DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 02: Code Review & Security Audit DAG ===")
    dag = DAG("security_audit_dag", models="models.json", llm_client=MockLLMClient())

    # Branch 1 (Layer 0): Static Code Quality Analysis
    dag.add_edge(
        input_tags=["source_code"],
        processor=lambda code: f"CodeQuality: Clean structure, 0 lint warnings in {code[:30]}",
        output_tags=["quality_report"],
        layer=0,
    )

    # Branch 2 (Layer 0): Security Vulnerability Scan (SQLi, XSS)
    dag.add_edge(
        input_tags=["source_code"],
        processor=lambda code: f"SecurityScan: Detected raw SQL query, potential SQLi risk in {code[:30]}",
        output_tags=["vulnerability_report"],
        layer=0,
    )

    # Convergence (Layer 1): Combined Audit & Patch Suggestions
    dag.add_edge(
        input_tags=["quality_report", "vulnerability_report"],
        processor=lambda reports: f"SecurityRemediationPlan: Use parameterized queries to fix SQLi. Details: {reports}",
        output_tags=["audit_passed_patch"],
        layer=1,
    )

    dag.add_content("def query_user(name): return db.execute(f'SELECT * FROM users WHERE name = {name}')", tags=["source_code"], gen=0)

    dag.run(until_tag="audit_passed_patch", max_steps=5)
    final_res = dag.get_result("audit_passed_patch")
    print(f"Audit Result: {final_res[:100]}... | Steps: {dag.step_count}")
    assert "SecurityRemediationPlan" in final_res
    print("Example 02 Passed Successfully!")


if __name__ == "__main__":
    main()
