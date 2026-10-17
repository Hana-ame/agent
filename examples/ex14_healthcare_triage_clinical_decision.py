"""Example 14: Healthcare Symptom Triage & Clinical Decision Support DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 14: Healthcare Symptom Triage DAG ===")
    dag = DAG("healthcare_triage_dag", models="models.json", llm_client=MockLLMClient())

    # Layer 0: Emergency Red Flag Screening
    dag.add_edge(
        input_tags=["patient_intake"],
        processor=lambda p: "RedFlagCheck: No acute shortness of breath or chest pain. Vital signs stable.",
        output_tags=["red_flags_cleared"],
        layer=0,
    )

    # Layer 1: Specialty Department Recommendation
    dag.add_edge(
        input_tags=["patient_intake", "red_flags_cleared"],
        processor=lambda p: "Referral: Recommended department is Dermatology (Skin Clinic). Level 3 Non-Urgent.",
        output_tags=["specialty_referral"],
        layer=1,
    )

    dag.add_content("Patient age 34, mild erythematous rash on forearm for 3 days, no fever.", tags=["patient_intake"], gen=0)

    dag.run(until_tag="specialty_referral", max_steps=5)
    ref = dag.get_result("specialty_referral")
    print(f"Clinical Referral: {ref}")
    assert "Dermatology" in ref
    print("Example 14 Passed Successfully!")


if __name__ == "__main__":
    main()
