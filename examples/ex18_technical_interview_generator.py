"""Example 18: Technical Interview Question Generator & Rubric DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 18: Technical Interview Generator DAG ===")
    dag = DAG("interview_gen_dag", models="models.json", llm_client=MockLLMClient(), similarity_threshold=0.75)

    # Edge 0: Extract required core competencies from Job Description
    dag.add_edge(
        input_tags=["job_description"],
        processor=lambda jd: "Competencies: Distributed consensus (Raft/Paxos), Goroutine concurrency, high QPS cache design.",
        output_tags=["core_competencies"],
        layer=0,
    )

    # Edge 1: Generate coding challenge
    dag.add_edge(
        input_tags=["core_competencies"],
        processor=lambda c: f"CodingProblem: Implement a distributed rate limiter using sliding-window counter in Go. Competencies: {c[:40]}",
        output_tags=["coding_problem"],
        layer=1,
    )

    # Edge 2: Generate interviewer grading rubric & follow-up questions
    dag.add_edge(
        input_tags=["coding_problem"],
        processor=lambda prob: f"InterviewerRubric: Level 1: Mutex locks (Basic). Level 2: Lock-free atomic (Senior). Follow-up: Clock skew handling. [{prob[:30]}]",
        output_tags=["interviewer_rubric"],
        layer=2,
    )

    dag.add_content("Senior Backend Engineer (Go / Distributed Systems) - 5+ years experience", tags=["job_description"], gen=0)

    dag.run(until_tag="interviewer_rubric", max_steps=5)
    rubric = dag.get_result("interviewer_rubric")
    print(f"Interviewer Rubric: {rubric[:100]}...")
    assert "InterviewerRubric" in rubric
    print("Example 18 Passed Successfully!")


if __name__ == "__main__":
    main()
