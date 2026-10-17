"""Example 17: Microservice Incident Root Cause Analysis (RCA) DAG."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import DAG, MockLLMClient


def main():
    print("=== Running Example 17: Microservice Incident RCA DAG ===")
    dag = DAG("incident_rca_dag", models="models.json", llm_client=MockLLMClient())

    # Branch 1: Prometheus Metrics (CPU, Memory, Pod restarts)
    dag.add_edge(
        input_tags=["incident_alert"],
        processor=lambda a: "MetricsTelemetry: Auth-service Pod OOMKilled 5 times in 10 mins. Memory > 99%.",
        output_tags=["metrics_analysis"],
        layer=0,
    )

    # Branch 2: Distributed Trace (Jaeger / OpenTelemetry)
    dag.add_edge(
        input_tags=["incident_alert"],
        processor=lambda a: "TraceTelemetry: Trace #a9f0: Gateway -> Auth-service HTTP 504 Gateway Timeout after 30s.",
        output_tags=["trace_analysis"],
        layer=0,
    )

    # Convergence: Root Cause & Action Items
    dag.add_edge(
        input_tags=["metrics_analysis", "trace_analysis"],
        processor=lambda t: f"RCA_Conclusion: Memory leak in JWT cache caused OOMKills leading to gateway 504s. Fix: Rollback v2.4.1. Details: {t[:50]}",
        output_tags=["incident_postmortem"],
        layer=1,
    )

    dag.add_content("P0 ALERT: Production API 5xx rate spiked to 38% at 22:30 UTC", tags=["incident_alert"], gen=0)

    dag.run(until_tag="incident_postmortem", max_steps=5)
    postmortem = dag.get_result("incident_postmortem")
    print(f"Incident Postmortem: {postmortem[:100]}...")
    assert "RCA_Conclusion" in postmortem
    print("Example 17 Passed Successfully!")


if __name__ == "__main__":
    main()
