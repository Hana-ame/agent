"""Automated unit tests for Watcher DAG and Iterative Meta-DAG optimization loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dag_agent import (
    DAG,
    BenchmarkTestCase,
    EvaluationReport,
    MockLLMClient,
    PerformanceProfile,
    QualityProfile,
    WatcherOptimizerLoop,
    create_meta_dag,
    create_watcher_dag,
    transform_dag_with_meta_dag,
)


class TestWatcherMetaOptimizerLoop:
    """Test suite verifying Watcher DAG monitoring, reporting, and Meta-DAG iterative improvement."""

    def test_01_watcher_dag_creation_and_topology(self):
        """Test Watcher DAG instantiates with correct layers, tags, and callables."""
        watcher = create_watcher_dag(name="test_watcher")
        assert len(watcher.edges) == 3

        # Edge 0: Performance profiling
        assert watcher.edges[0].layer == 0
        assert "target_dag_input" in watcher.edges[0].input_tags
        assert "perf_data" in watcher.edges[0].output_tags

        # Edge 1: Quality evaluation
        assert watcher.edges[1].layer == 1
        assert "perf_data" in watcher.edges[1].input_tags
        assert "quality_verdict" in watcher.edges[1].output_tags

        # Edge 2: Diagnostic synthesis & Meta prompt formulation
        assert watcher.edges[2].layer == 2
        assert "quality_verdict" in watcher.edges[2].input_tags
        assert "result" in watcher.edges[2].output_tags

    def test_02_watcher_monitors_target_dag_and_emits_report(self):
        """Test Watcher DAG correctly executes target DAG on benchmarks and reports bottlenecks."""
        # Create an intentionally sub-optimal target DAG:
        # - Has a dead edge (edge_dead: never triggered)
        # - Missing direct edge from [user_request] to [report], so default_processor fallback will trigger
        target = DAG("suboptimal_target", llm_client=MockLLMClient())
        target.add_edge(
            input_tags=["dead_input_tag_xyz"],
            processor="laptop-6.8: 此边从不被触发",
            output_tags=["dead_output"],
            default_id="edge_dead",
            layer=0,
        )
        target.add_edge(
            input_tags=["intermediate_step"],
            processor="sensenova-6.8: 输出最终报告",
            output_tags=["report", "result"],
            default_id="edge_report",
            layer=1,
        )

        test_suite = [
            BenchmarkTestCase(
                input_text="请分析2026年第二季度算力成本波动",
                input_tags=["user_request"],
                expected_output_tag="result",
                required_keywords=["算力成本"],
            )
        ]

        watcher = create_watcher_dag(llm_client=MockLLMClient())
        payload = {
            "target_dag": target.to_dict(),
            "test_suite": [tc.to_dict() for tc in test_suite],
            "iteration": 0,
            "llm_type": "mock",
        }

        watcher.add_content(json.dumps(payload, ensure_ascii=False), tags=["target_dag_input"], gen=0)
        results = watcher.run(until_tag="result", max_steps=10)
        assert len(results) >= 3

        report_raw = watcher.get_result("result")
        assert report_raw is not None

        report_data = json.loads(report_raw)
        assert report_data["iteration"] == 0
        assert report_data["target_dag_name"] == "suboptimal_target"

        # Check performance diagnosis
        perf = report_data["performance"]
        assert "edge_dead" in perf["dead_edges"]
        assert perf["default_processor_fallbacks"] >= 1

        # Check that actionable meta_prompt was generated
        meta_prompt = report_data["meta_prompt"]
        assert "删除边 edge_dead" in meta_prompt
        assert len(report_data["diagnoses"]) >= 2

    def test_03_closed_loop_iterative_optimization_converges(self):
        """Test full closed-loop: Target -> Watcher -> Report -> Meta-DAG -> Mutated -> Re-watch -> Converged!"""
        # Step 1: Initial defective target DAG
        initial_target = DAG("data_pipeline_dag", llm_client=MockLLMClient())
        # Dead edge to be pruned
        initial_target.add_edge(
            input_tags=["legacy_unsupported_tag"],
            processor="laptop-6.8: 旧版废弃逻辑",
            output_tags=["legacy_out"],
            default_id="edge_legacy_dead",
            layer=0,
        )
        # Incomplete chain: produces [raw_triage], but test expects [final_clean_result]
        initial_target.add_edge(
            input_tags=["raw_query"],
            processor="laptop-6.8: 初始粗略清洗",
            output_tags=["raw_triage"],
            default_id="edge_triage",
            layer=1,
        )

        benchmark_suite = [
            BenchmarkTestCase(
                input_text="订单 #100293 用户要求取消并退款",
                input_tags=["raw_query"],
                expected_output_tag="raw_triage",
                required_keywords=["订单", "退款"],
                max_latency_ms=200.0,
            )
        ]

        optimizer = WatcherOptimizerLoop(
            target_dag=initial_target,
            test_suite=benchmark_suite,
            max_iterations=3,
            target_quality_score=0.85,
            target_max_fallbacks=0,
            llm_client=MockLLMClient(),
        )

        outcome = optimizer.run()

        # Verify optimization progression
        assert outcome["total_iterations"] >= 1
        history = outcome["history"]
        assert len(history) >= 1

        # Iteration 0 detected dead edge and suggested pruning
        iter0 = history[0]
        assert "edge_legacy_dead" in iter0.performance.dead_edges
        assert "删除边 edge_legacy_dead" in iter0.meta_prompt

        # Final DAG should have pruned the dead edge!
        final_dag = outcome["final_dag"]
        final_edge_ids = [e.id for e in final_dag.edges]
        assert "edge_legacy_dead" not in final_edge_ids

        # Final quality and performance should show convergence / improvement
        final_rep = history[-1]
        assert len(final_rep.performance.dead_edges) == 0
