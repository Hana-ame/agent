"""Tests for Discontinuous Tag Flows and Mid-Execution Dynamic Content Injections.

Verifies:
1. Discontinuous / Broken Tag Flow & Semantic Gap Bridging via `default_processor`.
2. Multi-Tag Fragmented Puzzle Assembly across disparate items with different `gen` values.
3. Mid-Execution High-Priority Content Injection (Emergency Preemption via `layer=0`).
4. Mid-Execution Human-in-the-Loop Override / Side-Channel Data Injection.
5. Out-of-Order / Mixed-Gen Insertion and Generation Monotonicity Preservation.
6. Repeated Mid-Flight Interventions Stress Test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from dag_agent import (
    DAG,
    ContentItem,
    DefaultProcessor,
    MockLLMClient,
    ModelConfig,
    TagMatcher,
)


class TestDynamicAndDiscontinuousTagDAG:
    """Test suite validating resilience under broken tag flows and mid-flight content modifications."""

    def test_01_discontinuous_tag_bridging_via_default_processor(self):
        """Test discontinuous tag gap: input tag ['raw_survey'] does not match edge ['normalized_data'].

        `default_processor` must autonomously intervene, bridge the gap with clean context,
        and allow the pipeline to proceed to the target tag.
        """
        dag = DAG("discontinuous_tag_dag", llm_client=MockLLMClient())

        # Edge expects 'normalized_data', but we will inject 'raw_survey' (discontinuous tag)
        dag.add_edge(
            input_tags=["normalized_data"],
            processor="laptop-6.8: 分析标准化调研数据",
            output_tags=["analysis_report"],
            layer=0,
            default_id="edge_analyze",
        )
        dag.add_edge(
            input_tags=["analysis_report"],
            processor="sensenova-6.8: 生成决议",
            output_tags=["final_resolution"],
            layer=1,
            default_id="edge_decide",
        )

        # Initial discontinuous tag injection
        dag.add_content("2026年Q3用户问卷原始记录: 满意度85%，响应延迟偏高", tags=["raw_survey"], gen=0)

        # Step 1: No edge directly matches 'raw_survey', so default_processor routes autonomously
        step1 = dag.step()
        assert step1.progressed
        assert step1.action in ("default_targeted_assembly", "default_autonomous")
        assert len(dag.contents) >= 2

        # Step 2: Now the pipeline has bridged the tag gap and normal edges can execute
        step2 = dag.step()
        assert step2.progressed

        # Complete run until final_resolution
        dag.run(until_tag="final_resolution", max_steps=5)
        assert dag.has_tag("final_resolution")
        res = dag.get_result("final_resolution")
        assert res is not None
        assert len(res) > 0

    def test_02_fragmented_multi_tag_puzzle_assembly(self):
        """Test discontinuous multi-tag assembly: Three disjoint items with different tags and gens

        are pieced together by default_processor Mode 2a to satisfy an edge needing 3 input tags.
        """
        dag = DAG("fragmented_puzzle_dag", llm_client=MockLLMClient())

        # Target edge requires all three distinct tags
        dag.add_edge(
            input_tags=["user_id", "transaction_amount", "merchant_info"],
            processor="laptop-6.8: 整合三要素执行风控核验",
            output_tags=["fraud_audit_passed"],
            layer=0,
            default_id="edge_fraud_check",
        )

        # Three fragmented items injected at different generation stages
        dag.add_content("用户账号: UID_998182", tags=["user_id"], gen=0)
        dag.add_content("支付金额: 8900.00 元", tags=["transaction_amount"], gen=1)
        dag.add_content("收款商户: 海外虚拟主机服务商", tags=["merchant_info"], gen=2)

        # Initial direct match fails because no single item holds all 3 tags
        direct = dag.matcher.find_best_candidate(dag.edges, dag.contents, dag._executed_edge_sigs)
        assert direct is None

        # default_processor Mode 2a pieces them together!
        step = dag.step()
        assert step.action == "default_targeted_assembly"
        assert step.edge_id == "edge_fraud_check"

        # Verify that the synthesized item's gen = max(0, 1, 2) + 1 = 3
        # and edge output's gen = 4
        assert dag.has_tag("fraud_audit_passed")
        output_item = dag.get_latest("fraud_audit_passed")
        assert output_item is not None
        assert output_item.gen >= 3

    def test_03_mid_flight_high_priority_preemption(self):
        """Test mid-execution content injection with high priority (layer=0) preemption.

        A normal 2-step pipeline is running. After step 1, Python interrupts and injects
        an emergency alert. The emergency handler (layer=0) must preempt the continuation (layer=1).
        """
        dag = DAG("preemption_dag", llm_client=MockLLMClient())

        # Normal pipeline: step 1 (layer=1) -> step 2 (layer=1)
        dag.add_edge(
            input_tags=["user_task"],
            processor="laptop-6.8: 执行常规计算任务",
            output_tags=["task_in_progress"],
            layer=1,
            default_id="edge_normal_step1",
        )
        dag.add_edge(
            input_tags=["task_in_progress"],
            processor="laptop-6.8: 完成常规任务输出",
            output_tags=["task_done"],
            layer=1,
            default_id="edge_normal_step2",
        )

        # Emergency edge with highest priority layer=0
        dag.add_edge(
            input_tags=["emergency_interrupt"],
            processor="sensenova-6.8: 触发熔断保护并隔离环境",
            output_tags=["system_safe_shutdown"],
            layer=0,
            default_id="edge_emergency_kill",
        )

        # 1. Start normal pipeline
        dag.add_content("常规批处理作业 #404", tags=["user_task"], gen=0)
        s1 = dag.step()
        assert s1.edge_id == "edge_normal_step1"
        assert dag.has_tag("task_in_progress")

        # 2. Python interrupts mid-execution! Inject an emergency alert!
        print("Mid-flight: Injecting emergency interrupt into dag.contents...")
        dag.add_content(
            "ALERT: 检测到底层存储卷只读异常！立刻中止后续处理",
            tags=["emergency_interrupt"],
            gen=0,
        )

        # 3. Next step: Even though 'task_in_progress' is ready for 'edge_normal_step2' (layer=1),
        # the emergency edge has layer=0, so it MUST preempt and execute first!
        s2 = dag.step()
        assert s2.edge_id == "edge_emergency_kill"
        assert s2.edge_layer == 0
        assert dag.has_tag("system_safe_shutdown")

        # 4. Emergency was handled first; normal task remains paused or follows after
        assert s2.action == "edge"

    def test_04_mid_flight_human_in_the_loop_feedback_injection(self):
        """Test mid-execution human review intervention.

        Model generates draft -> Python pauses -> Human inserts critique -> Model incorporates feedback.
        """
        dag = DAG("hitl_intervention_dag", llm_client=MockLLMClient())

        dag.add_edge(
            input_tags=["source_prompt"],
            processor="laptop-6.8: 起草公文初稿",
            output_tags=["initial_draft"],
            layer=0,
            default_id="edge_draft",
        )
        # Edge requiring both the draft and the human feedback
        dag.add_edge(
            input_tags=["initial_draft", "human_review"],
            processor="sensenova-6.8: 吸收人工评审意见修订终稿",
            output_tags=["final_approved_document"],
            layer=1,
            default_id="edge_human_revision",
        )

        # Phase 1: Run draft edge
        dag.add_content("关于2026年度算力集群升级的立项请示", tags=["source_prompt"], gen=0)
        step1 = dag.step()
        assert step1.edge_id == "edge_draft"
        assert dag.has_tag("initial_draft")

        # Phase 2: Human in the loop intervenes via Python context
        with dag.contents.edit() as store:
            store.add(
                contents="人工专家审核意见：请补充GPU能效比对比表，并削减15%非核心采购预算。",
                tags=["human_review"],
                gen=store[-1].gen,
            )

        # Phase 3: Resume DAG execution
        dag.run(until_tag="final_approved_document", max_steps=5)
        assert dag.has_tag("final_approved_document")
        final_doc = dag.get_result("final_approved_document")
        assert final_doc is not None
        assert len(final_doc) > 0

    def test_05_out_of_order_and_mixed_gen_monotonicity(self):
        """Test that injecting older or mixed-gen contents mid-flight maintains generation monotonicity."""
        dag = DAG("gen_monotonicity_dag", llm_client=MockLLMClient())

        dag.add_edge(
            input_tags=["doc_alpha"],
            processor="laptop-6.8: Alpha -> Beta",
            output_tags=["doc_beta"],
            layer=0,
            default_id="edge_alpha_to_beta",
        )
        dag.add_edge(
            input_tags=["doc_beta", "meta_gamma"],
            processor="laptop-6.8: Beta + Gamma -> Delta",
            output_tags=["doc_delta"],
            layer=1,
            default_id="edge_beta_gamma_to_delta",
        )

        dag.add_content("Content Alpha", tags=["doc_alpha"], gen=5)
        dag.step()
        # Item Beta should have gen = 6
        item_b = dag.get_latest("doc_beta")
        assert item_b is not None
        assert item_b.gen == 6

        # Now mid-flight inject Item Gamma with older gen=1
        dag.add_content("Content Gamma with older gen", tags=["meta_gamma"], gen=1)

        # When edge_beta_gamma_to_delta combines Beta (gen=6) and Gamma (gen=1):
        # DefaultProcessor Mode 2a bundles them into a synthesized item (gen = max(6,1)+1 = 7),
        # and edge execution produces item_d with gen = 7 + 1 = 8.
        dag.step()
        item_d = dag.get_latest("doc_delta")
        assert item_d is not None
        # Strict monotonicity: final gen must be >= max(6, 1) + 1
        assert item_d.gen >= 7

    def test_06_stress_repeated_mid_flight_mutation_and_step(self):
        """Stress test: Repeated mid-flight content additions across 5 consecutive step cycles."""
        dag = DAG("stress_mutation_dag", llm_client=MockLLMClient())

        transitions = [
            ("user_inquiry", "intent_profile"),
            ("intent_profile", "query_solution"),
            ("query_solution", "reviewed_response"),
            ("reviewed_response", "delivery_package"),
            ("delivery_package", "archived_record"),
        ]

        for i, (in_tag, out_tag) in enumerate(transitions):
            dag.add_edge(
                input_tags=[in_tag],
                processor=f"laptop-6.8: {in_tag} -> {out_tag}",
                output_tags=[out_tag],
                layer=0,
                default_id=f"edge_step_{i}",
            )

        dag.add_content("启动流水线任务", tags=["user_inquiry"], gen=0)

        for step_idx in range(5):
            res = dag.step()
            assert res.progressed
            # Mid-flight: Python context injects telemetry audit annotation
            dag.add_content(
                f"Telemetry: step {step_idx} checkpoint recorded",
                tags=[f"telemetry_checkpoint_{step_idx}"],
                gen=step_idx + 1,
            )

        # Confirm all 5 phases completed sequentially and all 5 telemetry items were tracked
        for in_t, out_t in transitions:
            assert dag.has_tag(out_t)
        for i in range(5):
            assert dag.has_tag(f"telemetry_checkpoint_{i}")
        assert len(dag.history) == 5
