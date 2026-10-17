"""Diverse DAG Architecture Patterns & Evaluation Suite for DAG Agent.

Demonstrates and evaluates 6 fundamentally different DAG topologies:
1. Multi-Branch Fan-Out / Fan-In Map-Reduce DAG (跨模型多路分支汇聚)
2. Low-Count Priority Puzzle Assembly & Two-Phase Clean-Context DAG (目标导向拼凑与两阶段纯净上下文回退)
3. Multi-Generation Iterative Refinement Pipeline DAG (多代际迭代审查与演化)
4. Cross-Lingual Semantic Fuzzy Tag Matching DAG (语义向量余弦相似度阈值匹配)
5. Dynamic Meta-DAG Self-Modification DAG (元 DAG 自然语言指令动态重构拓扑)
6. Transactional State Rollback & Infinite-Loop Guard DAG (事务原子回滚与环路上限熔断)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import (
    DAG,
    DefaultProcessor,
    MockLLMClient,
    OpenAICompatibleClient,
    Processor,
    TagMatcher,
    VersionA_MinimalDAG,
    VersionB_SemanticEmbeddingDAG,
    VersionC_CleanContextDefaultDAG,
    VersionD_MetaSelfModifyingDAG,
    VersionE_AsyncParallelDAG,
    VersionF_TransactionalReactiveDAG,
    VersionG_ProductionCompleteDAG,
    create_meta_dag,
    transform_dag_with_meta_dag,
)


def evaluate_dag_1_map_reduce(client=None):
    """DAG Pattern 1: Multi-Branch Fan-Out / Fan-In Map-Reduce."""
    print("\n" + "=" * 60)
    print("🌟 [DAG Pattern 1] Multi-Branch Fan-Out / Fan-In Map-Reduce")
    print("=" * 60)
    
    models_path = Path("models.json")
    dag = DAG(name="map_reduce_dag", models=models_path, llm_client=client)

    # Branch A (Layer 0, laptop-6.8)
    dag.add_edge(
        input_tags=["proposal"],
        processor={
            "model": "laptop-6.8",
            "prompt": "作为首席技术官，评估以下提案的技术可行性与潜在架构风险（30字内）：\n{input}",
        },
        output_tags=["tech_review"],
        layer=0,
    )

    # Branch B (Layer 0, sensenova-6.8)
    dag.add_edge(
        input_tags=["proposal"],
        processor={
            "model": "sensenova-6.8",
            "prompt": "作为商业顾问，评估以下提案的商业变现潜力与市场竞争力（30字内）：\n{input}",
        },
        output_tags=["biz_review"],
        layer=0,
    )

    # Convergence / Reducer Edge (Layer 1, N-in-1-out)
    dag.add_edge(
        input_tags=["tech_review", "biz_review"],
        processor={
            "model": "laptop-6.8",
            "prompt": "综合技术评估与商业评估，给出最终投资决策结论（30字内）：\n{input}",
        },
        output_tags=["executive_decision"],
        layer=1,
    )

    dag.add_content(
        contents="构建基于标签流驱动与元图自修改的高内聚分布式智能体系统",
        tags=["proposal"],
        gen=0,
    )

    # Step 1: Branch A
    s1 = dag.step()
    print(f"  Step 1: Action={s1.action}, Edge={s1.edge_id}, OutputTags={s1.produced_items[0].tags}, Gen={s1.produced_items[0].gen}")
    print(f"    Content: {s1.produced_items[0].contents.strip()[:80]}...")

    # Step 2: Branch B
    s2 = dag.step()
    print(f"  Step 2: Action={s2.action}, Edge={s2.edge_id}, OutputTags={s2.produced_items[0].tags}, Gen={s2.produced_items[0].gen}")
    print(f"    Content: {s2.produced_items[0].contents.strip()[:80]}...")

    # Step 3: Reducer (Convergence)
    s3 = dag.step()
    final_out = s3.produced_items[-1]
    print(f"  Step 3 (Convergence): Action={s3.action}, Edge={s3.edge_id}, OutputTags={final_out.tags}, Gen={final_out.gen}")
    print(f"    Final Decision: {final_out.contents.strip()[:100]}")

    assert "executive_decision" in final_out.tags
    assert final_out.gen == 3 or final_out.gen == 2
    print("  ✅ DAG Pattern 1 验证通过！")
    return dag


def evaluate_dag_2_puzzle_assembly_and_clean_context(client=None):
    """DAG Pattern 2: Target-Oriented Low-Count Puzzle Assembly & Clean Context Autonomous Routing."""
    print("\n" + "=" * 60)
    print("🌟 [DAG Pattern 2] Low-Count Puzzle Assembly & Clean-Context Routing")
    print("=" * 60)

    dag = VersionC_CleanContextDefaultDAG("puzzle_clean_dag", models="models.json", llm_client=client)

    # Edge 0: Low count (count=0)
    dag.add_edge(
        input_tags=["metrics", "error_log"],
        processor=lambda x: f"IncidentRCA({x})",
        output_tags=["root_cause_analysis"],
        layer=0,
        count=0,
    )

    # Edge 1: High count (count=5)
    dag.add_edge(
        input_tags=["user_stat", "survey_score"],
        processor=lambda x: f"UserReport({x})",
        output_tags=["user_satisfaction"],
        layer=0,
        count=5,
    )

    # Initial fragmented items (cannot directly trigger any edge above threshold)
    dag.add_content("CPU使用率 98%, 内存泄漏", tags=["system_metrics_raw"], gen=0)
    dag.add_content("NullPointerException at Worker.go:42", tags=["service_error_log_raw"], gen=0)

    # Step: default_processor checks edges, sees Edge 0 has count=0 < count=5,
    # pieces together metrics and error_log via N-in-1-out, and executes Edge 0!
    s1 = dag.step()
    print(f"  Step 1: Action={s1.action}, Mode={s1.action}, EdgeID={s1.edge_id}")
    print(f"    Produced: tags={s1.produced_items[0].tags}, gen={s1.produced_items[0].gen}")
    print(f"    Content: {s1.produced_items[0].contents[:80]}")

    assert s1.edge_id == "edge_0"
    assert s1.produced_items[-1].tags == ["root_cause_analysis"]
    print("  ✅ DAG Pattern 2 验证通过！")
    return dag


def evaluate_dag_3_iterative_chain(client=None):
    """DAG Pattern 3: Multi-Generation Code Review & Refine Pipeline."""
    print("\n" + "=" * 60)
    print("🌟 [DAG Pattern 3] Multi-Generation Code Review & Refine Pipeline")
    print("=" * 60)

    dag = DAG("iterative_dag", models="models.json", llm_client=client)

    # Edge 0 (Layer 0): Requirement -> Initial Code (gen 0 -> 1)
    dag.add_edge(
        input_tags=["requirement"],
        processor={
            "model": "laptop-6.8",
            "prompt": "编写 Python 函数实现快速排序：\n{input}",
        },
        output_tags=["code_draft"],
        layer=0,
    )

    # Edge 1 (Layer 1): Initial Code -> Code Review (gen 1 -> 2)
    dag.add_edge(
        input_tags=["code_draft"],
        processor={
            "model": "sensenova-6.8",
            "prompt": "审查以下代码的边界条件、时间复杂度及优化建议（30字内）：\n{input}",
        },
        output_tags=["code_review"],
        layer=1,
    )

    # Edge 2 (Layer 2): Convergence of Code + Review -> Final Code (gen 2 -> 3)
    dag.add_edge(
        input_tags=["code_draft", "code_review"],
        processor={
            "model": "laptop-6.8",
            "prompt": "结合原始代码和代码审查建议，输出优化后的最终代码（30字内）：\n{input}",
        },
        output_tags=["final_code"],
        layer=2,
    )

    dag.add_content("实现带基准选择的原地快速排序", tags=["requirement"], gen=0)

    # Run full pipeline
    dag.run(until_tag="final_code", max_steps=5)
    final_item = dag.contents[-1]
    print(f"  Pipeline Finished: Total steps={dag.step_count}, Final Tags={final_item.tags}, Final Gen={final_item.gen}")
    assert final_item.gen >= 3
    assert all(e.count == 1 for e in dag.edges)
    print("  ✅ DAG Pattern 3 验证通过！")
    return dag


def evaluate_dag_4_semantic_tag_matching():
    """DAG Pattern 4: Cross-Lingual Semantic Fuzzy Tag Matching."""
    print("\n" + "=" * 60)
    print("🌟 [DAG Pattern 4] Cross-Lingual Semantic Fuzzy Tag Matching")
    print("=" * 60)

    dag = VersionB_SemanticEmbeddingDAG("semantic_dag", similarity_threshold=0.60)

    dag.add_edge(
        input_tags=["financial_revenue"],
        processor=lambda x: f"NetProfitReport({x})",
        output_tags=["profit_margin_analysis"],
        layer=0,
    )

    # Content has tag 'sales_income', edge expects 'financial_revenue'
    # Semantic embedding similarity between 'sales_income' and 'financial_revenue' exceeds threshold 0.70
    dag.add_content("2026年Q3营业收入 $50M", tags=["sales_income"], gen=0)

    s1 = dag.step()
    print(f"  Action={s1.action}, Similarity={s1.similarity:.3f}, Output={s1.produced_items[0].tags}")
    assert s1.action == "edge"
    assert s1.produced_items[0].tags == ["profit_margin_analysis"]
    print("  ✅ DAG Pattern 4 验证通过！")
    return dag


def evaluate_dag_5_meta_dag_transformation(client=None):
    """DAG Pattern 5: Meta-DAG Dynamic Transformation & Reconfiguration."""
    print("\n" + "=" * 60)
    print("🌟 [DAG Pattern 5] Meta-DAG Dynamic Transformation")
    print("=" * 60)

    # Start with a base DAG
    base_dag = DAG("base_dag", models="models.json", llm_client=client)
    base_dag.add_content("原始系统告警数据", tags=["raw_alert"], gen=0)

    # Instruction to transform topology
    prompt = (
        "添加边: [raw_alert] -> (laptop-6.8: 分析告警级别) -> [alert_severity] layer=0; "
        "添加边: [alert_severity] -> (sensenova-6.8: 给出应急预案) -> [remediation_plan] layer=1"
    )
    print(f"  Modification Prompt: '{prompt}'")

    modified = transform_dag_with_meta_dag(base_dag, prompt, llm_client=client)
    print(f"  Modified DAG edges count: {len(modified.edges)} (Orig: {len(base_dag.edges)})")
    for i, e in enumerate(modified.edges):
        print(f"    Edge {i}: {e.input_tags} -> {e.output_tags} (layer={e.layer})")

    assert len(modified.edges) == 2
    assert modified.edges[0].output_tags == ["alert_severity"]
    assert modified.edges[1].output_tags == ["remediation_plan"]
    print("  ✅ DAG Pattern 5 验证通过！")
    return modified


def evaluate_dag_6_transaction_and_loop_guard():
    """DAG Pattern 6: Transactional State Rollback & Infinite-Loop Guard."""
    print("\n" + "=" * 60)
    print("🌟 [DAG Pattern 6] Transactional State Rollback & Infinite-Loop Guard")
    print("=" * 60)

    dag = VersionF_TransactionalReactiveDAG("safe_dag", max_edge_count=3)

    # Part A: Self-loop edge with max_edge_count=3
    dag.add_edge(
        input_tags=["ping"],
        processor=lambda x: f"Pong({x})",
        output_tags=["ping"],
        layer=0,
    )
    dag.add_content("StartPing", tags=["ping"], gen=0)

    dag.run(max_steps=10)
    print(f"  Self-loop terminated after edge count = {dag.edges[0].count} (Max Guard: 3)")
    assert dag.edges[0].count == 3

    # Part B: Transaction Rollback on Exception
    snap_len = len(dag.contents)
    try:
        with dag.transaction():
            dag.add_content("Temp Mutation", tags=["temp"], gen=99)
            dag.edges[0].count += 10
            raise RuntimeError("Simulation failure in pipeline step!")
    except RuntimeError as ex:
        print(f"  Caught expected error inside transaction: {ex}")

    assert len(dag.contents) == snap_len
    assert dag.edges[0].count == 3
    print("  Transaction successfully rolled back! Contents count and edge count restored.")
    print("  ✅ DAG Pattern 6 验证通过！")
    return dag


def run_all_evaluations():
    print("=" * 70)
    print("🚀 开始运行 DAG Agent 多范式综合测试评估套件...")
    print("=" * 70)

    # 1. 结构化离线/模拟评估（覆盖全部 6 种拓扑机制）
    mock_client = MockLLMClient()
    evaluate_dag_1_map_reduce(mock_client)
    evaluate_dag_2_puzzle_assembly_and_clean_context(mock_client)
    evaluate_dag_3_iterative_chain(mock_client)
    evaluate_dag_4_semantic_tag_matching()
    evaluate_dag_5_meta_dag_transformation(mock_client)
    evaluate_dag_6_transaction_and_loop_guard()

    print("\n" + "=" * 70)
    print("🎉 恭喜！全部 6 种不同的 DAG 架构范式均通过验证！符合预期！")
    print("=" * 70)


if __name__ == "__main__":
    run_all_evaluations()
