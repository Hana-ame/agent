"""Showcase script demonstrating Versions A through G of DAG Agent Framework."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import (
    MockLLMClient,
    Processor,
    VersionA_MinimalDAG,
    VersionB_SemanticEmbeddingDAG,
    VersionC_CleanContextDefaultDAG,
    VersionD_MetaSelfModifyingDAG,
    VersionE_AsyncParallelDAG,
    VersionF_TransactionalReactiveDAG,
    VersionG_ProductionCompleteDAG,
)


def demo_all_versions() -> None:
    print("=== Version A: 极简精确标签 + Layer/Gen 优先级 + Count 计数 ===")
    dag_a = VersionA_MinimalDAG("demo_a")
    dag_a.add_edge(["query"], lambda x: f"Step1({x})", ["mid"], layer=0)
    dag_a.add_edge(["mid"], lambda x: f"Final({x})", ["result"], layer=1)
    dag_a.add_content("Hello DAG", tags=["query"], gen=0)
    dag_a.run(until_tag="result")
    print("  Result:", dag_a.get_result("result"), "| Edge counts:", [e.count for e in dag_a.edges])

    print("\n=== Version B: 语义向量 Embedding 阈值匹配 ===")
    dag_b = VersionB_SemanticEmbeddingDAG("demo_b", similarity_threshold=0.65)
    dag_b.add_edge(["revenue_report"], lambda x: f"Analyzed({x})", ["result"], layer=0)
    dag_b.add_content("2026 Sales Income $50M", tags=["sales_income_report"], gen=0)
    dag_b.step()
    print("  Semantic match result:", dag_b.get_result("result"))

    print("\n=== Version C: 干净上下文 DefaultProcessor (N-in-1-out 凑边 + 自主路由) ===")
    mock_c = MockLLMClient(
        rules={
            "selected_indices": '{"selected_indices": [0, 2], "output_tags": ["result"]}',
            "请将以下输入信息整合": "Merged [Item0 + Item2] in 100% Clean Context",
        }
    )
    dag_c = VersionC_CleanContextDefaultDAG("demo_c", llm_client=mock_c)
    dag_c.add_content("Part A: User Profile", tags=["profile"], gen=0)
    dag_c.add_content("Irrelevant Distractor Log", tags=["noise"], gen=0)
    dag_c.add_content("Part B: Purchase History", tags=["orders"], gen=1)
    dag_c.step()
    print("  Autonomous output:", dag_c.get_result("result"))
    print("  Clean context verified:", dag_c.get_clean_output_calls()[0].is_clean_context)

    print("\n=== Version D: 元 DAG (Meta-DAG) 根据 Prompt 修改目标 DAG ===")
    dag_d = VersionD_MetaSelfModifyingDAG("demo_d")
    dag_d.add_edge(["raw"], lambda x: f"Cleaned({x})", ["cleaned"], layer=0)
    modified_d = dag_d.modify_with_prompt(
        "添加边: [cleaned] -> (default: 生成摘要报告: {contents}) -> [result] layer=1;\n"
        "添加内容: tags=[raw] contents=Quarterly Data"
    )
    modified_d.run(until_tag="result")
    print("  Modified DAG edges:", len(modified_d.edges), "| Result:", modified_d.get_result("result"))

    print("\n=== Version E: 异步并发同层 Frontier 批处理 ===")
    dag_e = VersionE_AsyncParallelDAG("demo_e", max_concurrency=3)
    dag_e.add_edge(["doc"], lambda x: f"A({x})", ["part_a"], layer=0)
    dag_e.add_edge(["doc"], lambda x: f"B({x})", ["part_b"], layer=0)
    dag_e.add_edge(["part_a", "part_b"], lambda x: f"Combined({x})", ["result"], layer=1)
    dag_e.add_content("Source Document", tags=["doc"], gen=0)
    asyncio.run(dag_e.step_parallel_frontier())
    asyncio.run(dag_e.step_parallel_frontier())
    print("  Async parallel result:", dag_e.get_result("result"))

    print("\n=== Version F: 事务快照回滚 + 循环次数守卫 ===")
    dag_f = VersionF_TransactionalReactiveDAG("demo_f", max_edge_count=2)
    dag_f.add_edge(["loop"], lambda x: f"{x}!", ["loop"], layer=0)
    dag_f.add_content("go", tags=["loop"], gen=0)
    snap = dag_f.checkpoint("before_run")
    dag_f.step()
    dag_f.rollback(snap)
    print("  After rollback contents len:", len(dag_f.contents), "| edge count:", dag_f.edges[0].count)

    print("\n=== Version G: 终极生产版 (Trace + Mermaid 导出) ===")
    dag_g = VersionG_ProductionCompleteDAG("demo_g")
    dag_g.add_edge(["input"], lambda x: f"Done({x})", ["result"], layer=0)
    dag_g.add_content("Production Payload", tags=["input"], gen=0)
    dag_g.step()
    print("  Trace:", dag_g.export_trace_summary())
    print("  Mermaid:\n" + dag_g.to_mermaid())


if __name__ == "__main__":
    demo_all_versions()
