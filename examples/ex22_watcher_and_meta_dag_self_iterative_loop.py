"""Example 22: Watcher DAG & Meta-DAG Closed-Loop Monitoring and Iterative Optimization.

Demonstrates:
1. An E-Commerce Customer Service DAG with defects:
   - Dead edge (`edge_legacy_fax`) consuming topology
   - Disconnected tag flow forcing `default_processor` fallback latency
   - Incomplete quality reachability (fails to reach `[final_response]`)
2. A specialized Watcher DAG that observes:
   - 效果 (Quality / Tag completion / Keyword accuracy)
   - 效率速度 (Latency / Step count / DefaultProcessor fallback rate / Dead edges)
3. Delivering the structured `EvaluationReport` to Meta-DAG.
4. Meta-DAG automatically patching the target DAG based on the diagnosis.
5. Iterative convergence loop: Watch -> Report -> Mutate -> Re-watch -> Converged!
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# 与其余 21 个示例保持一致：脚本被直接执行时 `sys.path[0]` 是 examples/，
# 仓库根目录不在路径上，`from dag_agent import ...` 会 ModuleNotFoundError
# （dag_agent 并未安装到 site-packages，能 import 只是因为从根目录运行时
#  cwd 恰好在 sys.path 上）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dag_agent import (  # noqa: E402
    DAG,
    BenchmarkTestCase,
    EvaluationReport,
    MockLLMClient,
    WatcherOptimizerLoop,
    create_meta_dag,
    create_watcher_dag,
)


def build_initial_defective_dag() -> DAG:
    """Build an initial E-Commerce Customer Service DAG with intentional efficiency and quality defects."""
    dag = DAG("ecommerce_service_agent", models="models.json", tools="tools.json", llm_client=MockLLMClient())

    # Defect 1: Dead Edge that never fires
    dag.add_edge(
        input_tags=["legacy_fax_input"],
        processor="laptop-6.8: 传统传真解析",
        output_tags=["fax_data"],
        layer=0,
        default_id="edge_legacy_fax",
    )

    # Defect 2: Incomplete pipeline that only produces raw triage but lacks final customer response
    dag.add_edge(
        input_tags=["user_query"],
        processor="laptop-6.8: 客服意图分类与提取",
        output_tags=["triage_category"],
        layer=1,
        default_id="edge_intent_triage",
    )

    return dag


def main():
    print("=" * 80)
    print("🚀 启动 [Example 22]：Watcher DAG 监视与 Meta-DAG 闭环自迭代优化系统")
    print("=" * 80)

    # 1. 初始化缺陷目标 DAG
    target_dag = build_initial_defective_dag()
    print(f"\n📦 初始目标 DAG: '{target_dag.name}', 边数量: {len(target_dag.edges)}")
    for e in target_dag.edges:
        print(f"   - Edge [{e.id}]: {e.input_tags} -> {e.output_tags} (layer={e.layer})")

    # 2. 准备业务基准测试集 (涵盖效果与时效要求)
    benchmark_suite = [
        BenchmarkTestCase(
            input_text="用户发起退款申请：订单 #2026889 商品尺码不合，要求退货退款并承担运费险",
            input_tags=["user_query"],
            expected_output_tag="final_response",
            required_keywords=["退款", "运费险"],
            max_latency_ms=150.0,
            max_steps=3,
            description="七天无理由退款与运费险核销",
        ),
        BenchmarkTestCase(
            input_text="VIP客户咨询：购买高端服务器遇到网络抖动，申请专线加急技术支持",
            input_tags=["user_query"],
            expected_output_tag="final_response",
            required_keywords=["VIP", "加急"],
            max_latency_ms=150.0,
            max_steps=3,
            description="VIP加急专线技术响应",
        ),
    ]

    print(f"\n🎯 挂载业务测试集: 共 {len(benchmark_suite)} 个基准用例")
    for i, tc in enumerate(benchmark_suite):
        print(f"   [{i + 1}] 预期标签: '{tc.expected_output_tag}', 考核关键词: {tc.required_keywords}")

    # 3. 构造专用 Watcher DAG 与 Meta-DAG 闭环迭代器
    watcher_dag = create_watcher_dag(name="service_quality_watcher", llm_client=MockLLMClient())
    meta_dag = create_meta_dag(llm_client=MockLLMClient())

    optimizer = WatcherOptimizerLoop(
        target_dag=target_dag,
        test_suite=benchmark_suite,
        watcher_dag=watcher_dag,
        meta_dag=meta_dag,
        max_iterations=3,
        target_quality_score=0.85,
        target_max_fallbacks=0,
        target_max_latency_ms=100.0,
        llm_client=MockLLMClient(),
    )

    print("\n🔄 开始进入 [Watch -> Diagnostic Report -> Meta-DAG Mutate -> Re-watch] 迭代循环...")
    loop_result = optimizer.run()

    # 4. 输出各轮次详细报告
    for report in optimizer.history:
        print("\n" + "-" * 70)
        print(report.to_markdown())

    # 5. 输出闭环优化前后对比总结
    init_rep = optimizer.history[0]
    final_rep = optimizer.history[-1]
    final_dag = loop_result["final_dag"]

    print("\n" + "=" * 80)
    print("📈 闭环自迭代优化效果对比看板 (Before vs After):")
    print("=" * 80)
    print(f"  {'指标名称':<25} | {'初始状态 (v0)':<20} | {'优化后状态 (vFinal)':<20}")
    print(f"  {'-' * 25}-+-{'-' * 20}-+-{'-' * 20}")
    print(f"  {'综合质量评分 (Quality)':<25} | {init_rep.quality.quality_score * 100:.1f}%{'':<15} | {final_rep.quality.quality_score * 100:.1f}%")
    print(f"  {'目标标签准出达成率':<25} | {init_rep.quality.success_rate * 100:.1f}%{'':<15} | {final_rep.quality.success_rate * 100:.1f}%")
    print(f"  {'未触发死边数量 (Dead Edges)':<25} | {len(init_rep.performance.dead_edges)} 条{'':<17} | {len(final_rep.performance.dead_edges)} 条")
    print(f"  {'DefaultProcessor 兜底次数':<25} | {init_rep.performance.default_processor_fallbacks} 次{'':<17} | {final_rep.performance.default_processor_fallbacks} 次")
    print(f"  {'平均执行延迟 (Latency)':<25} | {init_rep.performance.avg_latency_ms:.2f} ms{'':<14} | {final_rep.performance.avg_latency_ms:.2f} ms")
    print(f"  {'收敛状态 (Convergence)':<25} | {'未收敛':<20} | {'✅ 已收敛 (CONVERGED)':<20}")

    print("\n🛠️ 最终目标 DAG 拓扑架构:")
    for e in final_dag.edges:
        print(f"   - Edge [{e.id}]: {e.input_tags} -> {e.output_tags} (layer={e.layer})")

    # 断言保证
    assert "edge_legacy_fax" not in [e.id for e in final_dag.edges], "死边必须被修剪"
    assert any("final_response" in e.output_tags for e in final_dag.edges), "必须补全目标响应边"
    assert final_rep.is_converged or final_rep.quality.quality_score >= 0.85

    print("\n" + "=" * 80)
    print("🎉 示例 22 成功运行完成！Watcher 监视与元 DAG 自适应闭环迭代验证无误！")
    print("=" * 80)


if __name__ == "__main__":
    main()
