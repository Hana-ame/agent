"""示例 3：完整的“生成 DAG -> 运行 -> 验证好坏 -> 修改 DAG 内容 -> 增量重跑”闭环演示.

场景说明：
1. 初始阶段：通过分立句子生成一张财务核算 DAG：
   - 从 `原始财报` 经过 `提取营收与税率` 到 `营收数据`
   - 从 `原始财报` 经过 `提取成本明细` 到 `成本数据`
   - 从 `营收数据与成本数据` 经过 `计算净利润` 到 `利润报告`
2. 第一轮运行：
   - `提取成本明细` 边漏掉了研发费用（只算了 200 万，实际应为 300 万），且 `计算净利润` 忘记扣除 20% 税率；
   - 验证器（Verifier）检测到结果不对，给出反馈与修改建议；
3. 第二轮闭环自演进：
   - 闭环引擎根据验证结果，使用 Tool-Call 原语更新 `提取成本明细` 和 `计算净利润` 两条边；
   - **增量重跑亮点**：第一条分支 `原始财报 -> 营收数据` 已经算对且未被修改，第二轮自动跳过不重跑！
   - 最终产出正确的税后净利润 `1600.0`，验证通过，闭环结束！

运行方式：
    python3 examples/03_closed_loop_evolution.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v5 import ClosedLoopRunner, DAG, DAGToolKit, MergeMode, SmartMockLLM, VerifyResult


def main() -> None:
    sentences = [
        "从 原始财报 经过 提取营收与税率 到 营收数据",
        "从 原始财报 经过 提取成本明细(初版漏算) 到 成本数据",
        "从 营收数据与成本数据 经过 计算净利润(初版未扣税) 到 利润报告",
    ]

    mock_llm = SmartMockLLM(
        rules={
            "提取营收与税率": '{"revenue": 5000, "tax_rate": 0.2}',
            "提取成本明细(初版漏算)": '{"cost": 2000}',
            "提取成本明细(修正版含研发费用)": '{"cost": 3000}',
            "计算净利润(初版未扣税)": lambda p: json.dumps(
                {
                    "net_profit": json.loads(p.split("输入数据：")[-1])["revenue"]
                    - json.loads(p.split("输入数据：")[-1])["cost"]
                }
            ),
        }
    )

    # 定义结果好坏的验证器 (Verifier)
    def verify_profit_report(output: Any, dag: DAG) -> VerifyResult:
        if not isinstance(output, dict) or output.get("net_profit") != 1600.0:
            return VerifyResult(
                passed=False,
                score=0.4,
                feedback=(
                    f"结果校验不通过！期望税后净利润 net_profit=1600.0 (营收5000-总成本3000=2000，扣20%税后1600)，"
                    f"但当前 DAG 产出为: {output}"
                ),
            )
        return VerifyResult(passed=True, score=1.0, feedback="验证完全通过！税后净利润与审计标准一致。")

    # 定义当验证不通过时，如何通过 Tool-Call 原语更新 DAG 内容
    def optimize_dag_on_failure(
        toolkit: DAGToolKit, verify_res: VerifyResult, run_report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        mutations = []
        # 找出需要修正的边并调用 update_edge 原语
        for eid, edge in list(toolkit.dag.edges.items()):
            if "提取成本明细" in edge.action:
                res = toolkit.update_edge(
                    edge_id=eid,
                    action="提取成本明细(修正版含研发费用)",
                    prompt="提取成本明细(修正版含研发费用)：请包含生产成本与研发费用：\n{input}",
                )
                mutations.append({"action": "更新成本提取边 Prompt", "detail": res})
            elif "计算净利润" in edge.action:
                # 甚至可以直接把 LLM 边升级为精确的 Python 代码边！
                res = toolkit.update_edge(
                    edge_id=eid,
                    action="计算税后净利润(Python精确公式)",
                    code=(
                        "def transform(data):\n"
                        "    rev = data['revenue']\n"
                        "    cost = data['cost']\n"
                        "    tax = data.get('tax_rate', 0.2)\n"
                        "    return {'revenue': rev, 'cost': cost, 'net_profit': round((rev - cost) * (1 - tax), 2)}\n"
                    ),
                )
                mutations.append({"action": "将利润计算边升级为含税率公式的 Python 代码边", "detail": res})
        return mutations

    runner = ClosedLoopRunner(
        verifier_fn=verify_profit_report,
        optimizer_fn=optimize_dag_on_failure,
        llm=mock_llm,
        max_rounds=3,
    )

    # 一键启动：句子合并为 DAG -> 运行 -> 验证 -> 更新 DAG -> 增量重跑
    summary = runner.run_from_sentences_sync(
        sentences=sentences,
        initial_inputs={"原始财报": "Q4财报：总营收5000万，生产成本2000万，研发费用1000万，企业所得税率20%。"},
    )

    # 将虚拟边合并策略设为 JSON_MERGE 以便字典自动合并
    print("=== 合并器生成的虚拟边日志 ===")
    if summary.merger_result:
        for log in summary.merger_result.similarity_logs:
            print(" ", log)

    for rec in summary.history:
        print(f"\n==================== 第 {rec.round_index} 轮闭环执行 ====================")
        print("  实际执行的边        :", rec.executed_edges)
        print("  跳过(复用缓存)的顶点:", rec.skipped_cached_vertices)
        print("  当前 DAG 输出结果   :", rec.output)
        print(f"  验证结论 (通过={rec.verification.passed}, 得分={rec.verification.score}): {rec.verification.feedback}")
        if rec.mutations_applied:
            print("  触发的 DAG 原语更新 :")
            for m in rec.mutations_applied:
                print(f"    * {m['action']} (重置下游顶点: {m['detail'].get('invalidated_vertices')})")

    print("\n闭环最终是否收敛通过:", summary.converged, f"(共经历 {summary.total_rounds} 轮)")


if __name__ == "__main__":
    main()
