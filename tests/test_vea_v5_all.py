"""VEA V5 完整自动化单元测试与闭环验证套件.

运行方式：
    pytest tests/test_vea_v5_all.py -v
    或直接运行：
    python3 tests/test_vea_v5_all.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v5 import (
    ClosedLoopRunner,
    CycleError,
    DAG,
    DAGToolKit,
    Executor,
    MergeMode,
    SentenceToDAGMerger,
    SmartMockLLM,
    VerifyResult,
    parse_vev_from_text,
)


def test_1_basic_vev_chain_and_auto_roots_sinks() -> None:
    """测试 1：一句话添加 V-E-V 步骤，自动推导起点与终点并执行。"""
    dag = DAG("test_basic")
    dag.add_step("x1", lambda s: s.strip().upper(), "z1", edge_id="e1")
    dag.add_step("z1", lambda s: f"HELLO {s}!", "z2", edge_id="e2")

    assert dag.roots() == ["x1"]
    assert dag.sinks() == ["z2"]

    report = Executor(dag).run_sync({"x1": "world"})
    assert report.success is True
    assert report.executed_edges == ["e1", "e2"]
    assert report.final_output == "HELLO WORLD!"


def test_2_cycle_detection_and_auto_rollback() -> None:
    """测试 2：当插入成环边时自动回滚，绝不污染已有 DAG。"""
    dag = DAG("test_cycle")
    dag.add_step("A", lambda x: x, "B", edge_id="e_ab")
    dag.add_step("B", lambda x: x, "C", edge_id="e_bc")

    try:
        dag.add_step("C", lambda x: x, "A", edge_id="e_ca")
        assert False, "应当抛出 CycleError"
    except CycleError:
        pass

    # 确认非法边已被自动回滚，原图依然正常可运行
    assert "e_ca" not in dag.edges
    rep = Executor(dag).run_sync({"A": "ok"})
    assert rep.success is True
    assert rep.final_output == "ok"


def test_3_sentence_parser_and_first_vea_merger_dag() -> None:
    """测试 3：测试‘第一个 VEA DAG（合并器）’将分立句子拆为 V-E-V 并用虚拟边连接相似顶点。"""
    text = """
    从 x1 经过 翻倍处理 到 中间结果A
    从 中间结果A_相似节点 经过 加十处理 到 z2
    """
    triplets = parse_vev_from_text(text)
    assert len(triplets) == 2
    assert triplets[0].source_name == "x1"
    assert triplets[0].target_name == "中间结果A"
    assert triplets[1].source_name == "中间结果A_相似节点"
    assert triplets[1].target_name == "z2"

    merger = SentenceToDAGMerger(
        similarity_threshold=0.45,
        edge_bindings={
            "翻倍处理": lambda x: int(x) * 2,
            "加十处理": lambda x: int(x) + 10,
        },
    )
    res = merger.merge(text)
    # 应当自动创建一条从 '中间结果A' 到 '中间结果A_相似节点' 的虚拟边
    assert len(res.virtual_edges) == 1
    ve = res.virtual_edges[0]
    assert ve.source == "中间结果A"
    assert ve.target == "中间结果A_相似节点"
    assert ve.is_virtual is True

    # 运行合并后的 DAG：5 * 2 = 10 -> (通过虚拟边桥接) -> 10 + 10 = 20
    rep = Executor(res.compiled_dag).run_sync({"x1": 5})
    assert rep.success is True
    assert rep.final_output == 20


def test_4_incremental_closed_loop_evolution() -> None:
    """测试 4：完整闭环测试（生成 DAG -> 运行 -> 验证失败 -> 修改边 -> 仅增量重跑下游分支直至通过）。"""
    call_counter = {"e1_runs": 0, "e2_runs": 0}

    def step1_expensive(x: int) -> int:
        call_counter["e1_runs"] += 1
        return x * 10

    def step2_buggy(x: int) -> int:
        call_counter["e2_runs"] += 1
        return x + 1  # 错误逻辑：期望是 + 5

    dag = DAG("closed_loop_dag")
    dag.add_step("x1", step1_expensive, "z1", edge_id="e1")
    dag.add_step("z1", step2_buggy, "z2", edge_id="e2")

    def verifier(output, g):
        return VerifyResult(
            passed=(output == 35),
            score=1.0 if output == 35 else 0.2,
            feedback=f"期望 35，实际 {output}",
            suggestion={
                "tool_calls": [
                    {
                        "name": "update_edge",
                        "arguments": {
                            "edge_id": "e2",
                            "code": "def transform(x): return int(x) + 5",
                        },
                    }
                ]
            },
        )

    runner = ClosedLoopRunner(verifier_fn=verifier, max_rounds=3)
    summary = runner.run_on_dag_sync(dag, initial_inputs={"x1": 3})

    assert summary.converged is True
    assert summary.total_rounds == 2
    assert summary.final_output == 35
    # 核心断言：上游昂贵的 e1 只在第一轮跑了 1 次，第二轮被缓存跳过！
    assert call_counter["e1_runs"] == 1
    assert "z1" in summary.history[1].skipped_cached_vertices
    assert summary.history[1].executed_edges == ["e2"]


if __name__ == "__main__":
    test_1_basic_vev_chain_and_auto_roots_sinks()
    test_2_cycle_detection_and_auto_rollback()
    test_3_sentence_parser_and_first_vea_merger_dag()
    test_4_incremental_closed_loop_evolution()
    print("ALL 4 VEA V5 TEST SUITES PASSED SUCCESSFULLY!")
