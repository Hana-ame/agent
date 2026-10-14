"""VEA V5 闭环自演进引擎模块 (Closed-Loop Engine).

本模块将整个闭环串联为一个自动化闭环系统：
1. **阶段 1（生成/合并 DAG）**：输入多句“从 x1 经过 y1 到 z1”分立自然语言描述，
   通过“第一个 VEA DAG（合并器 `SentenceToDAGMerger`）”自动提取 `V-E-V` 并用虚拟边 `VirtualEdge` 连接相似顶点，生成初始业务 DAG；
2. **阶段 2（执行与结果验证）**：在 `Executor` 下运行业务 DAG 产生结果，交给 `verifier_fn` 判断结果好坏（返回 `VerifyResult`）；
3. **阶段 3（闭环反馈更新 DAG）**：如果验证未通过（`passed=False`），闭环优化器（`optimizer_fn` 或 LLM Revisor）
   根据验证失败原因调用 `DAGToolKit` 的原语（如 `update_edge` 修改边逻辑、`add_vev_step` 增补新节点/边、`link_similar_vertices` 增补虚拟边）更新 DAG；
4. **阶段 4（增量重跑直至通过）**：自动保留未受影响的上游顶点结果，仅增量重跑修改过的下游分支，直到验证通过或达到最大轮次。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from vea_v5.dag import DAG
from vea_v5.llm import LLMProtocol, SmartMockLLM
from vea_v5.merger import MergerResult, SentenceToDAGMerger
from vea_v5.toolkit import DAGToolKit


@dataclass
class VerifyResult:
    """验证器对 DAG 运行结果的评估结论。"""

    passed: bool
    score: float = 0.0
    feedback: str = ""
    suggestion: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "feedback": self.feedback,
            "suggestion": self.suggestion,
        }


@dataclass
class LoopIterationRecord:
    """闭环中每一轮迭代的完整记录。"""

    round_index: int
    executed_edges: List[str]
    skipped_cached_vertices: List[str]
    output: Any
    verification: VerifyResult
    mutations_applied: List[Dict[str, Any]] = field(default_factory=list)
    dag_ascii_snapshot: str = ""


@dataclass
class ClosedLoopSummary:
    """整个闭环运行完成后的汇总报告。"""

    converged: bool
    total_rounds: int
    final_output: Any
    final_dag: DAG
    merger_result: Optional[MergerResult] = None
    history: List[LoopIterationRecord] = field(default_factory=list)


class ClosedLoopRunner:
    """VEA V5 闭环自演进运行器：从分立句子合并 DAG -> 运行 -> 验证好坏 -> 动态改图 -> 增量重跑。"""

    def __init__(
        self,
        verifier_fn: Callable[[Any, DAG], Union[VerifyResult, Dict[str, Any]]],
        optimizer_fn: Optional[Callable[[DAGToolKit, VerifyResult, Dict[str, Any]], List[Dict[str, Any]]]] = None,
        llm: Optional[LLMProtocol] = None,
        max_rounds: int = 5,
        similarity_threshold: float = 0.45,
    ) -> None:
        """初始化闭环运行器。

        Args:
            verifier_fn: 验证函数 `(final_output, dag) -> VerifyResult`，判断当前 DAG 运行结果的好坏并给出反馈。
            optimizer_fn: 闭环改图函数 `(toolkit, verify_result, run_report) -> List[mutation_logs]`。
                          当验证未通过时被调用，可使用 `toolkit.update_edge` / `toolkit.add_vev_step` 等原语更新 DAG。
                          若未提供 `optimizer_fn`，但 `VerifyResult.suggestion` 中带有 `tool_calls` 列表，
                          引擎会自动按 `suggestion["tool_calls"]` 分发执行原语改图！
            llm: 驱动边推理的大语言模型实例。
            max_rounds: 最大闭环迭代轮数。
            similarity_threshold: 合并分立句子时的相似顶点虚拟边阈值。
        """
        self.verifier_fn = verifier_fn
        self.optimizer_fn = optimizer_fn
        self.llm = llm or SmartMockLLM()
        self.max_rounds = max(1, max_rounds)
        self.similarity_threshold = similarity_threshold

    def _normalize_verify(self, raw: Union[VerifyResult, Dict[str, Any], bool]) -> VerifyResult:
        if isinstance(raw, VerifyResult):
            return raw
        if isinstance(raw, bool):
            return VerifyResult(passed=raw, score=1.0 if raw else 0.0)
        return VerifyResult(
            passed=bool(raw.get("passed", False)),
            score=float(raw.get("score", 1.0 if raw.get("passed") else 0.0)),
            feedback=str(raw.get("feedback") or raw.get("reason") or ""),
            suggestion=raw.get("suggestion"),
        )

    async def run_from_sentences(
        self,
        sentences: Union[str, Sequence[str]],
        initial_inputs: Dict[str, Any],
        edge_bindings: Optional[Dict[str, Any]] = None,
    ) -> ClosedLoopSummary:
        """一步到位：先运行“第一个 VEA DAG（合并器）”把分立句子合并成 DAG，再进入验证-改图闭环！"""
        merger = SentenceToDAGMerger(
            similarity_threshold=self.similarity_threshold,
            edge_bindings=edge_bindings,
        )
        merger_res = merger.merge(sentences)
        summary = await self.run_on_dag(merger_res.compiled_dag, initial_inputs=initial_inputs)
        summary.merger_result = merger_res
        return summary

    async def run_on_dag(
        self,
        dag: DAG,
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> ClosedLoopSummary:
        """在给定 DAG 上执行“运行 -> 验证 -> 修改 DAG -> 增量重跑”闭环。"""
        toolkit = DAGToolKit(
            dag=dag,
            llm=self.llm,
            verifier_fn=lambda out, g: self._normalize_verify(self.verifier_fn(out, g)).to_dict(),
        )

        if initial_inputs:
            for v_name, val in initial_inputs.items():
                toolkit.set_vertex_input(v_name, val)

        history: List[LoopIterationRecord] = []
        converged = False
        final_output: Any = None

        for r_idx in range(1, self.max_rounds + 1):
            run_res = await toolkit.run_and_verify()
            v_res = self._normalize_verify(run_res["verification"])
            final_output = run_res["final_output"]

            rec = LoopIterationRecord(
                round_index=r_idx,
                executed_edges=list(run_res["executed_edges"]),
                skipped_cached_vertices=list(run_res["skipped_cached_vertices"]),
                output=final_output,
                verification=v_res,
                dag_ascii_snapshot=dag.visualize_ascii(),
            )
            history.append(rec)

            # 如果验证通过，闭环成功收敛！
            if v_res.passed:
                converged = True
                break

            # 如果已经是最后一轮，不再改图
            if r_idx >= self.max_rounds:
                break

            # 根据验证结果好坏，调用 optimizer_fn 或 suggestion 中的 tool_calls 更新 DAG
            mutations: List[Dict[str, Any]] = []
            if self.optimizer_fn is not None:
                mutations = self.optimizer_fn(toolkit, v_res, run_res) or []
            elif v_res.suggestion and "tool_calls" in v_res.suggestion:
                for tc in v_res.suggestion["tool_calls"]:
                    t_out = await toolkit.call_tool(tc["name"], tc.get("arguments", {}))
                    mutations.append({"tool": tc["name"], "args": tc.get("arguments", {}), "result": t_out})

            rec.mutations_applied = mutations
            if not mutations:
                # 没有任何改图动作，提前结束防止空转
                break

        return ClosedLoopSummary(
            converged=converged,
            total_rounds=len(history),
            final_output=final_output,
            final_dag=dag,
            history=history,
        )

    def run_from_sentences_sync(
        self,
        sentences: Union[str, Sequence[str]],
        initial_inputs: Dict[str, Any],
        edge_bindings: Optional[Dict[str, Any]] = None,
    ) -> ClosedLoopSummary:
        """同步版本的 `run_from_sentences`。"""
        return asyncio.run(self.run_from_sentences(sentences, initial_inputs, edge_bindings))

    def run_on_dag_sync(
        self,
        dag: DAG,
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> ClosedLoopSummary:
        """同步版本的 `run_on_dag`。"""
        return asyncio.run(self.run_on_dag(dag, initial_inputs))
