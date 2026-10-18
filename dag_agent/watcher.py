"""Watcher DAG and Iterative Optimization Loop (`dag_agent.watcher`).

Implements a dedicated Watcher DAG that monitors another target DAG's:
1. 效果 (Effectiveness / Quality / Content correctness / Tag reachability)
2. 效率与速度 (Efficiency / Latency / Step count / Edge usage / DefaultProcessor fallback rate)

The Watcher DAG outputs an `EvaluationReport` with actionable diagnoses and optimization
directives, which is fed to the Meta-DAG (`meta_dag`) to mutate and improve the target DAG
in an automated closed-loop iteration.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from dag_agent.engine import DAG, StepResult
from dag_agent.llm import LLMClientProtocol, MockLLMClient
from dag_agent.meta import create_meta_dag, transform_dag_with_meta_dag
from dag_agent.schema import ContentItem, ModelRegistry, ToolRegistry


@dataclass
class BenchmarkTestCase:
    """A benchmark test case used by the Watcher DAG to evaluate a target DAG."""

    input_text: str
    input_tags: List[str]
    expected_output_tag: str = "result"
    required_keywords: List[str] = field(default_factory=list)
    max_latency_ms: float = 500.0
    max_steps: int = 5
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkTestCase":
        raw = dict(data)
        return cls(
            input_text=str(raw.get("input_text", raw.get("input", ""))),
            input_tags=list(raw.get("input_tags", ["input"])),
            expected_output_tag=str(raw.get("expected_output_tag", "result")),
            required_keywords=list(raw.get("required_keywords", [])),
            max_latency_ms=float(raw.get("max_latency_ms", 500.0)),
            max_steps=int(raw.get("max_steps", 5)),
            description=str(raw.get("description", "")),
        )


@dataclass
class PerformanceProfile:
    """Efficiency and speed metrics recorded by the Watcher DAG."""

    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    total_steps: int = 0
    avg_steps_per_case: float = 0.0
    edge_firing_counts: Dict[str, int] = field(default_factory=dict)
    dead_edges: List[str] = field(default_factory=list)
    default_processor_fallbacks: int = 0
    unmatched_lingering_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityProfile:
    """Effectiveness and quality metrics recorded by the Watcher DAG."""

    success_rate: float = 0.0  # 0.0 to 1.0 (fraction of test cases reaching expected tag)
    keyword_coverage_rate: float = 0.0  # 0.0 to 1.0
    quality_score: float = 0.0  # Combined score 0.0 to 1.0
    failed_cases: List[str] = field(default_factory=list)
    sample_outputs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Comprehensive diagnostic and optimization report produced by the Watcher DAG."""

    iteration: int
    target_dag_name: str
    performance: PerformanceProfile
    quality: QualityProfile
    overall_score: float
    diagnoses: List[str]
    meta_prompt: str
    is_converged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "target_dag_name": self.target_dag_name,
            "performance": self.performance.to_dict(),
            "quality": self.quality.to_dict(),
            "overall_score": round(self.overall_score, 4),
            "diagnoses": self.diagnoses,
            "meta_prompt": self.meta_prompt,
            "is_converged": self.is_converged,
        }

    def to_markdown(self) -> str:
        status_badge = "✅ CONVERGED" if self.is_converged else "⚠️ REQUIRES_OPTIMIZATION"
        diag_lines = "\n".join(f"- {d}" for d in self.diagnoses) or "- 无显著瓶颈"
        return f"""### 📊 Watcher DAG 评估与诊断报告 (Iteration {self.iteration}) [{status_badge}]

- **目标 DAG**: `{self.target_dag_name}`
- **综合评分**: `{self.overall_score:.2f} / 1.00`
- **效果 (Quality)**:
  - 成功率: `{self.quality.success_rate * 100:.1f}%`
  - 关键词覆盖率: `{self.quality.keyword_coverage_rate * 100:.1f}%`
- **效率 (Efficiency & Speed)**:
  - 平均延迟: `{self.performance.avg_latency_ms:.2f} ms` (总耗时: `{self.performance.total_latency_ms:.2f} ms`)
  - 平均单例步数: `{self.performance.avg_steps_per_case:.1f}` (总步数: `{self.performance.total_steps}`)
  - DefaultProcessor 兜底干预次数: `{self.performance.default_processor_fallbacks}` 次
  - 未触发死边 (Dead Edges): `{', '.join(self.performance.dead_edges) or 'None'}`

#### 🔍 根因诊断 (Bottleneck Diagnoses):
{diag_lines}

#### 🛠️ 生成给元 DAG 的优化指令 (Meta-DAG Prompt):
```text
{self.meta_prompt}
```
"""


# -----------------------------------------------------------------------------
# Watcher DAG Internal Edge Callables
# -----------------------------------------------------------------------------

def watcher_profile_performance(payload: Any, item: Any = None) -> str:
    """Watcher Edge 0: Execute target DAG against test cases and collect speed/efficiency metrics."""
    raw_text = item.contents if isinstance(item, ContentItem) else str(payload)
    data = json.loads(raw_text)

    target_dag_dict = data.get("target_dag", {})
    test_suite_data = data.get("test_suite", [])
    llm_type = data.get("llm_type", "mock")

    test_cases = [BenchmarkTestCase.from_dict(tc) for tc in test_suite_data]
    if not test_cases:
        test_cases = [
            BenchmarkTestCase(
                input_text="默认基准测试输入",
                input_tags=["input"],
                expected_output_tag="result",
            )
        ]

    total_latency = 0.0
    total_steps = 0
    total_fallbacks = 0
    all_edge_firings: Dict[str, int] = {e.get("id", f"edge_{i}"): 0 for i, e in enumerate(target_dag_dict.get("edges", []))}
    test_runs_data: List[Dict[str, Any]] = []

    for idx, tc in enumerate(test_cases):
        # Instantiate fresh instance of target DAG for each benchmark case
        run_dag = DAG.from_dict(target_dag_dict)
        if llm_type == "mock":
            run_dag.llm_client = MockLLMClient()

        run_dag.add_content(tc.input_text, tags=tc.input_tags, gen=0)

        t0 = time.perf_counter()
        step_records = run_dag.run(until_tag=tc.expected_output_tag, max_steps=tc.max_steps)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        total_latency += elapsed_ms
        total_steps += len(step_records)

        direct_produced_tags: List[str] = []
        for step in step_records:
            if step.action in ("default_targeted_assembly", "default_autonomous"):
                total_fallbacks += 1
            elif step.action == "edge" and step.edge_id and step.edge_id in all_edge_firings:
                all_edge_firings[step.edge_id] += 1
                for it in step.produced_items:
                    direct_produced_tags.extend(it.tags)

        output_content = run_dag.get_result(tc.expected_output_tag) or ""
        all_tags = [t for it in run_dag.contents for t in it.tags]

        test_runs_data.append({
            "case_idx": idx,
            "description": tc.description,
            "elapsed_ms": elapsed_ms,
            "steps": len(step_records),
            "reached_expected_tag": run_dag.has_tag(tc.expected_output_tag),
            "expected_output_tag": tc.expected_output_tag,
            "input_tags": tc.input_tags,
            "direct_produced_tags": direct_produced_tags,
            "output_content": output_content,
            "required_keywords": tc.required_keywords,
            "max_latency_ms": tc.max_latency_ms,
            "all_tags": all_tags,
        })

    n_cases = max(1, len(test_cases))
    dead_edges = [eid for eid, count in all_edge_firings.items() if count == 0]

    perf_profile = PerformanceProfile(
        total_latency_ms=round(total_latency, 2),
        avg_latency_ms=round(total_latency / n_cases, 2),
        total_steps=total_steps,
        avg_steps_per_case=round(total_steps / n_cases, 2),
        edge_firing_counts=all_edge_firings,
        dead_edges=dead_edges,
        default_processor_fallbacks=total_fallbacks,
    )

    return json.dumps(
        {
            "target_dag": target_dag_dict,
            "perf_profile": perf_profile.to_dict(),
            "test_runs_data": test_runs_data,
            "iteration": data.get("iteration", 0),
        },
        ensure_ascii=False,
    )


def watcher_evaluate_quality(payload: Any, item: Any = None) -> str:
    """Watcher Edge 1: Inspect outputs, keyword coverage, and correctness of results."""
    raw_text = item.contents if isinstance(item, ContentItem) else str(payload)
    data = json.loads(raw_text)

    test_runs_data = data.get("test_runs_data", [])
    perf_dict = data.get("perf_profile", {})
    target_dag_dict = data.get("target_dag", {})
    iteration = int(data.get("iteration", 0))

    n_cases = max(1, len(test_runs_data))
    success_count = 0
    total_keyword_matches = 0
    total_keywords_expected = 0
    failed_cases: List[str] = []
    sample_outputs: Dict[str, str] = {}

    for run in test_runs_data:
        case_id = f"case_{run.get('case_idx', 0)}"
        out = run.get("output_content", "")
        sample_outputs[case_id] = out[:100]

        if run.get("reached_expected_tag"):
            success_count += 1
        else:
            failed_cases.append(f"{case_id}: 未到达目标标签 '{run.get('expected_output_tag')}'")

        kws = run.get("required_keywords", [])
        if kws:
            total_keywords_expected += len(kws)
            matches = sum(1 for kw in kws if kw.lower() in out.lower())
            total_keyword_matches += matches
            if matches < len(kws):
                missing = [kw for kw in kws if kw.lower() not in out.lower()]
                failed_cases.append(f"{case_id}: 缺失必要关键词 {missing}")

    success_rate = success_count / n_cases
    kw_rate = (total_keyword_matches / max(1, total_keywords_expected)) if total_keywords_expected > 0 else 1.0
    quality_score = round(0.6 * success_rate + 0.4 * kw_rate, 4)

    quality_profile = QualityProfile(
        success_rate=round(success_rate, 4),
        keyword_coverage_rate=round(kw_rate, 4),
        quality_score=quality_score,
        failed_cases=failed_cases,
        sample_outputs=sample_outputs,
    )

    return json.dumps(
        {
            "target_dag": target_dag_dict,
            "perf_profile": perf_dict,
            "quality_profile": quality_profile.to_dict(),
            "test_runs_data": test_runs_data,
            "iteration": iteration,
        },
        ensure_ascii=False,
    )


def watcher_generate_report(payload: Any, item: Any = None) -> str:
    """Watcher Edge 2: Synthesize findings and formulate targeted Meta-DAG directives."""
    raw_text = item.contents if isinstance(item, ContentItem) else str(payload)
    data = json.loads(raw_text)

    target_dag_dict = data.get("target_dag", {})
    perf_dict = data.get("perf_profile", {})
    qual_dict = data.get("quality_profile", {})
    iteration = int(data.get("iteration", 0))
    test_runs = data.get("test_runs_data", [])

    perf = PerformanceProfile(**perf_dict)
    qual = QualityProfile(**qual_dict)

    diagnoses: List[str] = []
    meta_directives: List[str] = []

    # 1. 速度与效率分析
    if perf.default_processor_fallbacks > 0:
        diagnoses.append(
            f"效率瓶颈: 发生了 {perf.default_processor_fallbacks} 次 DefaultProcessor 兜底，"
            f"表明标签流中断或缺乏直接边，造成了额外调度延迟与计算开销。"
        )

    if perf.dead_edges:
        diagnoses.append(
            f"冗余边诊断: 检测到 {len(perf.dead_edges)} 条从不触发的死边: {perf.dead_edges}，应当修剪以保持图精简。"
        )
        for dead_id in perf.dead_edges:
            meta_directives.append(f"删除边 {dead_id}")

    # 2. 效果与质量分析
    if qual.quality_score < 0.8:
        diagnoses.append(
            f"效果缺陷: 当前综合质量评分仅为 {qual.quality_score:.2f}，"
            f"成功率 {qual.success_rate * 100:.1f}%，部分关键业务标签或内容未被有效生成。"
        )
        # Check if any test case failed to reach expected tag
        for r in test_runs:
            if not r.get("reached_expected_tag"):
                exp_t = r.get("expected_output_tag", "result")
                direct_t = r.get("direct_produced_tags", [])
                candidate_src = direct_t[-1] if direct_t else (r.get("input_tags", ["input"])[0])
                directive = f"添加边: [{candidate_src}] -> (sensenova-6.8: 生成目标结果) -> [{exp_t}] layer=0"
                if directive not in meta_directives:
                    meta_directives.append(directive)

    # 3. 兜底直连边修补 (仅在尚未生成定向连接边时提供)
    if perf.default_processor_fallbacks > 0 and not any("添加边" in d for d in meta_directives) and target_dag_dict.get("edges"):
        for edge in target_dag_dict["edges"]:
            if edge.get("id") not in perf.dead_edges:
                in_t = edge.get("input_tags", ["input"])
                out_t = edge.get("output_tags", ["processed"])
                meta_directives.append(
                    f"添加边: [{', '.join(in_t)}] -> (laptop-6.8: 高速直连预处理) -> [{', '.join(out_t)}] layer=0"
                )
                break

    is_converged = (
        qual.quality_score >= 0.85
        and perf.default_processor_fallbacks == 0
        and len(perf.dead_edges) == 0
    )

    if not meta_directives:
        meta_prompt = "微调边优先级: 保持架构稳定并确认全流程顺畅。"
    else:
        meta_prompt = "; ".join(meta_directives)

    overall_score = round(0.5 * qual.quality_score + 0.5 * max(0.0, 1.0 - (perf.avg_latency_ms / 1000.0)), 4)

    report = EvaluationReport(
        iteration=iteration,
        target_dag_name=target_dag_dict.get("name", "target_dag"),
        performance=perf,
        quality=qual,
        overall_score=overall_score,
        diagnoses=diagnoses,
        meta_prompt=meta_prompt,
        is_converged=is_converged,
    )

    return json.dumps(report.to_dict(), ensure_ascii=False)


# -----------------------------------------------------------------------------
# Watcher DAG Factory
# -----------------------------------------------------------------------------

def create_watcher_dag(
    name: str = "watcher_dag",
    *,
    models: Optional[Union[str, Path, Dict[str, Any], ModelRegistry]] = None,
    tools: Optional[Union[str, Path, Dict[str, Any], ToolRegistry]] = None,
    llm_client: Optional[LLMClientProtocol] = None,
) -> DAG:
    """Create a specialized Watcher DAG that monitors another DAG's effectiveness and efficiency.

    Topology:
      Edge 0 (`layer=0`): `["target_dag_input"]` -> `watcher_profile_performance` -> `["perf_data"]`
      Edge 1 (`layer=1`): `["perf_data"]` -> `watcher_evaluate_quality` -> `["quality_verdict"]`
      Edge 2 (`layer=2`): `["quality_verdict"]` -> `watcher_generate_report` -> `["evaluation_report", "result"]`
    """
    watcher = DAG(
        name=name,
        models=models,
        tools=tools,
        llm_client=llm_client or MockLLMClient(),
    )

    watcher.add_edge(
        input_tags=["target_dag_input"],
        processor=watcher_profile_performance,
        output_tags=["perf_data"],
        layer=0,
    )
    watcher.add_edge(
        input_tags=["perf_data"],
        processor=watcher_evaluate_quality,
        output_tags=["quality_verdict"],
        layer=1,
    )
    watcher.add_edge(
        input_tags=["quality_verdict"],
        processor=watcher_generate_report,
        output_tags=["evaluation_report", "result"],
        layer=2,
    )
    return watcher


# -----------------------------------------------------------------------------
# Watcher Optimizer Closed Loop Controller
# -----------------------------------------------------------------------------

class WatcherOptimizerLoop:
    """Closed-loop controller orchestrating:

    `Target DAG` -> `Watcher DAG` -> `Evaluation Report` -> `Meta-DAG` -> `Improved Target DAG` -> `Retest`
    """

    def __init__(
        self,
        target_dag: DAG,
        test_suite: List[Union[BenchmarkTestCase, Dict[str, Any]]],
        *,
        watcher_dag: Optional[DAG] = None,
        meta_dag: Optional[DAG] = None,
        max_iterations: int = 3,
        target_quality_score: float = 0.90,
        target_max_fallbacks: int = 0,
        target_max_latency_ms: float = 300.0,
        llm_client: Optional[LLMClientProtocol] = None,
    ) -> None:
        self.target_dag = target_dag
        self.test_suite = [
            tc if isinstance(tc, BenchmarkTestCase) else BenchmarkTestCase.from_dict(tc)
            for tc in test_suite
        ]
        self.client = llm_client or target_dag.llm_client
        self.watcher_dag = watcher_dag or create_watcher_dag(llm_client=self.client)
        self.meta_dag = meta_dag or create_meta_dag(llm_client=self.client)
        self.max_iterations = max_iterations
        self.target_quality_score = target_quality_score
        self.target_max_fallbacks = target_max_fallbacks
        self.target_max_latency_ms = target_max_latency_ms
        self.history: List[EvaluationReport] = []

    def run_watch_step(self, iteration: int) -> EvaluationReport:
        """Run the Watcher DAG against the current target DAG and return the structured report."""
        self.watcher_dag.contents.clear()
        self.watcher_dag._executed_edge_sigs.clear()
        self.watcher_dag._assembled_edge_sigs.clear()
        self.watcher_dag._autonomous_sigs.clear()

        payload = {
            "target_dag": self.target_dag.to_dict(),
            "test_suite": [tc.to_dict() for tc in self.test_suite],
            "iteration": iteration,
            "llm_type": "mock" if isinstance(self.client, MockLLMClient) else "real",
        }

        self.watcher_dag.add_content(json.dumps(payload, ensure_ascii=False), tags=["target_dag_input"], gen=0)
        self.watcher_dag.run(until_tag="result", max_steps=10)

        raw_report = self.watcher_dag.get_result("result")
        if not raw_report:
            raise RuntimeError(f"Watcher DAG did not produce an evaluation report for iteration {iteration}")

        report_dict = json.loads(raw_report)
        report = EvaluationReport(
            iteration=report_dict["iteration"],
            target_dag_name=report_dict["target_dag_name"],
            performance=PerformanceProfile(**report_dict["performance"]),
            quality=QualityProfile(**report_dict["quality"]),
            overall_score=report_dict["overall_score"],
            diagnoses=report_dict["diagnoses"],
            meta_prompt=report_dict["meta_prompt"],
            is_converged=report_dict.get("is_converged", False),
        )
        return report

    def run(self) -> Dict[str, Any]:
        """Execute the closed-loop optimization iterations until convergence or max iterations reached."""
        current_dag = self.target_dag

        for iter_idx in range(self.max_iterations):
            report = self.run_watch_step(iteration=iter_idx)
            self.history.append(report)

            # Check convergence conditions
            meets_quality = report.quality.quality_score >= self.target_quality_score
            meets_fallbacks = report.performance.default_processor_fallbacks <= self.target_max_fallbacks
            meets_dead_edges = len(report.performance.dead_edges) == 0

            if meets_quality and meets_fallbacks and meets_dead_edges:
                report.is_converged = True
                break

            if iter_idx == self.max_iterations - 1:
                # Last iteration reached
                break

            # Send report's meta_prompt to Meta-DAG to modify the target DAG!
            modified_dag = transform_dag_with_meta_dag(
                target_dag=current_dag,
                prompt=report.meta_prompt,
                meta_dag=self.meta_dag,
                llm_client=self.client,
            )
            current_dag = modified_dag
            self.target_dag = current_dag

        # Final verification report
        initial_rep = self.history[0]
        final_rep = self.history[-1]

        summary = {
            "converged": final_rep.is_converged,
            "total_iterations": len(self.history),
            "initial_quality": initial_rep.quality.quality_score,
            "final_quality": final_rep.quality.quality_score,
            "initial_latency_ms": initial_rep.performance.avg_latency_ms,
            "final_latency_ms": final_rep.performance.avg_latency_ms,
            "initial_fallbacks": initial_rep.performance.default_processor_fallbacks,
            "final_fallbacks": final_rep.performance.default_processor_fallbacks,
            "initial_dead_edges": len(initial_rep.performance.dead_edges),
            "final_dead_edges": len(final_rep.performance.dead_edges),
            "final_dag": self.target_dag,
            "history": self.history,
        }
        return summary
