"""VEA V5 增量拓扑执行器模块 (Executor).

核心设计特性：
1. **按层并发调度**：利用 `dag.topological_generations()` 按层（Tier）并发执行所有就绪边。
2. **原生支持虚拟边 (`VirtualEdge`)**：遇到虚拟边时无需调用 LLM，自动将上游相似顶点的数据按合并策略
   （`TEXT_JOIN` / `JSON_MERGE` / `LIST_APPEND` / `OVERWRITE`）汇聚到目标顶点。
3. **真正的增量重跑（Incremental Re-execution）**：如果某个顶点已经处于 `READY` 状态（在上一轮闭环中已算好且未受本次修改影响），
   执行器会自动跳过它的入边，只重算处于 `PENDING` 状态的脏分支！
4. **同步/异步双模式无感调用**：既支持 `await executor.run(inputs)`，也支持在普通脚本中直接 `executor.run_sync(inputs)`。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from vea_v5.dag import DAG
from vea_v5.llm import LLMProtocol, SmartMockLLM
from vea_v5.primitives import Edge, MergeMode, VertexStatus


@dataclass
class EdgeTrace:
    """单条边的执行轨迹记录。"""

    edge_id: str
    source: str
    target: str
    kind: str
    action: str
    is_virtual: bool
    input_value: Any
    output_value: Any
    elapsed_ms: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class ExecutionReport:
    """DAG 单次运行完整结果报告。"""

    success: bool
    executed_edges: List[str] = field(default_factory=list)
    skipped_vertices: List[str] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    all_vertices: Dict[str, Any] = field(default_factory=dict)
    traces: List[EdgeTrace] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def final_output(self) -> Any:
        """若只有一个终点汇点（Sink），直接返回其内容；若有多个则返回字典。"""
        if len(self.outputs) == 1:
            return next(iter(self.outputs.values()))
        return self.outputs


def merge_values(existing: Any, incoming: Any, mode: MergeMode) -> Any:
    """根据合并策略将新产出的数据 `incoming` 合并入目标顶点的 `existing` 数据中。"""
    if existing is None or mode == MergeMode.OVERWRITE:
        return incoming

    if mode == MergeMode.TEXT_JOIN:
        if isinstance(existing, dict) and isinstance(incoming, dict):
            merged_d = dict(existing)
            merged_d.update(incoming)
            return merged_d
        s1 = existing if isinstance(existing, str) else json.dumps(existing, ensure_ascii=False)
        s2 = incoming if isinstance(incoming, str) else json.dumps(incoming, ensure_ascii=False)
        if not s1:
            return s2
        if not s2 or s2 in s1:
            return s1
        return f"{s1}\n{s2}"

    if mode == MergeMode.LIST_APPEND:
        base_list = list(existing) if isinstance(existing, list) else [existing]
        if isinstance(incoming, list):
            base_list.extend(incoming)
        else:
            base_list.append(incoming)
        return base_list

    if mode == MergeMode.JSON_MERGE:
        def _to_dict(val: Any) -> Dict[str, Any]:
            if isinstance(val, dict):
                return dict(val)
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
                return {"text": val}
            return {"value": val}

        merged = _to_dict(existing)
        merged.update(_to_dict(incoming))
        return merged

    return incoming


class Executor:
    """VEA V5 增量并发 DAG 执行器。"""

    def __init__(
        self,
        dag: DAG,
        llm: Optional[LLMProtocol] = None,
        max_concurrency: int = 8,
    ) -> None:
        self.dag = dag
        self.llm: LLMProtocol = llm or SmartMockLLM()
        self.max_concurrency = max(1, max_concurrency)

    async def _run_single_edge(self, edge: Edge, semaphore: asyncio.Semaphore) -> EdgeTrace:
        """执行单条计算边或虚拟边。"""
        t0 = time.perf_counter()
        src_vertex = self.dag.vertices[edge.source]
        raw_input = src_vertex.content
        input_str = src_vertex.content_as_str()

        async with semaphore:
            try:
                # 1. 虚拟边（VirtualEdge）或直通边：零成本直接桥接上游数据
                if edge.is_virtual or edge.kind in ("virtual", "passthrough"):
                    out_val = raw_input

                # 2. Python 代码边（函数或源码字符串）
                elif edge.kind == "code":
                    fn = edge.fn
                    if fn is None and edge.code:
                        ns: Dict[str, Any] = {}
                        exec(edge.code, ns)
                        for cand in ("transform", "execute", "run", "process"):
                            if cand in ns and callable(ns[cand]):
                                fn = ns[cand]
                                break
                        if fn is None:
                            # 查找用户定义的第一个可调用函数
                            callables = [v for k, v in ns.items() if callable(v) and not k.startswith("__")]
                            if callables:
                                fn = callables[-1]
                    if fn is None:
                        raise ValueError(f"代码边 '{edge.id}' 未提供可执行的 fn 或 code 函数定义")

                    sig = inspect.signature(fn)
                    params = [
                        p for p in sig.parameters.values()
                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    ]
                    if len(params) >= 2:
                        res = fn(raw_input, edge.metadata)
                    else:
                        res = fn(raw_input)
                    if asyncio.iscoroutine(res):
                        res = await res
                    out_val = res

                # 3. LLM 推理边
                elif edge.kind == "llm":
                    rendered = edge.render_prompt(input_str)
                    llm_out = await self.llm.ask(rendered)
                    # 如果返回的是合法 JSON 字符串，尝试自动解析为结构化数据以方便下游使用
                    out_val = llm_out
                    if isinstance(llm_out, str):
                        stripped = llm_out.strip()
                        if (stripped.startswith("{") and stripped.endswith("}")) or (
                            stripped.startswith("[") and stripped.endswith("]")
                        ):
                            try:
                                out_val = json.loads(stripped)
                            except Exception:
                                out_val = llm_out
                else:
                    raise ValueError(f"未知的边类型: {edge.kind}")

                elapsed = (time.perf_counter() - t0) * 1000.0
                return EdgeTrace(
                    edge_id=edge.id,
                    source=edge.source,
                    target=edge.target,
                    kind=edge.kind,
                    action=edge.action,
                    is_virtual=edge.is_virtual,
                    input_value=raw_input,
                    output_value=out_val,
                    elapsed_ms=round(elapsed, 2),
                    success=True,
                )
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return EdgeTrace(
                    edge_id=edge.id,
                    source=edge.source,
                    target=edge.target,
                    kind=edge.kind,
                    action=edge.action,
                    is_virtual=edge.is_virtual,
                    input_value=raw_input,
                    output_value=None,
                    elapsed_ms=round(elapsed, 2),
                    success=False,
                    error=str(exc),
                )

    async def run(self, inputs: Optional[Dict[str, Any]] = None) -> ExecutionReport:
        """异步执行 DAG（自动跳过已就绪的未修改上游顶点，仅增量执行待计算分支）。

        Args:
            inputs: 可选的入口顶点初始数据字典 `{vertex_name: content}`。
                    若整个图只有一个入口起点（Root）且传入了非字典值或单键值，也会自动填入。
        """
        t_start = time.perf_counter()
        if inputs:
            for v_name, val in inputs.items():
                if v_name in self.dag.vertices:
                    self.dag.add_vertex(v_name, content=val)

        # 确保所有没有入边的起点（Roots）若已有默认内容则标记为 READY
        for r_name in self.dag.roots():
            r_v = self.dag.vertices[r_name]
            if r_v.content is not None and not r_v.is_ready:
                r_v.set_ready(r_v.content)

        generations = self.dag.topological_generations()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        executed_edges: List[str] = []
        skipped_vertices: List[str] = []
        traces: List[EdgeTrace] = []
        errors: List[str] = []

        # 按拓扑层级从上游到下游推进
        for layer_idx, layer_vertices in enumerate(generations):
            for v_name in layer_vertices:
                target_v = self.dag.vertices[v_name]
                in_edges = self.dag.incoming_edges(v_name)

                # 如果该顶点没有入边（是入口 Root），检查是否已就绪
                if not in_edges:
                    if target_v.is_ready:
                        skipped_vertices.append(v_name)
                    continue

                # 如果该顶点已经处于 READY 状态（上一轮已算好且未受本次 DAG 修改波及），直接跳过！
                if target_v.is_ready:
                    skipped_vertices.append(v_name)
                    continue

                # 检查所有前置上游顶点是否全部就绪
                unready_upstreams = [
                    e.source for e in in_edges if not self.dag.vertices[e.source].is_ready
                ]
                if unready_upstreams:
                    target_v.status = VertexStatus.FAILED
                    err_msg = f"顶点 '{v_name}' 的上游顶点 {unready_upstreams} 未就绪，无法执行入边"
                    errors.append(err_msg)
                    continue

                # 并发执行汇入 target_v 的所有入边（包括真实计算边与虚拟边）
                edge_traces: List[EdgeTrace] = await asyncio.gather(
                    *(self._run_single_edge(e, semaphore) for e in in_edges)
                )

                # 按边 ID 稳定顺序合并入边结果，保证并发执行结果 100% 确定性
                merged_content: Any = None
                all_ok = True
                for e_obj, tr in zip(in_edges, edge_traces):
                    traces.append(tr)
                    if not tr.success:
                        all_ok = False
                        errors.append(f"边 '{tr.edge_id}' ({tr.source} -> {tr.target}) 执行失败: {tr.error}")
                    else:
                        executed_edges.append(tr.edge_id)
                        merged_content = merge_values(merged_content, tr.output_value, e_obj.merge_mode)

                if all_ok:
                    target_v.set_ready(merged_content)
                else:
                    target_v.status = VertexStatus.FAILED

        sink_names = self.dag.sinks()
        outputs = {name: self.dag.vertices[name].content for name in sink_names}
        all_vertices = {name: v.content for name, v in self.dag.vertices.items()}
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return ExecutionReport(
            success=(len(errors) == 0 and all(self.dag.vertices[s].is_ready for s in sink_names)),
            executed_edges=executed_edges,
            skipped_vertices=skipped_vertices,
            outputs=outputs,
            all_vertices=all_vertices,
            traces=traces,
            errors=errors,
            elapsed_ms=round(total_ms, 2),
        )

    def run_sync(self, inputs: Optional[Dict[str, Any]] = None) -> ExecutionReport:
        """同步运行入口（方便在普通脚本中直接调用，无需手写 `asyncio.run`）。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 若已在事件循环中，开启独立线程执行协程
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, self.run(inputs))
                return fut.result()
        return asyncio.run(self.run(inputs))
