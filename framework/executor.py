"""Executor module - Runs the computation graph with concurrency control.

执行器(Executor)模块 —— 带并发控制地运行计算图。

The executor drives the graph to completion using EVENT-DRIVEN scheduling:
vertices signal the executor as soon as they become READY (via a callback),
so there is no polling. It still bounds concurrency with a semaphore and
still detects deadlocks (no new READY events but graph not settled).

执行器用「事件驱动」调度把图跑到完成：顶点一旦 READY 立即通过回调唤醒
主循环(不再轮询扫描)，并用信号量限制并发，仍能检测死锁/超时。
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .vertex import Vertex, VertexState
from .edge import Edge
from .graph import Graph
from .pi_agent import PIAgent, MockPIAgent

logger = logging.getLogger("vertex_edge_agent.executor")


class ExecutionResult:
    """Collects the outcome of a graph execution.

    收集一次图执行的最终结果。
    """

    def __init__(self):
        self.success: bool = False  # 是否成功
        self.vertex_results: Dict[str, Dict] = {}  # 顶点 ID -> 状态快照
        self.edge_results: Dict[str, Any] = {}     # 边 ID -> 结果
        self.errors: List[str] = []                # 错误信息列表
        self.execution_time: float = 0.0           # 执行耗时(秒)

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"ExecutionResult({status}, V={len(self.vertex_results)}, "
            f"E={len(self.edge_results)}, errors={len(self.errors)}, "
            f"time={self.execution_time:.3f}s)"
        )

    def summary(self) -> str:
        """Human-readable summary.

        生成人类可读的执行结果摘要。
        """
        lines = [
            "=" * 60,
            f"  Execution Result: {'SUCCESS ✓' if self.success else 'FAILED ✗'}",
            f"  Time: {self.execution_time:.3f}s",
            f"  Vertices processed: {len(self.vertex_results)}",
            f"  Edges completed: {len(self.edge_results)}",
            f"  Errors: {len(self.errors)}",
            "=" * 60,
        ]
        if self.errors:
            lines.append("  ERRORS:")
            for err in self.errors:
                lines.append(f"    • {err}")
        lines.append("")
        lines.append("  VERTEX STATES:")
        for vid, info in self.vertex_results.items():
            state = info.get("state", "?")
            data_keys = list(info.get("data", {}).keys())
            lines.append(f"    [{vid}]  state={state}  keys={data_keys}")
        lines.append("")
        lines.append("  EDGE RESULTS:")
        for eid, val in self.edge_results.items():
            lines.append(f"    [{eid}]  {repr(val)[:100]}")
        lines.append("=" * 60)
        return "\n".join(lines)


class Executor:
    """Async executor that drives the graph to completion.

    驱动图运行至完成的异步执行器。

    Args:
        graph:            The Graph to execute.          要执行的图。
        pi_agent:         PI Agent instance (defaults to MockPIAgent).
                          PI Agent 实例(默认 MockPIAgent)。
        max_concurrency:  Max concurrent edge executions. 最大并发执行的边数。
        scan_interval:    Deprecated(事件驱动下已不再轮询，保留仅为兼容旧签名)。
        timeout:          Overall execution timeout in seconds.
                          整体执行超时(秒)。
    """

    def __init__(
        self,
        graph: Graph,
        pi_agent: Optional[PIAgent] = None,
        max_concurrency: int = 10,
        scan_interval: float = 0.05,
        timeout: Optional[float] = None,
    ):
        self.graph = graph
        self.pi_agent = pi_agent or MockPIAgent()
        self.max_concurrency = max_concurrency
        self.scan_interval = scan_interval
        self.timeout = timeout or 300.0
        self._semaphore = asyncio.Semaphore(max_concurrency)  # 并发限流信号量
        self._result = ExecutionResult()
        # 事件驱动调度：顶点进入 READY 时置位，唤醒主循环(取代轮询扫描)
        self._ready_signal = asyncio.Event()
        for v in self.graph.vertices.values():
            v.set_ready_callback(self._on_vertex_ready)

    # ------------------------------------------------------------------
    # Public API  公共接口
    # ------------------------------------------------------------------
    async def run(self) -> ExecutionResult:
        """Execute the graph and return an ``ExecutionResult``.

        执行整张图并返回 ``ExecutionResult``。
        """
        t0 = time.monotonic()  # 记录开始时间

        logger.info("=" * 60)
        logger.info("[Executor] ▶ Starting graph execution")
        logger.info("[Executor]   graph=%s", self.graph)
        logger.info("[Executor]   concurrency=%d  timeout=%ss", self.max_concurrency, self.timeout)
        logger.info("=" * 60)

        try:
            self._init_sources()  # 先把所有源顶点置为 READY
            await asyncio.wait_for(self._loop(), timeout=self.timeout)
            # 全部顶点都进入 DONE 才算整体成功
            self._result.success = all(
                v.state == VertexState.DONE for v in self.graph.vertices.values()
            )
        except asyncio.TimeoutError:
            # 超时
            msg = f"Execution timed out after {self.timeout}s"
            logger.error("[Executor] %s", msg)
            self._result.errors.append(msg)
        except Exception as exc:
            # 其他致命错误
            logger.error("[Executor] Fatal: %s", exc, exc_info=True)
            self._result.errors.append(str(exc))

        self._result.execution_time = time.monotonic() - t0
        await self._collect_results()  # 汇总各顶点最终状态与数据

        logger.info("=" * 60)
        logger.info("[Executor] ■ Finished: %s", self._result)
        logger.info("=" * 60)

        return self._result

    def _on_vertex_ready(self, vertex):
        """顶点进入 READY 时被回调：置位事件，通知主循环立即处理(幂等)。"""
        if not self._ready_signal.is_set():
            self._ready_signal.set()

    # ------------------------------------------------------------------
    # Internal  内部实现
    # ------------------------------------------------------------------
    def _init_sources(self):
        """Mark source vertices (no incoming edges) as READY.

        将所有源顶点(无入边)标记为 READY。
        """
        for v in self.graph.get_source_vertices():
            v.state = VertexState.READY
            logger.info("[Executor] Source vertex '%s' → READY", v.id)

    async def _loop(self):
        """Main event-driven loop.

        事件驱动主循环：不再轮询扫描，而是等待顶点进入 READY 的事件。
        每轮：唤醒 → 收集当前所有 READY 顶点 → 并发处理 → 终止/死锁检查。
        """
        while True:
            # 1. 等待有顶点进入 READY(事件驱动，无轮询)
            await self._ready_signal.wait()
            self._ready_signal.clear()

            # 2. 收集当前所有 READY 顶点
            ready = [
                v for v in self.graph.vertices.values()
                if v.state == VertexState.READY
            ]

            if ready:
                logger.info(
                    "[Executor] READY vertices: %s", [v.id for v in ready]
                )
                tasks = []
                for v in ready:
                    # 置为 PROCESSING 后并发处理
                    v.state = VertexState.PROCESSING
                    tasks.append(
                        asyncio.create_task(
                            self._process_vertex(v), name=f"proc_{v.id}"
                        )
                    )
                await asyncio.gather(*tasks, return_exceptions=True)

            # 3. 终止检查：所有顶点都已 DONE 或 ERROR
            states = {v.state for v in self.graph.vertices.values()}
            if states <= {VertexState.DONE, VertexState.ERROR}:
                logger.info("[Executor] All vertices settled, exiting loop")
                break

            # 4. 死锁检测：本轮 gather 完成后，若没有产生新的 READY 事件，
            #    而图又未结束，说明没有顶点能被推进 → 死锁
            if not self._ready_signal.is_set():
                self._log_state_dump()
                msg = "Deadlock – no new READY vertices but graph not settled"
                logger.error("[Executor] %s", msg)
                self._result.errors.append(msg)
                break

    async def _process_vertex(self, vertex: Vertex):
        """Fire all outgoing edges of *vertex*.

        触发 *vertex* 的所有出边。
        """
        logger.info("[Executor] Processing vertex '%s'", vertex.id)

        outgoing = self.graph.get_outgoing_edges(vertex.id)
        if not outgoing:
            # 汇顶点没有出边，直接完成
            vertex.state = VertexState.DONE
            logger.info("[Executor] Vertex '%s' is a sink → DONE", vertex.id)
            return

        # 先运行 on_ready 钩子，把多个输入整合成输出供出边读取
        try:
            await vertex.prepare_outputs()
        except Exception as exc:
            vertex.state = VertexState.ERROR
            vertex.error_message = f"prepare_outputs failed: {exc}"
            self._result.errors.append(f"Vertex '{vertex.id}': {exc}")
            return

        # 并发触发所有出边
        edge_tasks = [
            asyncio.create_task(self._fire_edge(e), name=f"edge_{e.id}")
            for e in outgoing
        ]
        results = await asyncio.gather(*edge_tasks, return_exceptions=True)

        # 汇总每条边的执行结果，判断是否有失败
        ok = True
        for edge, res in zip(outgoing, results):
            if isinstance(res, Exception):
                logger.error("[Executor] Edge '%s' error: %s", edge.id, res)
                self._result.errors.append(f"Edge '{edge.id}': {res}")
                ok = False

        if ok:
            vertex.state = VertexState.DONE
            logger.info("[Executor] Vertex '%s' → DONE", vertex.id)
        else:
            # 任一出边失败 => 顶点进入 ERROR
            vertex.state = VertexState.ERROR
            vertex.error_message = "One or more outgoing edges failed"
            logger.error("[Executor] Vertex '%s' → ERROR", vertex.id)

    async def _fire_edge(self, edge: Edge) -> Any:
        """Execute one edge, bounded by the concurrency semaphore.

        执行一条边，受并发信号量限流。
        """
        async with self._semaphore:
            src = self.graph.vertices[edge.source_id]
            dst = self.graph.vertices[edge.destination_id]
            result = await edge.execute(src, dst, self.pi_agent)
            self._result.edge_results[edge.id] = result
            return result

    async def _collect_results(self):
        """Snapshot every vertex's final state and data.

        快照每个顶点的最终状态与数据。
        """
        for v in self.graph.vertices.values():
            data = await v.get_all_data()
            # 把 (data_id, (tags...)) 键转换为 "data_id:tag1,tag2" 便于阅读
            self._result.vertex_results[v.id] = {
                "state": v.state.value,
                "data": {
                    str(k): val for k, val in data.items()
                },
                "error": v.error_message,
            }

    def _log_state_dump(self):
        """Dump the state of every vertex for debugging.

        输出所有顶点的状态，便于死锁排查。
        """
        logger.warning("[Executor] ── state dump ──")
        for v in self.graph.vertices.values():
            logger.warning(
                "  [%s] state=%s  in=%d/%d  out=%s",
                v.id, v.state.value,
                v._received_input_count, v.required_input_count,
                v.outgoing_edges,
            )
