"""Executor module - Runs the computation graph with concurrency control.

The executor repeatedly scans for READY vertices, fires their outgoing
edges concurrently (bounded by a semaphore), and advances vertex states
until the entire graph is DONE or a deadlock / timeout is detected.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .vertex import Vertex, VertexState, EdgeSignal
from .edge import Edge
from .graph import Graph
from .pi_agent import PIAgent, MockPIAgent

logger = logging.getLogger("vertex_edge_agent.executor")


class ExecutionResult:
    """Collects the outcome of a graph execution."""

    def __init__(self):
        self.success: bool = False
        self.vertex_results: Dict[str, Dict] = {}
        self.edge_results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.execution_time: float = 0.0

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"ExecutionResult({status}, V={len(self.vertex_results)}, "
            f"E={len(self.edge_results)}, errors={len(self.errors)}, "
            f"time={self.execution_time:.3f}s)"
        )

    def summary(self) -> str:
        """Human-readable summary."""
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
            abort_str = f" (aborted: {info.get('abort_reason')})" if state == "aborted" else ""
            err_str = f" (error: {info.get('error')})" if state == "error" else ""
            lines.append(f"    [{vid}]  state={state}{abort_str}{err_str}  keys={data_keys}")
        lines.append("")
        lines.append("  EDGE RESULTS:")
        for eid, val in self.edge_results.items():
            lines.append(f"    [{eid}]  {repr(val)[:100]}")
        lines.append("=" * 60)
        return "\n".join(lines)


class Executor:
    """Async executor that drives the graph to completion.

    Args:
        graph:            The Graph to execute.
        pi_agent:         PI Agent instance (defaults to MockPIAgent).
        max_concurrency:  Max concurrent edge executions.
        scan_interval:    Seconds between ready-vertex scans.
        timeout:          Overall execution timeout in seconds.
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
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._result = ExecutionResult()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run(self) -> ExecutionResult:
        """Execute the graph and return an ``ExecutionResult``."""
        t0 = time.monotonic()

        logger.info("=" * 60)
        logger.info("[Executor] ▶ Starting graph execution")
        logger.info("[Executor]   graph=%s", self.graph)
        logger.info("[Executor]   concurrency=%d  timeout=%ss", self.max_concurrency, self.timeout)
        logger.info("=" * 60)

        try:
            self._init_sources()
            await asyncio.wait_for(self._loop(), timeout=self.timeout)
            self._result.success = (
                len(self._result.errors) == 0
                and all(v.state in (VertexState.DONE, VertexState.ABORTED) for v in self.graph.vertices.values())
            )
        except asyncio.TimeoutError:
            msg = f"Execution timed out after {self.timeout}s"
            logger.error("[Executor] %s", msg)
            self._result.errors.append(msg)
        except Exception as exc:
            logger.error("[Executor] Fatal: %s", exc, exc_info=True)
            self._result.errors.append(str(exc))

        self._result.execution_time = time.monotonic() - t0
        await self._collect_results()

        logger.info("=" * 60)
        logger.info("[Executor] ■ Finished: %s", self._result)
        logger.info("=" * 60)

        return self._result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _init_sources(self):
        """Mark source vertices (no incoming edges) as READY."""
        for v in self.graph.get_source_vertices():
            v.state = VertexState.READY
            logger.info("[Executor] Source vertex '%s' → READY", v.id)

    async def _loop(self):
        """Event-driven main loop."""
        iteration = 0

        async def wait_and_process(vertex: Vertex):
            if vertex.state not in (VertexState.READY, VertexState.PROCESSING, VertexState.DONE, VertexState.ABORTED, VertexState.ERROR):
                await vertex.wait_ready()
            
            if vertex.state == VertexState.READY:
                vertex.state = VertexState.PROCESSING
                await self._process_vertex(vertex)
            elif vertex.state == VertexState.ABORTED:
                await self._abort_vertex(vertex)

        # Create a task for each vertex
        pending = {
            asyncio.create_task(wait_and_process(v), name=f"task_{v.id}")
            for v in self.graph.vertices.values()
        }

        while pending:
            iteration += 1
            logger.debug("[Executor] ── event wait #%d ──", iteration)

            # Wait for at least one task to complete, or timeout to check for deadlocks
            done, pending = await asyncio.wait(
                pending,
                timeout=self.scan_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Handle exceptions from done tasks
            for task in done:
                exc = task.exception()
                if exc:
                    logger.error("[Executor] Task failed: %s", exc)

            # 1. Terminal check
            states = {v.state for v in self.graph.vertices.values()}
            if states <= {VertexState.DONE, VertexState.ABORTED, VertexState.ERROR}:
                logger.info("[Executor] All vertices settled, exiting loop")
                break

            # 2. Deadlock detection
            # If no tasks completed in this interval AND no vertex is READY or PROCESSING,
            # then nothing is happening and nothing will happen (deadlock).
            if not done and VertexState.READY not in states and VertexState.PROCESSING not in states:
                self._log_state_dump()
                msg = "Deadlock – no READY/PROCESSING vertices but graph not settled"
                logger.error("[Executor] %s", msg)
                self._result.errors.append(msg)
                
                # Cancel remaining tasks
                for t in pending:
                    t.cancel()
                break

    async def _abort_vertex(self, vertex: Vertex):
        """Cascade abort to all outgoing edges of an aborted vertex."""
        logger.info("[Executor] Vertex '%s' aborted → cascading to outgoing edges", vertex.id)
        outgoing = self.graph.get_outgoing_edges(vertex.id)
        for edge in outgoing:
            edge.aborted = True
            edge.abort_reason = f"Upstream vertex '{vertex.id}' was aborted"
            self._result.edge_results[edge.id] = f"<ABORTED: {edge.abort_reason}>"
            dst = self.graph.vertices[edge.destination_id]
            await dst.handle_edge_signal(edge.id, EdgeSignal.ABORTED, payload=edge.abort_reason)

    async def _process_vertex(self, vertex: Vertex):
        """Fire all outgoing edges of *vertex*."""
        logger.info("[Executor] Processing vertex '%s'", vertex.id)

        outgoing = self.graph.get_outgoing_edges(vertex.id)
        if not outgoing:
            vertex.state = VertexState.DONE
            logger.info("[Executor] Vertex '%s' is a sink → DONE", vertex.id)
            return

        # Run on_ready hook to consolidate data for outgoing reads
        try:
            await vertex.prepare_outputs()
        except Exception as exc:
            vertex.state = VertexState.ERROR
            vertex.error_message = f"prepare_outputs failed: {exc}"
            self._result.errors.append(f"Vertex '{vertex.id}': {exc}")
            return

        # Fire edges concurrently
        edge_tasks = [
            asyncio.create_task(self._fire_edge(e), name=f"edge_{e.id}")
            for e in outgoing
        ]
        results = await asyncio.gather(*edge_tasks, return_exceptions=True)

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
            vertex.state = VertexState.ERROR
            vertex.error_message = "One or more outgoing edges failed"
            logger.error("[Executor] Vertex '%s' → ERROR", vertex.id)

    async def _fire_edge(self, edge: Edge) -> Any:
        """Execute one edge, bounded by the concurrency semaphore."""
        async with self._semaphore:
            src = self.graph.vertices[edge.source_id]
            dst = self.graph.vertices[edge.destination_id]
            result = await edge.execute(src, dst, self.pi_agent)
            if edge.aborted:
                self._result.edge_results[edge.id] = f"<ABORTED: {edge.abort_reason}>"
            else:
                self._result.edge_results[edge.id] = result
            return result

    async def _collect_results(self):
        """Snapshot every vertex's final state and data."""
        for v in self.graph.vertices.values():
            data = await v.get_all_data()
            self._result.vertex_results[v.id] = {
                "state": v.state.value,
                "data": {
                    f"{k[0]}:{','.join(k[1])}": val for k, val in data.items()
                },
                "error": v.error_message,
                "abort_reason": v.abort_reason,
            }
        for edge in self.graph.edges.values():
            if edge.id not in self._result.edge_results:
                if edge.aborted:
                    self._result.edge_results[edge.id] = f"<ABORTED: {edge.abort_reason}>"
                elif edge.error:
                    self._result.edge_results[edge.id] = f"<FAILED: {edge.error}>"

    def _log_state_dump(self):
        """Dump the state of every vertex for debugging."""
        logger.warning("[Executor] ── state dump ──")
        for v in self.graph.vertices.values():
            logger.warning(
                "  [%s] state=%s  in=%d/%d  out=%s",
                v.id, v.state.value,
                v._received_input_count, v.required_input_count,
                v.outgoing_edges,
            )
