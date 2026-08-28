"""Executor module - Runs the computation graph with concurrency control.

The executor repeatedly scans for READY vertices, fires their outgoing
edges concurrently (bounded by a semaphore), and advances vertex states
until the entire graph is DONE or a deadlock / timeout is detected.

Stateful loop support: when a loop-back edge delivers to a destination
vertex that is already DONE, ``Vertex.receive_signal`` resets that vertex
to READY.  The executor detects newly-READY vertices on each scan and
spawns fresh processing tasks for them, enabling bounded self-correction
cycles without changing the overall event-driven architecture.
"""

import asyncio
import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from ..vertex import Vertex, VertexState, EdgeSignal
from ..edge import Edge
from ..graph import Graph
from ..agents import BaseAgent, MockAgent
from ..utils.memory import MemoryStore
from ..utils.telemetry import TelemetryTracker, UsageMetrics

logger = logging.getLogger("vertex_edge_agent.executor")


@dataclass
class GraphEvent:
    """Standard event emitted during graph execution."""
    event_type: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"))
    vertex_id: Optional[str] = None
    edge_id: Optional[str] = None
    payload: Optional[Any] = None


class ExecutionResult:
    """Collects the outcome of a graph execution."""

    def __init__(self):
        self.success: bool = False
        self.vertex_results: Dict[str, Dict] = {}
        self.edge_results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.execution_time: float = 0.0
        self.metrics: Optional[UsageMetrics] = None
        self.memory_snapshot: Dict[str, Any] = {}

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
        ]
        if self.metrics:
            lines.extend([
                f"  Prompt Tokens: {self.metrics.prompt_tokens}",
                f"  Completion Tokens: {self.metrics.completion_tokens}",
                f"  Total Tokens: {self.metrics.total_tokens}",
                f"  Estimated Cost: ${self.metrics.cost_usd:.6f}",
            ])
        lines.append("=" * 60)
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
            iter_str = f" (iterations: {info.get('iterations', 0)})" if info.get("iterations") else ""
            lines.append(f"    [{vid}]  state={state}{abort_str}{err_str}{iter_str}  keys={data_keys}")
        lines.append("")
        lines.append("  EDGE RESULTS:")
        for eid, val in self.edge_results.items():
            lines.append(f"    [{eid}]  {repr(val)[:100]}")
        if self.memory_snapshot:
            lines.append("")
            lines.append("  GLOBAL MEMORY SNAPSHOT:")
            for mk, mv in self.memory_snapshot.items():
                lines.append(f"    • {mk}: {repr(mv)[:100]}")
        lines.append("=" * 60)
        return "\n".join(lines)


class ExecutorHooks:
    """Optional callbacks for observing and intercepting workflow execution."""
    async def on_workflow_start(self, graph: Graph) -> None:
        pass

    async def on_vertex_state_changed(self, vertex: Vertex, state: VertexState) -> None:
        pass

    async def on_edge_started(self, edge: Edge) -> None:
        pass

    async def on_edge_completed(self, edge: Edge, result: Any) -> None:
        pass

    async def on_cancel_edges(self, edge_ids: List[str]) -> None:
        pass

    async def on_workflow_finish(self, result: ExecutionResult) -> None:
        pass


class Executor:
    """Async executor that drives the graph to completion with streaming event observability.

    Args:
        graph:            The Graph to execute.
        agents:           PI Agent instance (defaults to MockAgent).
        max_concurrency:  Max concurrent edge executions.
        scan_interval:    Seconds between ready-vertex scans.
        timeout:          Overall execution timeout in seconds.
        memory:           Optional MemoryStore instance for shared cross-vertex context.
        telemetry:        Optional TelemetryTracker instance for token/cost tracking.
        hooks:            Optional ExecutorHooks instance for lifecycle interception.
    """

    def __init__(
        self,
        graph: Graph,
        agents: Optional[BaseAgent] = None,
        max_concurrency: int = 10,
        concurrency_config: Optional[Dict[str, int]] = None,
        scan_interval: float = 0.05,
        timeout: Optional[float] = None,
        memory: Optional[MemoryStore] = None,
        telemetry: Optional[TelemetryTracker] = None,
        hooks: Optional[ExecutorHooks] = None,
    ):
        self.graph = graph
        self.agents = agents or MockAgent()
        self.max_concurrency = max_concurrency
        self.scan_interval = scan_interval
        self.timeout = timeout or 300.0
        self.memory = memory or MemoryStore()
        self.telemetry = telemetry or TelemetryTracker()
        self.hooks = hooks
        # Per-pipeline-type semaphores: llm, fetch, default
        default_concurrency = concurrency_config or {}
        self._semaphores = {
            "llm": asyncio.Semaphore(default_concurrency.get("llm", max_concurrency)),
            "fetch": asyncio.Semaphore(default_concurrency.get("fetch", max_concurrency)),
            "default": asyncio.Semaphore(default_concurrency.get("default", max_concurrency)),
        }
        self._result = ExecutionResult()
        self.active_edge_tasks = {}
        self._event_queue: asyncio.Queue[Optional[GraphEvent]] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _emit(
        self,
        event_type: str,
        vertex_id: Optional[str] = None,
        edge_id: Optional[str] = None,
        payload: Optional[Any] = None,
    ) -> None:
        """Non-blocking event emitter via internal sidecar queue and hooks."""
        event = GraphEvent(
            event_type=event_type,
            vertex_id=vertex_id,
            edge_id=edge_id,
            payload=payload,
        )
        self._event_queue.put_nowait(event)

        if self.hooks:
            try:
                if event_type == "workflow_started" and hasattr(self.hooks, "on_workflow_start"):
                    res = self.hooks.on_workflow_start(self.graph)
                    if asyncio.iscoroutine(res):
                        asyncio.create_task(res)
                elif event_type == "workflow_finished" and hasattr(self.hooks, "on_workflow_finish"):
                    res = self.hooks.on_workflow_finish(self._result)
                    if asyncio.iscoroutine(res):
                        asyncio.create_task(res)
                elif event_type == "vertex_state_changed" and vertex_id and hasattr(self.hooks, "on_vertex_state_changed"):
                    v = self.graph.vertices.get(vertex_id)
                    if v:
                        res = self.hooks.on_vertex_state_changed(v, v.state)
                        if asyncio.iscoroutine(res):
                            asyncio.create_task(res)
                elif event_type == "edge_started" and edge_id and hasattr(self.hooks, "on_edge_started"):
                    e = self.graph.edges.get(edge_id)
                    if e:
                        res = self.hooks.on_edge_started(e)
                        if asyncio.iscoroutine(res):
                            asyncio.create_task(res)
                elif event_type == "edge_completed" and edge_id and hasattr(self.hooks, "on_edge_completed"):
                    e = self.graph.edges.get(edge_id)
                    if e:
                        res = self.hooks.on_edge_completed(e, payload.get("result") if isinstance(payload, dict) else payload)
                        if asyncio.iscoroutine(res):
                            asyncio.create_task(res)
            except Exception as hook_err:
                logger.warning("[Executor] Hook dispatch error: %s", hook_err)

    async def stream(self) -> AsyncGenerator[GraphEvent, None]:
        """Stream execution events asynchronously as they occur without blocking execution."""
        run_task = asyncio.create_task(self._run_internal())
        try:
            while True:
                event = await self._event_queue.get()
                if event is None:
                    break
                yield event
        finally:
            await run_task

    async def run(self) -> ExecutionResult:
        """Execute the graph and return an ``ExecutionResult``."""
        async for _ in self.stream():
            pass
        return self._result

    async def _run_internal(self) -> ExecutionResult:
        """Internal runner that powers both run() and stream()."""
        t0 = time.monotonic()

        logger.debug("=" * 60)
        
        # --- Wire up cancellation callbacks for Race mode ---
        def cancel_edges_callback(edge_ids):
            for eid in edge_ids:
                if eid in self.active_edge_tasks:
                    logger.debug("[Executor] Race condition won. Cancelling pending edge '%s'", eid)
                    self.active_edge_tasks[eid].cancel()

        for v in self.graph.vertices.values():
            v.on_cancel_edges = cancel_edges_callback

        logger.debug("[Executor] ▶ Starting graph execution")
        logger.debug("[Executor]   graph=%s", self.graph)
        logger.debug("[Executor]   concurrency=%d  timeout=%ss", self.max_concurrency, self.timeout)
        logger.debug("=" * 60)
        self._emit("workflow_started", payload={"timeout": self.timeout, "concurrency": self.max_concurrency})

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
            self._emit("workflow_error", payload={"error": msg})
        except Exception as exc:
            logger.error("[Executor] Fatal: %s", exc, exc_info=True)
            self._result.errors.append(str(exc))
            self._emit("workflow_error", payload={"error": str(exc)})

        self._result.execution_time = time.monotonic() - t0
        await self._collect_results()

        self._emit(
            "workflow_finished",
            payload={"success": self._result.success, "execution_time": self._result.execution_time}
        )
        # Put sentinel to close the event stream
        self._event_queue.put_nowait(None)

        logger.debug("=" * 60)
        logger.debug("[Executor] ■ Finished: %s", self._result)
        logger.debug("=" * 60)

        return self._result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _init_sources(self):
        """Mark source vertices (no incoming edges) as READY (or PAUSED if approval required).

        In a cyclic graph, vertices whose *only* incoming edges are loop-back
        edges (``edge_id in vertex.loop_incoming_edges``) are treated as
        logical sources for the first iteration boot — the loop-back edge
        hasn't fired yet, so they must self-start.
        """
        for v in self.graph.vertices.values():
            non_loop_incoming = [
                eid for eid in v.incoming_edges
                if eid not in v.loop_incoming_edges
            ]
            if not non_loop_incoming:
                # Pure source OR loop-destination with no other inputs
                if v._require_approval and not v._approved:
                    v.state = VertexState.PAUSED
                    logger.debug("[Executor] Source/loop-boot vertex '%s' → PAUSED (requires approval)", v.id)
                else:
                    v.state = VertexState.READY
                    logger.debug("[Executor] Source/loop-boot vertex '%s' → READY", v.id)

    async def _loop(self):
        """Event-driven main loop with stateful-loop and PAUSED support.

        Strategy:
        1. Spawn one ``wait_and_process`` task per vertex for the first round.
        2. After each ``asyncio.wait`` iteration, scan for any vertex that has
           become READY again (loop re-entry or approval) and spawn a fresh
           ``_process_ready_vertex`` task for it.
        3. Exit when all vertices are settled, paused, or a deadlock is detected.
        """
        iteration = 0

        async def wait_and_process(vertex: Vertex):
            """First-round task: wait for READY then process once."""
            if vertex.state not in (
                VertexState.READY, VertexState.AWAITING_EDGES,
                VertexState.DONE, VertexState.ABORTED, VertexState.ERROR,
            ):
                await vertex.wait_ready()

            if vertex.state == VertexState.READY:
                vertex.state = VertexState.AWAITING_EDGES
                await self._process_vertex(vertex)
            elif vertex.state == VertexState.ABORTED:
                await self._abort_vertex(vertex)

        # Maps vertex_id -> most-recently-spawned task for that vertex.
        # Used to avoid double-scheduling when a vertex is already processing.
        vertex_tasks: Dict[str, asyncio.Task] = {}

        # Spawn initial tasks
        for v in self.graph.vertices.values():
            task = asyncio.create_task(
                wait_and_process(v), name=f"task_{v.id}"
            )
            vertex_tasks[v.id] = task

        pending: Set[asyncio.Task] = set(vertex_tasks.values())

        while pending:
            iteration += 1
            logger.debug("[Executor] ── event wait #%d ──", iteration)

            done, pending = await asyncio.wait(
                pending,
                timeout=self.scan_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Surface task exceptions
            for task in done:
                exc = task.exception()
                if exc:
                    logger.error("[Executor] Task failed: %s", exc)

            # 1. Terminal check: all vertices are in terminal states or PAUSED
            states = {v.state for v in self.graph.vertices.values()}
            if states <= {VertexState.DONE, VertexState.ABORTED, VertexState.ERROR, VertexState.PAUSED}:
                logger.debug("[Executor] All active work settled or paused, exiting loop")
                break

            # 2. Loop re-entry / Approval resume: schedule processing for any newly-READY vertex
            #    that doesn't already have an active task.
            for v in self.graph.vertices.values():
                if v.state == VertexState.READY:
                    vid = v.id
                    existing = vertex_tasks.get(vid)
                    if existing is None or existing.done():
                        logger.debug(
                            "[Executor] Vertex '%s' READY (iter #%d) — spawning task",
                            vid, v.iteration_count,
                        )
                        task = asyncio.create_task(
                            self._process_ready_vertex(v),
                            name=f"task_{vid}_iter{v.iteration_count}",
                        )
                        vertex_tasks[vid] = task
                        pending.add(task)

            # 3. Deadlock detection: no active tasks running/ready, but non-paused idle vertices remain
            states = {v.state for v in self.graph.vertices.values()}
            if not done and VertexState.READY not in states and VertexState.AWAITING_EDGES not in states:
                # If there are PAUSED vertices waiting for external approval, it's a pause, not a deadlock
                if VertexState.PAUSED in states:
                    logger.debug("[Executor] Graph paused at PAUSED vertex (waiting for external approval)")
                    break
                self._log_state_dump()
                msg = "Deadlock – no READY/PROCESSING vertices but graph not settled"
                logger.error("[Executor] %s", msg)
                self._result.errors.append(msg)
                for t in pending:
                    t.cancel()
                break

    async def _process_ready_vertex(self, vertex: Vertex):
        """Process a vertex that is already in READY state (loop re-entry)."""
        if vertex.state == VertexState.READY:
            vertex.state = VertexState.AWAITING_EDGES
            await self._process_vertex(vertex)
        elif vertex.state == VertexState.ABORTED:
            await self._abort_vertex(vertex)

    async def _abort_vertex(self, vertex: Vertex):
        """Cascade abort to all outgoing edges of an aborted vertex."""
        logger.debug("[Executor] Vertex '%s' aborted → cascading to outgoing edges", vertex.id)
        self._emit("vertex_state_changed", vertex_id=vertex.id, payload={"state": "aborted", "reason": vertex.abort_reason})
        outgoing = self.graph.get_outgoing_edges(vertex.id)
        for edge in outgoing:
            edge.aborted = True
            edge.abort_reason = f"Upstream vertex '{vertex.id}' was aborted"
            self._result.edge_results[edge.id] = f"<ABORTED: {edge.abort_reason}>"
            self._emit("edge_aborted", edge_id=edge.id, payload={"reason": edge.abort_reason})
            dst = self.graph.vertices[edge.destination_id]
            await dst.receive_signal(edge.id, EdgeSignal.ABORTED, payload=edge.abort_reason)

    async def _process_vertex(self, vertex: Vertex):
        """Execute compute/subgraph lifecycle for *vertex* and fire all outgoing edges."""
        logger.debug("[Executor] Processing vertex '%s'", vertex.id)
        self._emit("vertex_state_changed", vertex_id=vertex.id, payload={"state": vertex.state.value})

        # --- Nested Sub-Graph Execution (Phase 2 & 3) ---
        from ..subgraph import SubgraphVertex
        if isinstance(vertex, SubgraphVertex):
            try:
                inner_graph = vertex.initialize_inner_graph()
                await vertex.stage_inner_inputs(inner_graph)
                
                # If parent executor is CheckpointedExecutor, create namespaced CheckpointedExecutor for subgraph
                from .checkpoint import CheckpointedExecutor
                if isinstance(self, CheckpointedExecutor):
                    subgraph_run_id = f"{self.run_id}::{vertex.id}"
                    inner_executor = CheckpointedExecutor(
                        inner_graph,
                        agents=self.agents,
                        store=self.store,
                        run_id=subgraph_run_id,
                        graph_config=vertex.graph_config if isinstance(vertex.graph_config, dict) else None,
                        max_concurrency=self.max_concurrency,
                        scan_interval=self.scan_interval,
                    )
                else:
                    inner_executor = Executor(
                        inner_graph,
                        agents=self.agents,
                        max_concurrency=self.max_concurrency,
                        scan_interval=self.scan_interval,
                    )

                # Bubble up events from inner executor stream to parent queue in real-time
                async for inner_event in inner_executor.stream():
                    namespaced_vid = f"{vertex.id}.{inner_event.vertex_id}" if inner_event.vertex_id else vertex.id
                    namespaced_eid = f"{vertex.id}.{inner_event.edge_id}" if inner_event.edge_id else None
                    self._emit(
                        event_type=f"subgraph_{inner_event.event_type}",
                        vertex_id=namespaced_vid,
                        edge_id=namespaced_eid,
                        payload=inner_event.payload,
                    )

                inner_result = inner_executor._result
                if not inner_result.success:
                    err_msg = f"Inner graph execution failed: {'; '.join(inner_result.errors)}"
                    vertex.state = VertexState.ERROR
                    vertex.error_message = err_msg
                    self._result.errors.append(f"Vertex '{vertex.id}': {err_msg}")
                    self._emit("vertex_state_changed", vertex_id=vertex.id, payload={"state": "error", "error": err_msg})
                    return

                await vertex.collect_inner_outputs(inner_graph)
            except Exception as exc:
                logger.error("[Executor] Subgraph '%s' error: %s", vertex.id, exc, exc_info=True)
                vertex.state = VertexState.ERROR
                vertex.error_message = str(exc)
                self._result.errors.append(f"Vertex '{vertex.id}': {exc}")
                self._emit("vertex_state_changed", vertex_id=vertex.id, payload={"state": "error", "error": str(exc)})
                return

        # Run on_ready hook to consolidate data for outgoing reads (or for final sink output)
        try:
            await vertex.prepare_outputs()
        except Exception as exc:
            vertex.state = VertexState.ERROR
            vertex.error_message = f"prepare_outputs failed: {exc}"
            self._result.errors.append(f"Vertex '{vertex.id}': {exc}")
            self._emit("vertex_state_changed", vertex_id=vertex.id, payload={"state": "error", "error": str(exc)})
            return

        outgoing = self.graph.get_outgoing_edges(vertex.id)
        if not outgoing:
            vertex.state = VertexState.DONE
            logger.debug("[Executor] Vertex '%s' is a sink → DONE", vertex.id)
            self._emit("vertex_state_changed", vertex_id=vertex.id, payload={"state": "done"})
            return

        # Fire edges concurrently
        edge_tasks = []
        for e in outgoing:
            task = asyncio.create_task(self._fire_edge(e), name=f"edge_{e.id}")
            self.active_edge_tasks[e.id] = task
            edge_tasks.append(task)
            
        results = await asyncio.gather(*edge_tasks, return_exceptions=True)

        ok = True
        for edge, res in zip(outgoing, results):
            if isinstance(res, Exception):
                logger.error("[Executor] Edge '%s' error: %s", edge.id, res)
                self._result.errors.append(f"Edge '{edge.id}': {res}")
                ok = False

        if ok:
            # A concurrent loop-back edge may have already reset this vertex
            # to READY (re-entry) while we were in asyncio.gather.
            # If so, honour the re-entry — do NOT overwrite with DONE.
            if vertex.state == VertexState.AWAITING_EDGES:
                vertex.state = VertexState.DONE
                logger.debug("[Executor] Vertex '%s' → DONE", vertex.id)
                self._emit("vertex_state_changed", vertex_id=vertex.id, payload={"state": "done"})
            else:
                logger.debug(
                    "[Executor] Vertex '%s' gather done; state already=%s (loop re-entry detected)",
                    vertex.id, vertex.state.value,
                )
                self._emit("vertex_state_changed", vertex_id=vertex.id, payload={"state": vertex.state.value})
        else:
            vertex.state = VertexState.ERROR
            vertex.error_message = "One or more outgoing edges failed"
            logger.error("[Executor] Vertex '%s' → ERROR", vertex.id)
            self._emit("vertex_state_changed", vertex_id=vertex.id, payload={"state": "error"})

    async def _fire_edge(self, edge: Edge) -> Any:
        """Execute one edge, bounded by the per-type concurrency semaphore."""
        edge_type = edge.concurrency_type
        semaphore = self._semaphores.get(edge_type, self._semaphores["default"])
        async with semaphore:
            self._emit("edge_started", edge_id=edge.id, payload={"source": edge.source_id, "destination": edge.destination_id})
            src = self.graph.vertices[edge.source_id]
            dst = self.graph.vertices[edge.destination_id]
            result = await edge.execute(
                src,
                dst,
                self.agents,
                memory=self.memory,
                telemetry=self.telemetry,
            )
            edge_metrics = self.telemetry.edge_metrics.get(edge.id)
            metrics_payload = edge_metrics.to_dict() if edge_metrics else None

            if edge.aborted:
                self._result.edge_results[edge.id] = f"<ABORTED: {edge.abort_reason}>"
                self._emit("edge_aborted", edge_id=edge.id, payload={"reason": edge.abort_reason})
            else:
                self._result.edge_results[edge.id] = result
                self._emit("edge_completed", edge_id=edge.id, payload={
                    "result": repr(result)[:100],
                    "telemetry": metrics_payload,
                })
            return result

    async def _collect_results(self):
        """Snapshot every vertex's final state, data, telemetry metrics, and memory state."""
        self._result.metrics = self.telemetry.get_total_metrics()
        self._result.memory_snapshot = await self.memory.get_all()

        for v in self.graph.vertices.values():
            data = await v.get_all_data()
            self._result.vertex_results[v.id] = {
                "state": v.state.value,
                "data": {
                    str(k): val for k, val in data.items()
                },
                "error": v.error_message,
                "abort_reason": v.abort_reason,
                "iterations": v.iteration_count,
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
                "  [%s] state=%s  in=%d/%d  out=%s  iter=%d",
                v.id, v.state.value,
                v._received_input_count, v.required_input_count,
                v.outgoing_edges, v.iteration_count,
            )
