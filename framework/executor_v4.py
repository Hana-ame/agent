"""V4 DAG and Concurrency-Driven Edge Executor.

Executes graph edges directly respecting max concurrency limits and topological DAG tier order.
Schedules edges based on the two-sided handshake contract and in-memory dispatch deduplication.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from framework.edge_v4 import EdgeResultV4, EdgeV4, ReflexiveEdgeV4
from framework.graph_v4 import GraphV4
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.executor_v4")


@dataclass
class ExecutionResultV4:
    """Consolidated outcome of an ExecutorV4 run."""

    session_id: str
    success: bool = False
    execution_time: float = 0.0
    completed_edges: List[str] = field(default_factory=list)
    edge_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    vertex_states: Dict[str, str] = field(default_factory=dict)
    vertex_contents: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize execution outcome to dictionary."""
        return {
            "session_id": self.session_id,
            "success": self.success,
            "execution_time": self.execution_time,
            "completed_edges": self.completed_edges,
            "edge_results": self.edge_results,
            "vertex_states": self.vertex_states,
            "vertex_contents": self.vertex_contents,
            "errors": self.errors,
        }


@dataclass
class GraphEventV4:
    """Real-time observability event emitted during orchestration."""

    event_type: str
    session_id: str
    edge_id: Optional[str] = None
    vertex_name: Optional[str] = None
    payload: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)


class ExecutorV4:
    """Executes edges directly constrained by max concurrency and DAG topological order."""

    def __init__(
        self,
        graph: GraphV4,
        store: Optional[VertexStoreV4] = None,
        agent: Optional[Any] = None,
        max_concurrency: int = 4,
        scan_interval: float = 0.02,
        timeout: float = 120.0,
    ):
        self.graph = graph
        self.session_id = graph.session_id
        self.store = store or VertexStoreV4(":memory:")
        self._owns_store = store is None
        self.agent = agent
        self.max_concurrency = max(1, int(max_concurrency))
        self.scan_interval = scan_interval
        self.timeout = timeout

        # In-memory in-flight task tracking: (session_id, input_vertex, output_vertex)
        self.active_dispatches: Set[Tuple[str, str, str]] = set()
        self._event_queue: asyncio.Queue[Optional[GraphEventV4]] = asyncio.Queue()
        self._result = ExecutionResultV4(session_id=self.session_id)

    def _emit(
        self,
        event_type: str,
        edge_id: Optional[str] = None,
        vertex_name: Optional[str] = None,
        payload: Optional[Any] = None,
    ) -> None:
        """Emit asynchronous lifecycle event to queue."""
        ev = GraphEventV4(
            event_type=event_type,
            session_id=self.session_id,
            edge_id=edge_id,
            vertex_name=vertex_name,
            payload=payload,
        )
        self._event_queue.put_nowait(ev)

    def _get_eligible_edges(self) -> List[Tuple[int, int, EdgeV4]]:
        """Identify edges whose trigger prerequisites are met, sorted by priority and DAG tier.

        Priority order:
          0: Reflexive recovery edges (urgent handling of reject state)
          1: Forward edges targeting 'todo urgent'
          2: Forward edges targeting standard 'todo'

        Secondary order:
          DAG topological tier (Tier 0 executes before Tier 1)
        """
        candidates: List[Tuple[int, int, EdgeV4]] = []

        for edge in self.graph.edges.values():
            key = (self.session_id, edge.input_vertex, edge.output_vertex)
            if key in self.active_dispatches:
                # Already executing in-flight
                continue

            tier = self.graph.edge_tiers.get(edge.id, 0)

            if edge.is_reflexive or isinstance(edge, ReflexiveEdgeV4):
                # Check reflexive trigger state
                target_v = self.store.get_vertex(self.session_id, edge.output_vertex)
                trigger_state = getattr(edge, "trigger_state", VertexStateV4.REJECT.value)
                if target_v and target_v.state == trigger_state:
                    # Priority 0 for recovery
                    candidates.append((0, -1, edge))
            else:
                # Check two-sided handshake contract
                in_v = self.store.get_vertex(self.session_id, edge.input_vertex)
                out_v = self.store.get_vertex(self.session_id, edge.output_vertex)
                if in_v and out_v:
                    upstream_ready = in_v.state == VertexStateV4.DATA_READY.value
                    if upstream_ready:
                        if out_v.state == VertexStateV4.TODO_URGENT.value:
                            candidates.append((1, tier, edge))
                        elif out_v.state == VertexStateV4.TODO.value:
                            candidates.append((2, tier, edge))

        # Sort: priority ascending, then DAG tier ascending, then edge_id
        candidates.sort(key=lambda item: (item[0], item[1], item[2].id))
        return candidates

    async def _execute_single_edge(
        self,
        edge: EdgeV4,
        semaphore: asyncio.Semaphore,
    ) -> EdgeResultV4:
        """Run an edge bounded by concurrency semaphore and manage active_dispatches."""
        key = (self.session_id, edge.input_vertex, edge.output_vertex)
        # NOTE: active_dispatches.add(key) is done at the scheduling site, not here
        try:
            self._emit("edge_started", edge_id=edge.id, payload={"input": edge.input_vertex, "output": edge.output_vertex})

            async with semaphore:
                try:
                    res = await edge.run(
                        session_id=self.session_id,
                        store=self.store,
                        agent=self.agent,
                    )
                except Exception as exc:
                    logger.error("[ExecutorV4] Unhandled edge error in '%s': %s", edge.id, exc)
                    res = EdgeResultV4(
                        edge_id=edge.id,
                        success=False,
                        error=str(exc),
                        reason="Unhandled edge exception",
                    )
        finally:
            self.active_dispatches.discard(key)

        if res.success:
            self._emit("edge_completed", edge_id=edge.id, payload={"output": res.output})
        elif not res.skipped:
            self._emit("edge_failed", edge_id=edge.id, payload={"error": res.error})

        return res

    def _is_terminal(self) -> Tuple[bool, bool]:
        """Check if graph execution has reached completion or deadlocked.

        Returns:
            (is_done, is_success)
        """
        all_vertices = self.store.list_vertices(self.session_id)
        if not all_vertices:
            return True, True

        states = {v.name: v.state for v in all_vertices}

        # Check end/sink vertices
        end_vertices = [
            v for v in all_vertices
            if v.has_attribute(VertexAttributeV4.END)
            or not [e for e in self.graph.get_outgoing_edges(v.name) if not e.is_reflexive]
        ]

        has_forbidden = any(s == VertexStateV4.FORBIDDEN.value for s in states.values())

        # Success condition: All designated end vertices are 'data ready' and no forbidden nodes
        if end_vertices and all(v.state == VertexStateV4.DATA_READY.value for v in end_vertices) and not has_forbidden:
            return True, True

        # Check if any vertices are still actionable
        has_pending = any(
            s in (VertexStateV4.TODO.value, VertexStateV4.TODO_URGENT.value, VertexStateV4.REJECT.value)
            for s in states.values()
        )

        if not has_pending and not self.active_dispatches:
            # Everything settled
            is_success = (
                not has_forbidden
                and all(v.state == VertexStateV4.DATA_READY.value for v in end_vertices)
            ) if end_vertices else not has_forbidden
            return True, is_success

        # Deadlock check: No tasks running and no eligible edges available
        eligible = self._get_eligible_edges()
        if not self.active_dispatches and not eligible:
            logger.warning("[ExecutorV4] Deadlock detected: pending demands exist but no edges eligible.")
            return True, False

        return False, False

    async def stream(self) -> AsyncGenerator[GraphEventV4, None]:
        """Stream execution events asynchronously as they occur."""
        run_task = asyncio.create_task(self._run_internal())
        try:
            while True:
                ev = await self._event_queue.get()
                if ev is None:
                    break
                yield ev
        finally:
            if not run_task.done():
                run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

    async def run(self) -> ExecutionResultV4:
        """Run workflow to completion and return result."""
        async for _ in self.stream():
            pass
        return self._result

    async def _run_internal(self) -> ExecutionResultV4:
        """Core scheduler loop coordinating concurrency and DAG tier execution."""
        t0 = time.monotonic()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        running_tasks: Set[asyncio.Task] = set()

        try:
            # Seed initial graph vertices if not already in store
            for v in self.graph.vertices.values():
                if self.store.get_vertex(self.session_id, v.name) is None:
                    self.store.save_vertex(
                        session_id=self.session_id,
                        name=v.name,
                        content=v.content,
                        attributes=v.attributes,
                        state=v.state,
                        processed_count=v.processed_count,
                    )

            self._emit(
                "workflow_started",
                payload={
                    "concurrency": self.max_concurrency,
                    "session_id": self.session_id,
                },
            )
            while True:
                # 1. Dispatch eligible edges up to max_concurrency
                eligible = self._get_eligible_edges()
                available_slots = max(0, self.max_concurrency - len(running_tasks))

                for _prio, _tier, edge in eligible[:available_slots]:
                    key = (self.session_id, edge.input_vertex, edge.output_vertex)
                    if key not in self.active_dispatches:
                        self.active_dispatches.add(key)
                        task = asyncio.create_task(
                            self._execute_single_edge(edge, semaphore),
                            name=f"edge_{edge.id}",
                        )
                        running_tasks.add(task)

                # 2. Wait for either task completion or timeout
                if running_tasks:
                    done, running_tasks = await asyncio.wait(
                        running_tasks,
                        timeout=self.scan_interval,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for completed_task in done:
                        if completed_task.cancelled():
                            continue
                        try:
                            edge_res: EdgeResultV4 = completed_task.result()
                            self._result.edge_results[edge_res.edge_id] = edge_res.to_dict()
                            if edge_res.success:
                                self._result.completed_edges.append(edge_res.edge_id)
                            elif not edge_res.skipped and edge_res.error:
                                self._result.errors.append(f"[{edge_res.edge_id}] {edge_res.error}")
                        except Exception as t_err:
                            self._result.errors.append(str(t_err))
                else:
                    # Brief sleep before re-checking conditions
                    await asyncio.sleep(self.scan_interval)

                # 3. Check termination conditions
                is_done, is_success = self._is_terminal()
                if is_done:
                    self._result.success = is_success
                    break

                # 4. Timeout check
                if (time.monotonic() - t0) > self.timeout:
                    timeout_msg = f"Workflow timed out after {self.timeout}s"
                    self._result.errors.append(timeout_msg)
                    self._result.success = False
                    break

        finally:
            # Cancel any remaining tasks and await them to ensure clean shutdown
            for t in running_tasks:
                if not t.done():
                    t.cancel()
            if running_tasks:
                await asyncio.gather(*running_tasks, return_exceptions=True)

            # Record final vertex snapshots
            for v in self.store.list_vertices(self.session_id):
                self._result.vertex_states[v.name] = v.state
                self._result.vertex_contents[v.name] = v.content

            self._result.execution_time = time.monotonic() - t0
            self._emit(
                "workflow_finished",
                payload={
                    "success": self._result.success,
                    "execution_time": self._result.execution_time,
                },
            )
            # Sentinel to close event stream
            self._event_queue.put_nowait(None)

            if self._owns_store:
                self.store.close()

        return self._result


# Alias for backward compatibility
OrchestratorV4 = ExecutorV4
