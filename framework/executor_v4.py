"""V4 DAG and Concurrency-Driven Edge Executor.

Executes graph edges directly respecting max concurrency limits and topological DAG tier order.
Schedules edges based on the two-sided handshake contract and in-memory dispatch deduplication.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Type

from framework.edge_v4 import (
    CallableLLMEdgeV4,
    ChatLLMEdgeV4,
    CodeEdgeV4,
    EdgeResultV4,
    EdgeV4,
    GenerateLLMEdgeV4,
    LLMEdgeV4,
    ProcessLLMEdgeV4,
    ReflexiveEdgeV4,
)
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
    metrics_summary: Optional[Dict[str, Any]] = None

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
            "metrics_summary": self.metrics_summary,
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
        group_concurrency: Optional[Dict[str, int]] = None,
        scan_interval: float = 0.02,
        timeout: float = 120.0,
        llm_edge_cls: Optional[Type[LLMEdgeV4]] = None,
        snapshot_dir: Optional[Union[str, Path]] = None,
        snapshot_manager: Optional[Any] = None,
        enable_snapshots: bool = False,
    ):
        self.graph = graph
        self.session_id = graph.session_id
        self.store = store or VertexStoreV4(":memory:")
        self._owns_store = store is None
        self.agent = agent
        self.snapshot_dir = snapshot_dir
        if snapshot_manager is not None:
            self.snapshot_manager = snapshot_manager
            self.enable_snapshots = True
        elif snapshot_dir is not None:
            from framework.snapshot_v4 import GraphSnapshotManagerV4
            self.snapshot_manager = GraphSnapshotManagerV4(base_dir=snapshot_dir)
            self.enable_snapshots = True
        elif enable_snapshots:
            from framework.snapshot_v4 import GraphSnapshotManagerV4
            self.snapshot_manager = GraphSnapshotManagerV4(base_dir="snapshots")
            self.enable_snapshots = True
        else:
            self.snapshot_manager = None
            self.enable_snapshots = False
        self.max_concurrency = max(1, int(max_concurrency))
        self.group_concurrency: Dict[str, int] = {k: max(1, int(v)) for k, v in (group_concurrency or {}).items()}
        self.scan_interval = scan_interval
        self.timeout = timeout

        # Resolve designated LLM edge subclass for execution
        if llm_edge_cls is not None:
            self.llm_edge_cls = llm_edge_cls
        elif agent is not None:
            if hasattr(agent, "chat") and callable(agent.chat):
                self.llm_edge_cls = ChatLLMEdgeV4
            elif hasattr(agent, "process") and callable(agent.process):
                self.llm_edge_cls = ProcessLLMEdgeV4
            elif hasattr(agent, "generate") and callable(agent.generate):
                self.llm_edge_cls = GenerateLLMEdgeV4
            elif callable(agent):
                self.llm_edge_cls = CallableLLMEdgeV4
            else:
                self.llm_edge_cls = ChatLLMEdgeV4
        else:
            self.llm_edge_cls = ChatLLMEdgeV4

        # Bind base LLM edges in graph to the designated execution subclass
        for edge_id, edge in list(self.graph.edges.items()):
            if type(edge) is LLMEdgeV4:
                self.graph.edges[edge_id] = self.llm_edge_cls.from_base(edge)
                if self.agent and hasattr(self.graph.edges[edge_id], "agent"):
                    self.graph.edges[edge_id].agent = self.agent

        # Group and per-edge semaphores for granular concurrency control
        self._group_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._edge_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.active_group_counts: Dict[str, int] = defaultdict(int)
        self.active_edge_counts: Dict[str, int] = defaultdict(int)

        # In-memory in-flight task tracking: (session_id, input_vertex, output_vertex)
        self.active_dispatches: Set[Tuple[str, str, str]] = set()
        self._event_queue: asyncio.Queue[Optional[GraphEventV4]] = asyncio.Queue()
        self._result = ExecutionResultV4(session_id=self.session_id)
        
        self._fan_in_counts: Dict[str, int] = defaultdict(int)
        self._fan_in_failures: Dict[str, int] = defaultdict(int)
        self._fan_in_completed: Dict[str, Set[str]] = defaultdict(set)
        self._scheduler_event = asyncio.Event()
        # P2: Track running tasks by output vertex for cancellation on reentry
        self._running_tasks_by_output: Dict[str, Set[asyncio.Task]] = defaultdict(set)
        # P3: Lease tracking for crash recovery (edge_id -> expiry timestamp)
        self._dispatch_leases: Dict[str, float] = {}

    def _notify_scheduler(self) -> None:
        """Wake up the scheduler loop if it's waiting."""
        self._scheduler_event.set()

    def _save_snapshot(self, trigger: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Save a complete graph snapshot if snapshot recording is enabled."""
        if not self.enable_snapshots or not self.snapshot_manager:
            return None
        try:
            return self.snapshot_manager.save_snapshot(
                graph=self.graph,
                trigger=trigger,
                store=self.store,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning("[ExecutorV4] Failed saving snapshot for trigger '%s': %s", trigger, e)
            return None

    # ------------------------------------------------------------------
    # P2: Cancel in-flight tasks targeting downstream vertices on reentry
    # ------------------------------------------------------------------

    def cancel_downstream_tasks(self, vertex_names: Set[str]) -> List[str]:
        """Cancel all in-flight tasks whose output vertex is in *vertex_names*.

        Called by the server's reenter_vertex route to prevent stale results
        from stomping on freshly-reset vertex states.

        Returns a list of cancelled edge IDs.
        """
        cancelled: List[str] = []
        for vname in vertex_names:
            self._fan_in_completed.pop(vname, None)
            self._fan_in_counts.pop(vname, None)
            self._fan_in_failures.pop(vname, None)
            tasks = self._running_tasks_by_output.get(vname, set())
            for task in list(tasks):
                if not task.done():
                    edge_name = task.get_name() or ""
                    edge_id = edge_name.replace("edge_", "", 1) if edge_name else "?"
                    cancelled.append(edge_id)
                    task.cancel()
        return cancelled

    # ------------------------------------------------------------------
    # P3: Recover stale dispatch leases after process restart
    # ------------------------------------------------------------------

    def recover_stale_leases(self) -> List[str]:
        """Identify and clear stale dispatch leases from a previous process.

        After a crash/restart, ``active_dispatches`` is empty but the SQLite
        store still has upstream ``data ready`` and downstream ``todo``.
        This method returns edge IDs whose leases have expired so the caller
        can decide whether to re-dispatch or mark as failed.

        Returns a list of stale edge IDs.
        """
        now = time.monotonic()
        stale = [eid for eid, expiry in self._dispatch_leases.items() if expiry < now]
        for eid in stale:
            self._dispatch_leases.pop(eid, None)
        return stale

    @property
    def result(self) -> ExecutionResultV4:
        return self._result

    def _get_group_semaphore(self, group: str) -> Optional[asyncio.Semaphore]:
        """Fetch or instantiate the concurrency semaphore for a named edge group."""
        if group in self.group_concurrency:
            if group not in self._group_semaphores:
                self._group_semaphores[group] = asyncio.Semaphore(self.group_concurrency[group])
            return self._group_semaphores[group]
        return None

    def _get_edge_semaphore(self, edge: EdgeV4) -> Optional[asyncio.Semaphore]:
        """Fetch or instantiate a dedicated concurrency semaphore for a specific edge."""
        limit = edge.concurrency_limit
        if limit is not None and limit > 0:
            if edge.id not in self._edge_semaphores:
                self._edge_semaphores[edge.id] = asyncio.Semaphore(limit)
            return self._edge_semaphores[edge.id]
        return None

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

    def _get_eligible_edges(self) -> List[Tuple[int, int, int, EdgeV4]]:
        """Identify edges whose trigger prerequisites are met, sorted by priority and DAG tier.

        Priority order:
          1. Urgency rank (0: reflexive recovery, 1: todo urgent, 2: todo)
          2. Edge priority descending (-edge.priority)
          3. DAG topological tier (Tier 0 executes before Tier 1)
          4. Edge ID ascending
        """
        candidates: List[Tuple[int, int, int, EdgeV4]] = []

        for edge in self.graph.edges.values():
            key = (self.session_id, edge.input_vertex, edge.output_vertex)
            if key in self.active_dispatches:
                # Already executing in-flight
                continue

            if edge.id in self._fan_in_completed[edge.output_vertex]:
                # Already executed for the current fan-in cycle
                continue

            tier = self.graph.edge_tiers.get(edge.id, 0)
            edge_prio = edge.priority

            if hasattr(self.graph, "is_vertex_active"):
                if not self.graph.is_vertex_active(edge.input_vertex) or not self.graph.is_vertex_active(edge.output_vertex):
                    continue

            if edge.is_reflexive or isinstance(edge, ReflexiveEdgeV4):
                # Check reflexive trigger state
                target_v = self.store.get_vertex(self.session_id, edge.output_vertex)
                trigger_state = getattr(edge, "trigger_state", VertexStateV4.REJECT.value)
                if target_v and target_v.state == trigger_state:
                    # Urgency 0 for recovery
                    candidates.append((0, -edge_prio, -1, edge))
            else:
                # Check two-sided handshake contract
                in_v = self.store.get_vertex(self.session_id, edge.input_vertex)
                out_v = self.store.get_vertex(self.session_id, edge.output_vertex)
                if in_v and out_v:
                    upstream_ready = in_v.state == VertexStateV4.DATA_READY.value
                    if upstream_ready:
                        if out_v.state == VertexStateV4.TODO_URGENT.value:
                            candidates.append((1, -edge_prio, tier, edge))
                        elif out_v.state == VertexStateV4.TODO.value:
                            candidates.append((2, -edge_prio, tier, edge))

        # Sort: urgency rank, edge priority descending, DAG tier ascending, edge_id
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].id))
        return candidates

    async def _execute_single_edge(
        self,
        edge: EdgeV4,
        semaphore: asyncio.Semaphore,
    ) -> EdgeResultV4:
        """Run an edge bounded by multi-level concurrency semaphores and manage active_dispatches."""
        key = (self.session_id, edge.input_vertex, edge.output_vertex)
        group_sem = self._get_group_semaphore(edge.concurrency_group)
        edge_sem = self._get_edge_semaphore(edge)
        edge_timeout = edge.timeout or self.timeout
        start_time = time.perf_counter()

        try:
            self._emit("edge_started", edge_id=edge.id, payload={"input": edge.input_vertex, "output": edge.output_vertex})

            async with AsyncExitStack() as stack:
                # Acquire global semaphore
                await stack.enter_async_context(semaphore)
                # Acquire group semaphore if present
                if group_sem:
                    await stack.enter_async_context(group_sem)
                # Acquire per-edge semaphore if present
                if edge_sem:
                    await stack.enter_async_context(edge_sem)

                try:
                    edge_to_run = self.llm_edge_cls.from_base(edge) if type(edge) is LLMEdgeV4 else edge
                    participating_fan_in = [
                        e for e in self.graph.get_incoming_edges(edge.output_vertex)
                        if not (e.is_reflexive or isinstance(e, ReflexiveEdgeV4))
                        and (
                            e.settings.get("merge_strategy", "overwrite") in ("json_merge", "list_append", "reducer_script")
                            or e.settings.get("settlement_barrier", False)
                        )
                    ]
                    expected = len(participating_fan_in)
                    is_fan_in = expected > 1
                    try:
                        res = await asyncio.wait_for(
                            edge_to_run.run(
                                session_id=self.session_id,
                                store=self.store,
                                agent=self.agent,
                                auto_transition=(not is_fan_in),
                            ),
                            timeout=edge_timeout,
                        )
                    except TypeError:
                        res = await asyncio.wait_for(
                            edge_to_run.run(
                                session_id=self.session_id,
                                store=self.store,
                                agent=self.agent,
                            ),
                            timeout=edge_timeout,
                        )
                except asyncio.TimeoutError:
                    err_msg = f"Edge '{edge.id}' execution timed out after {edge_timeout}s"
                    logger.error("[ExecutorV4] %s", err_msg)
                    self.store.stage_output(
                        session_id=self.session_id,
                        edge_id=edge.id,
                        key="error_feedback",
                        value=err_msg,
                        vertex_name=edge.output_vertex,
                        metadata={"error": "timeout", "timeout": edge_timeout},
                    )
                    self.store.update_vertex_state(self.session_id, edge.output_vertex, VertexStateV4.REJECT.value)
                    res = EdgeResultV4(
                        edge_id=edge.id,
                        success=False,
                        error=err_msg,
                        reason="Edge execution timeout",
                    )
                except Exception as exc:
                    err_msg = str(exc)
                    logger.error("[ExecutorV4] Unhandled edge error in '%s': %s", edge.id, err_msg)
                    res = EdgeResultV4(
                        edge_id=edge.id,
                        success=False,
                        error=err_msg,
                        reason="Unhandled edge exception",
                    )
        finally:
            grp = edge.concurrency_group
            if grp in self.group_concurrency:
                self.active_group_counts[grp] = max(0, self.active_group_counts[grp] - 1)
            if edge.concurrency_limit is not None and edge.concurrency_limit > 0:
                self.active_edge_counts[edge.id] = max(0, self.active_edge_counts[edge.id] - 1)
            self.active_dispatches.discard(key)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not res.skipped:
            try:
                p_tok = int(res.metadata.get("prompt_tokens") or res.metadata.get("usage", {}).get("prompt_tokens", 0))
                c_tok = int(res.metadata.get("completion_tokens") or res.metadata.get("usage", {}).get("completion_tokens", 0))
                t_tok = int(res.metadata.get("total_tokens") or res.metadata.get("usage", {}).get("total_tokens", p_tok + c_tok))
                cost = float(res.metadata.get("cost_usd") or res.metadata.get("cost", 0.0))
                self.store.record_edge_metric(
                    session_id=self.session_id,
                    edge_id=edge.id,
                    edge_type=edge.type,
                    input_vertex=edge.input_vertex,
                    output_vertex=edge.output_vertex,
                    execution_time_ms=elapsed_ms,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=t_tok,
                    cost_usd=cost,
                    success=res.success,
                    error=res.error,
                    metadata={"reason": res.reason, **res.metadata},
                )
            except Exception as metric_err:
                logger.warning("[ExecutorV4] Failed recording metric for '%s': %s", edge.id, metric_err)

        if res.success:
            self._emit("edge_completed", edge_id=edge.id, payload={"output": res.output})

            # Fan-in accumulation and settlement barrier
            if is_fan_in and not edge.is_reflexive and not isinstance(edge, ReflexiveEdgeV4):
                self._fan_in_completed[edge.output_vertex].add(edge.id)
                self._fan_in_counts[edge.output_vertex] = len(self._fan_in_completed[edge.output_vertex])
                if len(self._fan_in_completed[edge.output_vertex]) >= expected:
                    if self._fan_in_failures[edge.output_vertex] > 0:
                        self.store.update_vertex_state(self.session_id, edge.output_vertex, VertexStateV4.REJECT.value)
                        self._emit("fan_in_failed", vertex_name=edge.output_vertex,
                                   payload={"failures": self._fan_in_failures[edge.output_vertex],
                                            "expected": expected})
                    else:
                        self.store.update_vertex_state(self.session_id, edge.output_vertex, VertexStateV4.DATA_READY.value)
                        self.store.increment_processed_count(self.session_id, edge.output_vertex)
                    self._fan_in_completed[edge.output_vertex].clear()
                    self._fan_in_counts[edge.output_vertex] = 0
                    self._fan_in_failures[edge.output_vertex] = 0
                else:
                    self.store.update_vertex_state(self.session_id, edge.output_vertex, VertexStateV4.TODO.value)

        elif not res.skipped:
            self._emit("edge_failed", edge_id=edge.id, payload={"error": res.error})

            # P0 FIX: Track failures for fan-in settlement. When all incoming
            # edges have settled and any failed, downgrade to reject to
            # prevent deadlock (instead of staying todo forever).
            if not edge.is_reflexive and not isinstance(edge, ReflexiveEdgeV4):
                self._fan_in_completed[edge.output_vertex].add(edge.id)
                self._fan_in_counts[edge.output_vertex] = len(self._fan_in_completed[edge.output_vertex])
                self._fan_in_failures[edge.output_vertex] += 1
                settlement_expected = expected if is_fan_in else len([e for e in self.graph.get_incoming_edges(edge.output_vertex) if not (e.is_reflexive or isinstance(e, ReflexiveEdgeV4))])
                if self._fan_in_counts[edge.output_vertex] >= settlement_expected:
                    if self._fan_in_failures[edge.output_vertex] > 0:
                        # At least one predecessor failed permanently → reject to
                        # allow reflexive recovery or deadlock-free termination.
                        self.store.update_vertex_state(self.session_id, edge.output_vertex, VertexStateV4.REJECT.value)
                        self._emit("fan_in_failed", vertex_name=edge.output_vertex,
                                   payload={"failures": self._fan_in_failures[edge.output_vertex],
                                            "expected": settlement_expected})
                    else:
                        self.store.update_vertex_state(self.session_id, edge.output_vertex, VertexStateV4.DATA_READY.value)
                    self._fan_in_completed[edge.output_vertex].clear()
                    self._fan_in_counts[edge.output_vertex] = 0
                    self._fan_in_failures[edge.output_vertex] = 0

        # P3: Clear dispatch lease on completion
        self._dispatch_leases.pop(edge.id, None)

        if res.success:
            self._save_snapshot(f"edge_completed:{edge.id}")
        elif not res.skipped:
            self._save_snapshot(f"edge_failed:{edge.id}")
            
        self._notify_scheduler()
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

        def _is_active(name: str) -> bool:
            if hasattr(self.graph, "is_vertex_active"):
                return self.graph.is_vertex_active(name)
            return True

        # Check end/sink vertices: active nodes tagged END or sinks with incoming edges
        end_vertices = [
            v for v in all_vertices
            if _is_active(v.name)
            and (
                v.has_attribute(VertexAttributeV4.END)
                or (
                    not [e for e in self.graph.get_outgoing_edges(v.name) if not e.is_reflexive]
                    and [e for e in self.graph.get_incoming_edges(v.name) if not e.is_reflexive]
                )
            )
        ]

        has_forbidden = any(
            s == VertexStateV4.FORBIDDEN.value
            for v_name, s in states.items()
            if _is_active(v_name)
        )

        # Success condition: All designated end vertices are 'data ready' and no forbidden nodes
        if end_vertices and all(v.state == VertexStateV4.DATA_READY.value for v in end_vertices) and not has_forbidden:
            return True, True

        # Check if any vertices are still actionable
        has_pending = any(
            s in (VertexStateV4.TODO.value, VertexStateV4.TODO_URGENT.value, VertexStateV4.REJECT.value)
            for v_name, s in states.items()
            if _is_active(v_name)
        )

        if not has_pending and not self.active_dispatches:
            # Everything settled
            is_success = (
                not has_forbidden
                and all(v.state == VertexStateV4.DATA_READY.value for v in end_vertices)
            ) if end_vertices else not has_forbidden
            return True, is_success

        # Deadlock check: No tasks running and no eligible edges available.
        # NOTE: _is_terminal() does a full edge scan per loop iteration when checking eligible edges.
        # This could be optimized with a dirty flag for large graphs.
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

    async def step(self) -> ExecutionResultV4:
        """Execute one eligible edge (one hop), then return control.

        This enables edge-by-edge execution without batching or tiers.
        """
        t0 = time.monotonic()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        self._sync_and_reload_graph_from_store()

        result = ExecutionResultV4(session_id=self.session_id)

        is_done, is_success = self._is_terminal()
        if is_done:
            result.success = is_success
            result.execution_time = time.monotonic() - t0
            for v in self.store.list_vertices(self.session_id):
                result.vertex_states[v.name] = v.state
                result.vertex_contents[v.name] = v.content
            self._result = result
            return result

        eligible = self._get_eligible_edges()
        if not eligible:
            result.success = False
            result.errors.append("No eligible edges found")
            result.execution_time = time.monotonic() - t0
            for v in self.store.list_vertices(self.session_id):
                result.vertex_states[v.name] = v.state
                result.vertex_contents[v.name] = v.content
            self._result = result
            return result

        # Execute single highest-priority eligible edge (no tier batching)
        _rank, _prio, _t, edge = eligible[0]
        completed = await self._execute_single_edge(edge, semaphore)
        result.edge_results[completed.edge_id] = completed.to_dict()
        if completed.success:
            result.completed_edges.append(completed.edge_id)
        elif not completed.skipped and completed.error:
            result.errors.append(f"[{completed.edge_id}] {completed.error}")

        is_done, is_success = self._is_terminal()
        result.success = is_success if is_done else False
        result.execution_time = time.monotonic() - t0
        for v in self.store.list_vertices(self.session_id):
            result.vertex_states[v.name] = v.state
            result.vertex_contents[v.name] = v.content
        try:
            result.metrics_summary = self.store.get_edge_metrics_summary(self.session_id)
        except Exception:
            pass
        self._result = result
        return result

    run_one_tier = step  # Backward-compatible alias

    def is_terminal(self) -> bool:
        """Public check: has the graph reached a terminal state?"""
        done, _ = self._is_terminal()
        return done

    def _sync_and_reload_graph_from_store(self) -> None:
        """Read and update execution graph from store before execution without mutating original in-memory graph."""
        if not self.store:
            return

        # 1. Seed any vertices or edges from self.graph that are not yet in store
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

        existing_edges = {er.edge_id: er for er in self.store.list_edges(self.session_id)}
        for e in self.graph.edges.values():
            er = existing_edges.get(e.id)
            script_val = getattr(e, "script", None)
            script_str = script_val if isinstance(script_val, str) else None
            if er is None or er.input_vertex != e.input_vertex or er.output_vertex != e.output_vertex:
                self.store.save_edge(
                    session_id=self.session_id,
                    edge_id=e.id,
                    edge_type=e.type,
                    input_vertex=e.input_vertex,
                    output_vertex=e.output_vertex,
                    script=script_str,
                    trigger_state=getattr(e, "trigger_state", None),
                    target_state=getattr(e, "target_state", None),
                    max_retries=getattr(e, "max_retries", 3),
                    settings=e.settings,
                )

        # 2. Read and update fresh graph from store
        fresh_graph = GraphV4.load_from_store(self.store, self.session_id, name=self.graph.name)
        for eid, fresh_e in list(fresh_graph.edges.items()):
            old_e = self.graph.edges.get(eid)
            if old_e:
                if type(old_e) not in (CodeEdgeV4, ReflexiveEdgeV4, LLMEdgeV4):
                    fresh_graph.edges[eid] = old_e
                else:
                    if callable(getattr(old_e, "script", None)):
                        fresh_e.script = old_e.script
                        if hasattr(fresh_e, "_callable"):
                            fresh_e._callable = old_e.script
                    if isinstance(old_e, LLMEdgeV4) and hasattr(old_e, "agent") and old_e.agent is not None:
                        fresh_e.agent = old_e.agent

            curr_e = fresh_graph.edges.get(eid)
            if curr_e and type(curr_e) is LLMEdgeV4 and self.llm_edge_cls:
                fresh_graph.edges[eid] = self.llm_edge_cls.from_base(curr_e)
                if self.agent and hasattr(fresh_graph.edges[eid], "agent"):
                    fresh_graph.edges[eid].agent = self.agent

        if hasattr(self.graph, "_inactive_vertices"):
            fresh_graph._inactive_vertices = set(self.graph._inactive_vertices)

        self.graph = fresh_graph

    async def _run_internal(self) -> ExecutionResultV4:
        """Core scheduler loop coordinating concurrency and DAG tier execution."""
        t0 = time.monotonic()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        running_tasks: Set[asyncio.Task] = set()

        try:
            # Reload latest graph state from store before each execution
            self._sync_and_reload_graph_from_store()

            self._emit(
                "workflow_started",
                payload={
                    "concurrency": self.max_concurrency,
                    "session_id": self.session_id,
                },
            )
            self._save_snapshot("execution_start")
            while True:
                # 1. Dispatch eligible edges respecting priority, group limits, and max_concurrency
                eligible = self._get_eligible_edges()

                for _rank, _prio, _tier, edge in eligible:
                    if len(running_tasks) >= self.max_concurrency:
                        break
                    key = (self.session_id, edge.input_vertex, edge.output_vertex)
                    if key in self.active_dispatches:
                        continue

                    # Check group concurrency limit
                    grp = edge.concurrency_group
                    if grp in self.group_concurrency and self.active_group_counts[grp] >= self.group_concurrency[grp]:
                        continue

                    # Check per-edge concurrency limit
                    if edge.concurrency_limit is not None and edge.concurrency_limit > 0:
                        if self.active_edge_counts[edge.id] >= edge.concurrency_limit:
                            continue

                    # Mark in-flight before task creation to prevent scheduling races
                    self.active_dispatches.add(key)
                    # P3: Record dispatch lease for crash recovery
                    self._dispatch_leases[edge.id] = time.monotonic() + (edge.timeout or self.timeout)
                    if grp in self.group_concurrency:
                        self.active_group_counts[grp] += 1
                    if edge.concurrency_limit is not None and edge.concurrency_limit > 0:
                        self.active_edge_counts[edge.id] += 1

                    task = asyncio.create_task(
                        self._execute_single_edge(edge, semaphore),
                        name=f"edge_{edge.id}",
                    )
                    # P2: Track task by output vertex for cancellation on reentry
                    self._running_tasks_by_output[edge.output_vertex].add(task)
                    task.add_done_callback(
                        lambda t, ov=edge.output_vertex: self._running_tasks_by_output[ov].discard(t)
                    )
                    running_tasks.add(task)

                # 2. Wait for either task completion or timeout
                if running_tasks:
                    done, pending = await asyncio.wait(
                        running_tasks,
                        timeout=self.scan_interval,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    running_tasks = set(pending)
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
                    self._scheduler_event.clear()
                    try:
                        await asyncio.wait_for(self._scheduler_event.wait(), timeout=self.scan_interval)
                    except asyncio.TimeoutError:
                        pass

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
            try:
                self._result.metrics_summary = self.store.get_edge_metrics_summary(self.session_id)
            except Exception:
                pass
            self._save_snapshot("execution_finished")
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


# Deprecated: use ExecutorV4 directly. OrchestratorV4 is a backward-compatibility alias.
OrchestratorV4 = ExecutorV4
