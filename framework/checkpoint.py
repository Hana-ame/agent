"""Checkpoint module — Resumable execution and Human-in-the-Loop (HITL) gates.

Provides two public classes:

``CheckpointedExecutor``
    Subclass of :class:`~framework.executor.Executor` that saves a
    :class:`~framework.store.GraphSnapshot` to a :class:`~framework.store.SQLiteStateStore`
    after each vertex settles.  Supports:

    * **pause / resume** — stop mid-run and continue from the last snapshot.
    * **HITL gates** — block at a :class:`HumanGateVertex` until ``vertex.approve()``
      is called from external code.

``HumanGateVertex``
    A :class:`~framework.vertex.Vertex` subclass that pauses the executor when
    it becomes READY, waiting for a human to call ``vertex.approve()``.

Typical usage::

    store = SQLiteStateStore("runs.db")
    g     = Graph.from_dict(config)
    run   = CheckpointedExecutor(g, agent, store=store, run_id="r1")
    result = await run.run()

    # --- resume after crash ---
    g2     = Graph.from_dict(config)
    result = await CheckpointedExecutor.resume("r1", g2, agent, store=store)

    # --- HITL gate ---
    hitl_v = g.vertices["review"]
    task   = asyncio.create_task(run.run())
    # ... human reviews data ...
    hitl_v.approve()
    result = await task
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional

from .executor import Executor, ExecutionResult
from .graph import Graph
from .agents import BaseAgent
from .store import SQLiteStateStore, GraphSnapshot
from .vertex import Vertex, VertexState

logger = logging.getLogger("vertex_edge_agent.checkpoint")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialize_value(val: Any) -> Any:
    """Return a JSON-safe representation of *val*."""
    try:
        json.dumps(val)
        return val
    except (TypeError, ValueError):
        return str(val)


def _snapshot_vertex(v: Vertex) -> Dict:
    return {
        "state":                     v.state.value,
        "data":                      {k: _serialize_value(vv) for k, vv in v._data_store.items()},
        "error":                     v.error_message,
        "abort_reason":              v.abort_reason,
        "iteration_count":           v.iteration_count,
        "completed_incoming_edges":  list(v.completed_incoming_edges),
        "aborted_incoming_edges":    list(v.aborted_incoming_edges),
    }


def _snapshot_edge(e) -> Dict:
    return {
        "completed":    e.completed,
        "aborted":      e.aborted,
        "abort_reason": e.abort_reason,
        "error":        e.error,
        "result":       _serialize_value(e.result),
    }


# ---------------------------------------------------------------------------
# HumanGateVertex
# ---------------------------------------------------------------------------

class HumanGateVertex(Vertex):
    """A vertex that pauses execution and waits for human approval.

    The :class:`CheckpointedExecutor` detects this class and awaits
    ``_approval_event`` before firing outgoing edges.

    Lifecycle:
      1. Vertex becomes READY (all inputs received).
      2. Executor saves a checkpoint and blocks on :attr:`_approval_event`.
      3. External code calls :meth:`approve` (optionally supplying replacement data).
      4. Executor resumes, fires outgoing edges, vertex settles DONE.

    Example::

        gate = g.vertices["quality_check"]
        run_task = asyncio.create_task(executor.run())

        # ... human reviews the data ...
        gate.approve()           # or gate.approve({"decision": "accept"})
        result = await run_task
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set when waiting for external approval
        self._approval_event: asyncio.Event = asyncio.Event()
        # Set to True once approved (so resume skips the gate)
        self._approved: bool = False

    def approve(self, approved_data: Optional[Dict] = None) -> None:
        """Release the gate and allow execution to continue.

        Args:
            approved_data: Optional dict of ``{channel: value}`` to merge into
                           the vertex's data store before outgoing edges fire.
        """
        if approved_data:
            for channel, value in approved_data.items():
                self._data_store[str(channel)] = value
        self._approved = True
        self._approval_event.set()
        logger.info("[HumanGateVertex:%s] ✓ Approved", self.id)

    def reset(self):
        super().reset()
        self._approved = False
        self._approval_event.clear()

    def __repr__(self):
        status = "approved" if self._approved else "pending"
        return f"HumanGateVertex(id={self.id!r}, state={self.state.value}, approval={status})"


# ---------------------------------------------------------------------------
# CheckpointedExecutor
# ---------------------------------------------------------------------------

class CheckpointedExecutor(Executor):
    """Executor that persists a snapshot after every vertex state change.

    Args:
        graph:        The Graph to execute.
        agents:       Agent instance.
        store:        A :class:`~framework.store.SQLiteStateStore` instance.
                      Defaults to an **in-memory** store (useful for testing).
        run_id:       Unique string identifier for this run.  Auto-generated
                      (UUID4) if not supplied.
        graph_config: Optional original config dict, stored for documentation.
        _is_resume:   Internal flag — set by :meth:`resume` to skip source
                      initialisation (states are already restored).
        **kwargs:     Forwarded to :class:`~framework.executor.Executor`.
    """

    def __init__(
        self,
        graph: Graph,
        agents: Optional[BaseAgent] = None,
        *,
        store: Optional[SQLiteStateStore] = None,
        run_id: Optional[str] = None,
        graph_config: Optional[Dict] = None,
        _is_resume: bool = False,
        **kwargs,
    ):
        super().__init__(graph, agents, **kwargs)
        self.store = store or SQLiteStateStore(":memory:")
        self.run_id = run_id or str(uuid.uuid4())
        self._graph_config = graph_config
        self._is_resume = _is_resume
        self._step = 0

    # ------------------------------------------------------------------
    # Public API — run
    # ------------------------------------------------------------------
    async def run(self) -> ExecutionResult:
        if not self._is_resume:
            self.store.create_run(self.run_id, graph_config=self._graph_config)
            await self._checkpoint("start")
        else:
            logger.info(
                "[CheckpointedExecutor] Resuming run '%s' from step %d",
                self.run_id, self._step,
            )
            self.store.update_run_status(self.run_id, "running")

        result = await super().run()

        final_status = "completed" if result.success else "failed"
        self.store.update_run_status(self.run_id, final_status)
        return result

    # ------------------------------------------------------------------
    # Public API — resume (class method)
    # ------------------------------------------------------------------
    @classmethod
    async def resume(
        cls,
        run_id: str,
        graph: Graph,
        agents: Optional[BaseAgent] = None,
        *,
        store: Optional[SQLiteStateStore] = None,
        **kwargs,
    ) -> ExecutionResult:
        """Resume a paused or interrupted run from its latest checkpoint.

        Args:
            run_id: The run identifier to restore.
            graph:  A freshly-constructed Graph (same config as original run).
                    States will be overlaid from the snapshot.
            agents: Agent instance.
            store:  The store that holds the snapshots for *run_id*.
            **kwargs: Forwarded to :class:`CheckpointedExecutor`.

        Raises:
            ValueError: If no snapshot exists for *run_id*.
        """
        store = store or SQLiteStateStore()
        snap = store.load_latest_snapshot(run_id)
        if snap is None:
            raise ValueError(
                f"No snapshot found for run_id={run_id!r}. "
                "Cannot resume."
            )

        logger.info(
            "[CheckpointedExecutor] Restoring snapshot step=%d trigger=%s for run '%s'",
            snap.step, snap.trigger, run_id,
        )

        # --- Restore vertex states ---
        for vid, vs in snap.vertex_states.items():
            if vid not in graph.vertices:
                logger.warning("[Resume] Snapshot references unknown vertex '%s'", vid)
                continue
            v = graph.vertices[vid]

            raw_state = vs.get("state", "idle")
            # Treat AWAITING_EDGES as READY so it gets re-processed
            if raw_state == VertexState.AWAITING_EDGES.value:
                raw_state = VertexState.READY.value

            v._state = VertexState(raw_state)
            v._data_store = dict(vs.get("data", {}))
            v.error_message = vs.get("error")
            v.abort_reason = vs.get("abort_reason")
            v.iteration_count = vs.get("iteration_count", 0)
            v.completed_incoming_edges = set(vs.get("completed_incoming_edges", []))
            v.aborted_incoming_edges = set(vs.get("aborted_incoming_edges", []))

            # Sync the asyncio event
            if v._state in (VertexState.READY, VertexState.ABORTED, VertexState.ERROR):
                v._ready_event.set()
            else:
                v._ready_event.clear()

        # --- Recalculate readiness for IDLE vertices ---
        # A vertex might be IDLE with completed_incoming_edges already set
        # (the snapshot was taken before the vertex reached READY).
        for v in graph.vertices.values():
            if v._state == VertexState.IDLE:
                total = len(v.incoming_edges) if v.incoming_edges else v.required_input_count
                total_settled = (
                    len(v.completed_incoming_edges) + len(v.aborted_incoming_edges)
                )
                if total > 0 and total_settled >= total:
                    if v.completed_incoming_edges:
                        v._state = VertexState.READY
                        v._ready_event.set()
                    else:
                        v._state = VertexState.ABORTED
                        v._ready_event.set()

        # --- Restore edge states ---
        for eid, es in snap.edge_states.items():
            if eid not in graph.edges:
                continue
            e = graph.edges[eid]
            e.completed    = es.get("completed", False)
            e.aborted      = es.get("aborted",   False)
            e.abort_reason = es.get("abort_reason")
            e.error        = es.get("error")
            # Note: e.result is restored for the ExecutionResult only;
            # the actual edge.result is rebuilt on re-execution if needed.

        executor = cls(
            graph,
            agents,
            store=store,
            run_id=run_id,
            _is_resume=True,
            **kwargs,
        )
        executor._step = snap.step
        return await executor.run()

    # ------------------------------------------------------------------
    # Override: skip _init_sources on resume
    # ------------------------------------------------------------------
    def _init_sources(self):
        if self._is_resume:
            logger.info(
                "[CheckpointedExecutor] Resume mode — skipping _init_sources; "
                "states already restored from snapshot"
            )
            return
        super()._init_sources()

    # ------------------------------------------------------------------
    # Override: checkpoint after each vertex
    # ------------------------------------------------------------------
    async def _process_vertex(self, vertex: Vertex):
        # --- HITL gate detection ---
        if isinstance(vertex, HumanGateVertex) and not vertex._approved:
            await self._checkpoint(f"vertex:{vertex.id}:awaiting_approval")
            self.store.update_run_status(self.run_id, "awaiting_approval")
            logger.info(
                "[CheckpointedExecutor] ⏸ HITL gate at '%s' — waiting for approval "
                "(call vertex.approve() to continue)",
                vertex.id,
            )
            try:
                await asyncio.wait_for(
                    vertex._approval_event.wait(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                vertex.state = VertexState.ERROR
                vertex.error_message = (
                    f"HITL gate '{vertex.id}' timed out after {self.timeout}s"
                )
                self._result.errors.append(vertex.error_message)
                await self._checkpoint(f"vertex:{vertex.id}:approval_timeout")
                return
            self.store.update_run_status(self.run_id, "running")
            logger.info(
                "[CheckpointedExecutor] ▶ HITL gate '%s' approved — continuing",
                vertex.id,
            )

        await super()._process_vertex(vertex)
        await self._checkpoint(f"vertex:{vertex.id}:{vertex.state.value}")

    async def _abort_vertex(self, vertex: Vertex):
        await super()._abort_vertex(vertex)
        await self._checkpoint(f"vertex:{vertex.id}:aborted")

    # ------------------------------------------------------------------
    # Snapshot serialisation
    # ------------------------------------------------------------------
    async def _checkpoint(self, trigger: str) -> None:
        self._step += 1
        snap = GraphSnapshot(
            run_id=self.run_id,
            step=self._step,
            trigger=trigger,
            timestamp="",   # filled by store
            vertex_states={
                vid: _snapshot_vertex(v)
                for vid, v in self.graph.vertices.items()
            },
            edge_states={
                eid: _snapshot_edge(e)
                for eid, e in self.graph.edges.items()
            },
        )
        self.store.save_snapshot(snap)
        logger.debug(
            "[CheckpointedExecutor] ✓ Checkpoint saved  step=%d  trigger=%s",
            self._step, trigger,
        )
