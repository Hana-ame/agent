"""Vertex module - Node in the computation graph.

A Vertex stores data keyed by channel strings, has a state machine
for lifecycle management, and supports external Python scripts for data
handling, validation, and rejection.

Stateful loop support: vertices that receive signals from loop-back edges
(``loop_incoming_edges``) can re-enter READY state on successive iterations,
bounded by ``max_iterations`` per loop edge.
"""

import asyncio
import enum
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("vertex_edge_agent.vertex")


class VertexState(enum.Enum):
    """States for vertex lifecycle."""
    IDLE = "idle"                # Waiting for inputs
    READY = "ready"              # All inputs received, ready to process
    PAUSED = "paused"            # Intercepted waiting for human approval / intervention
    AWAITING_EDGES = "awaiting_edges"    # Outgoing edges being fired
    DONE = "done"                # All processing complete
    ABORTED = "aborted"          # Pruned or all inputs aborted
    ERROR = "error"              # Error occurred


class EdgeSignal(str, enum.Enum):
    """Signals exchanged between Edge and Vertex."""
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"



from .utils.errors import DataRejectedError



class InvalidTransition(Exception):
    pass


class StateMachine:
    """Declarative state machine with validated transitions."""

    # 只列真实生命周期流转；表外一律拒绝。
    # reset / 测试搭初始态 / 快照恢复走 force_state，不走这张表。
    TRANSITIONS = {
        # 等输入：可能收齐进 READY，也可能被要求审批（PAUSED），
        # 入边全被剪枝（ABORTED），或上游边直接失败（ERROR）。
        # AWAITING_EDGES 给无入边的源节点——executor 直接拉它开火。
        VertexState.IDLE: {
            VertexState.READY, VertexState.PAUSED,
            VertexState.AWAITING_EDGES, VertexState.ABORTED, VertexState.ERROR,
        },
        VertexState.AWAITING_EDGES: {
            VertexState.READY, VertexState.ABORTED, VertexState.ERROR, VertexState.DONE,
        },
        VertexState.READY: {
            VertexState.DONE, VertexState.PAUSED, VertexState.ERROR, VertexState.AWAITING_EDGES,
        },
        VertexState.PAUSED: {VertexState.READY, VertexState.ERROR},
        # DONE -> READY 是有界循环重入的关键：回边把 DONE 的顶点再拉起来。
        # 少了这条整个 loops 特性直接炸（done -> ready is not allowed）。
        VertexState.DONE: {VertexState.READY, VertexState.IDLE},
        VertexState.ABORTED: {VertexState.IDLE},
        # ERROR -> IDLE 让 reset() 能从任何终态干净回收。
        VertexState.ERROR: {VertexState.IDLE},
    }

    def __set_name__(self, owner, name):
        self._name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._name, VertexState.IDLE)

    def __set__(self, obj, new_state: VertexState):
        current = getattr(obj, self._name, VertexState.IDLE)
        if new_state != current and new_state not in self.TRANSITIONS.get(current, set()):
            raise InvalidTransition(
                f"Vertex[{obj.id}]: {current.value} -> {new_state.value} is not allowed"
            )
        logger.debug("Vertex[%s] %s -> %s", obj.id, current.value, new_state.value)
        self._commit(obj, new_state)

    def force_state(self, obj, new_state: VertexState):
        """Bypass validation for reset / recovery / snapshot restore / test setup."""
        self._commit(obj, new_state)

    @staticmethod
    def _commit(obj, new_state: VertexState):
        setattr(obj, "_state", new_state)
        if new_state in (VertexState.READY, VertexState.ABORTED, VertexState.ERROR):
            obj._ready_event.set()
        else:
            obj._ready_event.clear()

class Vertex:
    state = StateMachine()
    """A vertex (node) in the computation graph.

    Stores data keyed by channel strings.
    Has a state machine for lifecycle management.
    Supports external scripts for data handling/validation/rejection.
    Supports stateful loop re-entry via ``loop_incoming_edges``.
    Supports human approval and pause via ``PAUSED`` state and ``approve(data)``.

    Methods:
        handle_edge_signal(edge_id, signal, payload, data_id, tags) -> data/bool
        prepare_outputs()  -- runs on_ready hook before outgoing edges fire
        pause_for_approval() -- requests pausing for human review when ready
        approve(approved_data) -- approves paused vertex, merging optional data and transitioning to READY
    """

    def __init__(
        self,
        vertex_id: str,
        settings: Optional[Dict] = None,
        initial_data: Optional[List[Dict]] = None,
    ):
        self.id = vertex_id
        self.settings = settings or {}

        # Data store: key = channel string -> value
        self._data_store: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

        # State management
        self._ready_event = asyncio.Event()
        self.state = VertexState.IDLE

        # Approval / HITL support
        self._require_approval: bool = bool(self.settings.get("require_approval", False))
        self._approved: bool = False

        # Edge tracking
        self.incoming_edges: List[str] = []   # edge IDs
        self.outgoing_edges: List[str] = []   # edge IDs
        self.required_input_count: int = 0
        self.completed_incoming_edges: set = set()
        self.aborted_incoming_edges: set = set()
        self._received_input_count: int = 0
        self.on_cancel_edges = None   # Optional callback for Race mode (wait_policy='any') cancellation

        # Loop support
        # Maps loop-back edge ID -> max_iterations (0 = unlimited).
        # Populated by Graph.validate() for cyclic graphs.
        self.loop_incoming_edges: Dict[str, int] = {}
        # Total number of times this vertex has re-entered via a loop-back edge.
        self.iteration_count: int = 0

        # Error / Abort info
        self.error_message: Optional[str] = None
        self.abort_reason: Optional[str] = None

        # Load initial data
        self.initial_data = initial_data or []
        if initial_data:
            for item in initial_data:
                key = str(item.get("channel", item.get("data_id", "default")))
                self._data_store[key] = item.get("value")
                logger.debug(
                    "[Vertex:%s] Loaded initial data: channel=%s, value=%s",
                    self.id, key, repr(item.get("value"))[:120],
                )

        logger.debug(
            "[Vertex:%s] Created | settings=%s | channels=%s",
            self.id, self.settings, list(self._data_store.keys()),
        )

    # ------------------------------------------------------------------
    # Approval & HITL API
    # ------------------------------------------------------------------
    def pause_for_approval(self) -> None:
        """Mark this vertex as requiring approval before proceeding to READY."""
        self._require_approval = True
        self._approved = False
        if self.state == VertexState.READY:
            self.state = VertexState.PAUSED
        logger.debug("[Vertex:%s] Marked for approval (require_approval=True)", self.id)

    def approve(self, approved_data: Optional[Dict] = None) -> None:
        """Approve a PAUSED (or pending approval) vertex, inject data, and transition to READY."""
        if approved_data:
            for channel, value in approved_data.items():
                self._data_store[str(channel)] = value
        self._approved = True
        self.state = VertexState.READY
        logger.debug("[Vertex:%s] ✓ Approved -> state transition to READY", self.id)

    # ------------------------------------------------------------------
    # State property
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Data access & Edge signaling
    # ------------------------------------------------------------------
    async def fetch_data(self, channel: str = "default") -> Any:
        """Command: Fetch data from the vertex's data store."""
        async with self._lock:
            val = self._data_store.get(channel)
            logger.debug(f"[Vertex:{self.id}] FETCH channel='{channel}' -> {repr(val)[:120]}")
            return val

    async def set_data(self, channel: str, value: Any) -> None:
        """Command: Set data directly in the vertex's data store."""
        async with self._lock:
            self._data_store[str(channel)] = value
            logger.debug(f"[Vertex:{self.id}] SET channel='{channel}' -> {repr(value)[:120]}")

    async def receive_signal(
        self,
        edge_id: str,
        signal: EdgeSignal,
        payload: Any = None,
        channel: str = "default",
    ) -> Any:
        """Event: Receive state update or completed payload from an edge."""
        if signal == EdgeSignal.COMPLETED:
            data = payload
            key = str(channel)
            logger.debug("[Vertex:%s] COMPLETED %s <- %s", self.id, key, repr(data)[:120])

            # --- run vertex on_receive hook ---
            try:
                data = self.on_receive(data, channel, self.settings)
                logger.debug("[Vertex:%s] on_receive returned: %s", self.id, repr(data)[:120])
            except Exception as exc:
                logger.warning("[Vertex:%s] on_receive REJECTED data: %s", self.id, exc)
                raise DataRejectedError(f"Vertex '{self.id}' rejected data: {exc}") from exc

            async with self._lock:
                # ── Loop re-entry ────────────────────────────────────────
                # The loop-back edge may arrive while the vertex is either:
                # • DONE       — normal case: previous iteration fully settled
                # • AWAITING_EDGES — concurrent case: the back-edge was fired
                #   from a downstream vertex while THIS vertex's own outgoing
                #   gather is still running (both in-flight simultaneously).
                # In both cases, treat it as a re-entry signal.
                if (
                    edge_id
                    and edge_id in self.loop_incoming_edges
                    and self.state in (VertexState.DONE, VertexState.AWAITING_EDGES)
                ):
                    max_iter = self.loop_incoming_edges[edge_id]

                    # Check limit BEFORE incrementing — the blocked delivery
                    # should not count as a re-entry in iteration_count.
                    if max_iter > 0 and self.iteration_count >= max_iter:
                        logger.debug(
                            "[Vertex:%s] Loop limit (%d) reached after %d re-entries "
                            "via edge '%s' — staying %s.",
                            self.id, max_iter, self.iteration_count, edge_id,
                            self.state.value,
                        )
                        return True  # Stay in current state, loop is exhausted

                    self.iteration_count += 1
                    self._data_store[key] = data

                    # The loop-back edge is the *re-entry trigger*, not a
                    # required input for the round. Already-settled non-loop
                    # inputs (e.g. a one-shot seed) must stay settled — clearing
                    # them here would make the vertex wait forever for an edge
                    # that already fired once and will never fire again
                    # (silent deadlock on mixed seed+loop topologies).
                    self.completed_incoming_edges.add(edge_id)

                    logger.debug(
                        "[Vertex:%s] ↺ Loop re-entry (iteration %d/%s) via '%s'",
                        self.id, self.iteration_count,
                        max_iter if max_iter > 0 else "∞",
                        edge_id,
                    )

                    # Ready once all NON-LOOP inputs are settled; the loop-back
                    # only re-arms the next round.
                    required = [
                        eid for eid in self.incoming_edges
                        if eid not in self.loop_incoming_edges
                    ]
                    settled_required = [
                        eid for eid in required
                        if eid in self.completed_incoming_edges
                        or eid in self.aborted_incoming_edges
                    ]
                    wait_policy = self.settings.get("wait_policy", "all")
                    if wait_policy == "any":
                        is_ready = (len(self.completed_incoming_edges) > 0 or len(required) == 0)
                    else:
                        is_ready = (len(settled_required) == len(required))

                    if is_ready:
                        if self._require_approval and not self._approved:
                            self.state = VertexState.PAUSED
                        else:
                            self.state = VertexState.READY
                    # else: remain IDLE, still waiting for non-loop inputs
                    return True
                # ── End loop re-entry ─────────────────────────────────────

                self._data_store[key] = data
                if edge_id:
                    self.completed_incoming_edges.add(edge_id)
                else:
                    self._received_input_count += 1
                
                total = len(self.incoming_edges) if self.incoming_edges else self.required_input_count
                logger.debug(
                    "[Vertex:%s] Input received (completed: %d, aborted: %d, total: %d)",
                    self.id, len(self.completed_incoming_edges), len(self.aborted_incoming_edges), total
                )
                
                # Check readiness based on incoming edge settlement
                is_ready = False
                if self.incoming_edges:
                    total_settled = len(self.completed_incoming_edges) + len(self.aborted_incoming_edges)
                    # Loop-back edges are *re-entry triggers*, not required
                    # inputs for the round: a vertex fed by a one-shot seed
                    # PLUS a loop-back must be able to become READY once its
                    # non-loop inputs settle — the loop-back can only fire
                    # after this vertex has already run once (otherwise the
                    # topology deadlocks on the very first round).
                    required = [
                        eid for eid in self.incoming_edges
                        if eid not in self.loop_incoming_edges
                    ]
                    
                    wait_policy = self.settings.get("wait_policy", "all")
                    if wait_policy == "any" and len(self.completed_incoming_edges) > 0:
                        # RACE WON!
                        is_ready = True
                        pending = [eid for eid in self.incoming_edges if eid not in self.completed_incoming_edges and eid not in self.aborted_incoming_edges]
                        if pending and hasattr(self, "on_cancel_edges") and callable(self.on_cancel_edges):
                            self.on_cancel_edges(pending)
                    else:
                        settled_required = [
                            eid for eid in required
                            if eid in self.completed_incoming_edges
                            or eid in self.aborted_incoming_edges
                        ]
                        is_ready = len(settled_required) == len(required) and len(self.completed_incoming_edges) > 0
                elif self.required_input_count > 0:
                    is_ready = self._received_input_count >= self.required_input_count

                if is_ready:
                    if self._require_approval and not self._approved:
                        self.state = VertexState.PAUSED
                    else:
                        self.state = VertexState.READY
            return True

        elif signal == EdgeSignal.ABORTED:
            logger.debug("[Vertex:%s] ABORTED signal from edge '%s'", self.id, edge_id)
            async with self._lock:
                # If already settled and this abort is from a loop-back edge,
                # the loop simply terminated (guard failed or limit reached).
                # Stay in current state — nothing more to do.
                if (
                    edge_id
                    and edge_id in self.loop_incoming_edges
                    and self.state in (VertexState.DONE, VertexState.AWAITING_EDGES)
                ):
                    logger.debug(
                        "[Vertex:%s] Loop-back edge '%s' aborted — loop terminates cleanly.",
                        self.id, edge_id,
                    )
                    return True

                self.aborted_incoming_edges.add(edge_id)
                
                total = len(self.incoming_edges) if self.incoming_edges else self.required_input_count
                total_settled = len(self.completed_incoming_edges) + len(self.aborted_incoming_edges)
                
                if total > 0 and total_settled >= total:
                    if len(self.completed_incoming_edges) > 0:
                        self.state = VertexState.READY
                    else:
                        self.abort_reason = payload or f"All {total} incoming edges aborted"
                        self.state = VertexState.ABORTED
            return True

        elif signal == EdgeSignal.FAILED:
            logger.error("[Vertex:%s] FAILED signal from edge '%s': %s", self.id, edge_id, payload)
            async with self._lock:
                self.error_message = str(payload)
                self.state = VertexState.ERROR
            return True

    async def get_all_data(self) -> Dict[str, Any]:
        """Return a copy of the entire data store."""
        async with self._lock:
            return dict(self._data_store)

    async def prepare_outputs(self):
        """Run the on_ready hook to consolidate data.

        Called by the executor right before outgoing edges fire.
        The hook receives all stored data and the vertex settings, and
        should return a dict of ``{channel: value}`` that will be merged
        into the data store.
        """
        logger.debug("[Vertex:%s] Running on_ready hook", self.id)
        all_data = dict(self._data_store)
        try:
            outputs = self.on_ready(all_data, self.settings)
            if outputs and isinstance(outputs, dict):
                async with self._lock:
                    for key, value in outputs.items():
                        store_key = str(key)
                        self._data_store[store_key] = value
                        logger.debug(
                            "[Vertex:%s] on_ready set %s = %s",
                            self.id, store_key, repr(value)[:120],
                        )
        except Exception as exc:
            logger.error("[Vertex:%s] on_ready hook failed: %s", self.id, exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Subclass hooks — override in subclasses to customise behaviour
    # ------------------------------------------------------------------
    def on_receive(self, data: Any, channel: str, settings: Dict) -> Any:
        """Called when an edge delivers data. Return data to store, or raise to reject."""
        return data

    def on_ready(self, all_data: Dict[str, Any], settings: Dict) -> Optional[Dict[str, Any]]:
        """Called before outgoing edges fire. Return {channel: value} to merge, or None."""
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def wait_ready(self, timeout: Optional[float] = None):
        """Block until the vertex reaches READY state."""
        await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)

    def is_source(self) -> bool:
        """True if this vertex has no incoming edges."""
        return len(self.incoming_edges) == 0

    def is_sink(self) -> bool:
        """True if this vertex has no outgoing edges."""
        return len(self.outgoing_edges) == 0

    def force_state(self, new_state: VertexState) -> None:
        """直接设置状态，跳过转换校验。

        给三类场景用：
          1. ``reset()``  —— 整台重开机，不属于生命周期流转
          2. 快照恢复     —— 从持久化状态重建，无需重放流转历史
          3. 测试搭初始态 —— 验证某个状态的行为，而非验证如何到达它

        注意：``self.state = X`` 会走 ``StateMachine.__set__`` 并校验，
        本方法才是无校验的入口。
        """
        type(self).state.force_state(self, new_state)

    def reset(self):
        """Reset vertex to initial state (for re-runs).

        走 force_state：reset 是整台重开机，不是生命周期流转，
        不允许因为当前在 ERROR/READY/PAUSED 就 reset 不了。
        """
        self.force_state(VertexState.IDLE)
        self._ready_event.clear()
        self._approved = False
        self.completed_incoming_edges.clear()
        self.aborted_incoming_edges.clear()
        self._received_input_count = 0
        self.iteration_count = 0
        self.error_message = None
        self.abort_reason = None
        logger.debug("[Vertex:%s] Reset to IDLE", self.id)

    def __repr__(self):
        loop_str = f" loop={self.iteration_count}" if self.loop_incoming_edges else ""
        return (
            f"Vertex(id={self.id!r}, state={self.state.value}, "
            f"in={len(self.incoming_edges)}, out={len(self.outgoing_edges)}{loop_str})"
        )
