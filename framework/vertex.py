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



class DataRejectedError(Exception):
    """Raised when a vertex rejects incoming data via its script."""
    pass


class Vertex:
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
        script_path: Optional[str] = None,
        initial_data: Optional[List[Dict]] = None,
    ):
        self.id = vertex_id
        self.settings = settings or {}
        self.script_path = script_path
        self._script_module = None

        # Data store: key = channel string -> value
        self._data_store: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

        # State management
        self._state = VertexState.IDLE
        self._ready_event = asyncio.Event()

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
        if initial_data:
            for item in initial_data:
                key = str(item.get("channel", item.get("data_id", "default")))
                self._data_store[key] = item.get("value")
                logger.debug(
                    "[Vertex:%s] Loaded initial data: channel=%s, value=%s",
                    self.id, key, repr(item.get("value"))[:120],
                )

        logger.info(
            "[Vertex:%s] Created | settings=%s | script=%s | channels=%s",
            self.id, self.settings, self.script_path, list(self._data_store.keys()),
        )

    # ------------------------------------------------------------------
    # Approval & HITL API
    # ------------------------------------------------------------------
    def pause_for_approval(self) -> None:
        """Mark this vertex as requiring approval before proceeding to READY."""
        self._require_approval = True
        self._approved = False
        if self._state == VertexState.READY:
            self.state = VertexState.PAUSED
        logger.info("[Vertex:%s] Marked for approval (require_approval=True)", self.id)

    def approve(self, approved_data: Optional[Dict] = None) -> None:
        """Approve a PAUSED (or pending approval) vertex, inject data, and transition to READY."""
        if approved_data:
            for channel, value in approved_data.items():
                self._data_store[str(channel)] = value
        self._approved = True
        self.state = VertexState.READY
        logger.info("[Vertex:%s] ✓ Approved -> state transition to READY", self.id)

    # ------------------------------------------------------------------
    # State property
    # ------------------------------------------------------------------
    @property
    def state(self) -> VertexState:
        return self._state

    @state.setter
    def state(self, new_state: VertexState):
        old = self._state
        self._state = new_state
        logger.info("[Vertex:%s] %s -> %s", self.id, old.value, new_state.value)
        if new_state in (VertexState.READY, VertexState.ABORTED, VertexState.ERROR):
            self._ready_event.set()
        else:
            self._ready_event.clear()

    # ------------------------------------------------------------------
    # Script
    # ------------------------------------------------------------------
    def set_script_module(self, module):
        """Attach a loaded external script module."""
        self._script_module = module
        logger.debug("[Vertex:%s] Script module attached: %s", self.id, module)

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

            # --- run vertex script on_receive hook ---
            if hasattr(self, "on_receive") and callable(getattr(self, "on_receive")):
                try:
                    data = self.on_receive(data, channel, self.settings)
                    logger.debug("[Vertex:%s] self.on_receive returned: %s", self.id, repr(data)[:120])
                except Exception as exc:
                    logger.warning("[Vertex:%s] self.on_receive REJECTED data: %s", self.id, exc)
                    raise DataRejectedError(f"Vertex '{self.id}' rejected data: {exc}") from exc
            elif self._script_module and hasattr(self._script_module, "on_receive"):
                try:
                    data = self._script_module.on_receive(
                        data, channel, self.settings
                    )
                    logger.debug(
                        "[Vertex:%s] on_receive returned: %s", self.id, repr(data)[:120]
                    )
                except Exception as exc:
                    logger.warning(
                        "[Vertex:%s] on_receive REJECTED data: %s", self.id, exc
                    )
                    raise DataRejectedError(
                        f"Vertex '{self.id}' rejected data: {exc}"
                    ) from exc

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
                    and self._state in (VertexState.DONE, VertexState.AWAITING_EDGES)
                ):
                    max_iter = self.loop_incoming_edges[edge_id]

                    # Check limit BEFORE incrementing — the blocked delivery
                    # should not count as a re-entry in iteration_count.
                    if max_iter > 0 and self.iteration_count >= max_iter:
                        logger.info(
                            "[Vertex:%s] Loop limit (%d) reached after %d re-entries "
                            "via edge '%s' — staying %s.",
                            self.id, max_iter, self.iteration_count, edge_id,
                            self._state.value,
                        )
                        return True  # Stay in current state, loop is exhausted

                    self.iteration_count += 1

                    # Reset per-iteration tracking for the new round
                    self.completed_incoming_edges.clear()
                    self.aborted_incoming_edges.clear()
                    self._received_input_count = 0
                    self.completed_incoming_edges.add(edge_id)
                    self._data_store[key] = data

                    logger.info(
                        "[Vertex:%s] ↺ Loop re-entry (iteration %d/%s) via '%s'",
                        self.id, self.iteration_count,
                        max_iter if max_iter > 0 else "∞",
                        edge_id,
                    )

                    # If this is the only incoming edge (simple cycle), go READY/PAUSED.
                    # Otherwise fall to IDLE and wait for remaining inputs.
                    total_settled = (
                        len(self.completed_incoming_edges)
                        + len(self.aborted_incoming_edges)
                    )
                    if total_settled >= len(self.incoming_edges):
                        if self._require_approval and not self._approved:
                            self.state = VertexState.PAUSED
                        else:
                            self.state = VertexState.READY
                    # else: remain IDLE, other non-loop edges still pending
                    return True
                # ── End loop re-entry ─────────────────────────────────────

                self._data_store[key] = data
                if edge_id:
                    self.completed_incoming_edges.add(edge_id)
                else:
                    self._received_input_count += 1
                
                total = len(self.incoming_edges) if self.incoming_edges else self.required_input_count
                logger.info(
                    "[Vertex:%s] Input received (completed: %d, aborted: %d, total: %d)",
                    self.id, len(self.completed_incoming_edges), len(self.aborted_incoming_edges), total
                )
                
                # Check readiness based on incoming edge settlement
                is_ready = False
                if self.incoming_edges:
                    total_settled = len(self.completed_incoming_edges) + len(self.aborted_incoming_edges)
                    is_ready = total_settled >= len(self.incoming_edges) and len(self.completed_incoming_edges) > 0
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
                    and self._state in (VertexState.DONE, VertexState.AWAITING_EDGES)
                ):
                    logger.info(
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
        """Run the script's ``on_ready`` hook to consolidate data.

        Called by the executor right before outgoing edges fire.
        The hook receives all stored data and the vertex settings, and
        should return a dict of ``{channel: value}`` that will be merged
        into the data store.
        """
        if hasattr(self, "on_ready") and callable(getattr(self, "on_ready")):
            logger.debug("[Vertex:%s] Running self.on_ready hook", self.id)
            all_data = dict(self._data_store)
            try:
                outputs = self.on_ready(all_data, self.settings)
                if outputs and isinstance(outputs, dict):
                    async with self._lock:
                        for key, value in outputs.items():
                            store_key = str(key)
                            self._data_store[store_key] = value
                            logger.debug(
                                "[Vertex:%s] self.on_ready set %s = %s",
                                self.id, store_key, repr(value)[:120],
                            )
            except Exception as exc:
                logger.error("[Vertex:%s] self.on_ready hook failed: %s", self.id, exc, exc_info=True)
                raise
        elif self._script_module and hasattr(self._script_module, "on_ready"):
            logger.debug("[Vertex:%s] Running module on_ready hook", self.id)
            all_data = dict(self._data_store)
            try:
                outputs = self._script_module.on_ready(all_data, self.settings)
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
                logger.error(
                    "[Vertex:%s] on_ready hook failed: %s", self.id, exc, exc_info=True
                )
                raise

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

    def reset(self):
        """Reset vertex to initial state (for re-runs)."""
        self._state = VertexState.IDLE
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
            f"Vertex(id={self.id!r}, state={self._state.value}, "
            f"in={len(self.incoming_edges)}, out={len(self.outgoing_edges)}{loop_str})"
        )
