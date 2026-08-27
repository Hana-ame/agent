"""Vertex module - Node in the computation graph.

A Vertex stores data keyed by (data_id, tags) tuples, has a state machine
for lifecycle management, and supports external Python scripts for data
handling, validation, and rejection.
"""

import asyncio
import enum
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("vertex_edge_agent.vertex")


class VertexState(enum.Enum):
    """States for vertex lifecycle."""
    IDLE = "idle"                # Waiting for inputs
    READY = "ready"              # All inputs received, ready to process
    PROCESSING = "processing"    # Outgoing edges being fired
    DONE = "done"                # All processing complete
    ABORTED = "aborted"          # Pruned or all inputs aborted
    ERROR = "error"              # Error occurred


class EdgeSignal(str, enum.Enum):
    """Signals exchanged between Edge and Vertex."""
    READ = "read"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"



class DataRejectedError(Exception):
    """Raised when a vertex rejects incoming data via its script."""
    pass


class Vertex:
    """A vertex (node) in the computation graph.

    Stores data keyed by (data_id, tags) tuples.
    Has a state machine for lifecycle management.
    Supports external scripts for data handling/validation/rejection.

    Methods:
        handle_edge_signal(edge_id, signal, payload, data_id, tags) -> data/bool
        prepare_outputs()  -- runs on_ready hook before outgoing edges fire
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

        # Data store: key = (data_id, tuple(sorted_tags)) -> value
        self._data_store: Dict[Tuple[str, Tuple[str, ...]], Any] = {}
        self._lock = asyncio.Lock()

        # State management
        self._state = VertexState.IDLE
        self._ready_event = asyncio.Event()

        # Edge tracking
        self.incoming_edges: List[str] = []   # edge IDs
        self.outgoing_edges: List[str] = []   # edge IDs
        self.required_input_count: int = 0
        self.completed_incoming_edges: set = set()
        self.aborted_incoming_edges: set = set()
        self._received_input_count: int = 0

        # Error / Abort info
        self.error_message: Optional[str] = None
        self.abort_reason: Optional[str] = None

        # Load initial data
        if initial_data:
            for item in initial_data:
                key = self._make_key(
                    item.get("data_id", "default"),
                    item.get("tags", []),
                )
                self._data_store[key] = item.get("value")
                logger.debug(
                    "[Vertex:%s] Loaded initial data: key=%s, value=%s",
                    self.id, key, repr(item.get("value"))[:120],
                )

        logger.info(
            "[Vertex:%s] Created | settings=%s | script=%s | initial_keys=%s",
            self.id, self.settings, self.script_path, list(self._data_store.keys()),
        )

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_key(
        data_id: str, tags: Optional[List[str]] = None
    ) -> Tuple[str, Tuple[str, ...]]:
        """Create a canonical key from data_id and tags."""
        return (data_id, tuple(sorted(tags or [])))

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
    async def handle_edge_signal(
        self,
        edge_id: str,
        signal: EdgeSignal,
        payload: Any = None,
        channel: str = "default",
    ) -> Any:
        """Unified method for all edge-to-vertex and vertex-to-edge communication."""
        if signal == EdgeSignal.READ:
            key = self._make_key(channel)
            async with self._lock:
                data = self._data_store.get(key)
            logger.debug("[Vertex:%s] READ by edge '%s' %s -> %s", self.id, edge_id, key, repr(data)[:120])
            return data

        elif signal == EdgeSignal.FAILED:
            async with self._lock:
                self.error_message = f"Upstream edge {edge_id} failed: {payload}"
                self.state = VertexState.ERROR
                logger.error("[Vertex:%s] Failed due to upstream edge '%s'", self.id, edge_id)

        elif signal == EdgeSignal.ABORTED:
            async with self._lock:
                self.aborted_incoming_edges.add(edge_id)
                total = len(self.incoming_edges) if self.incoming_edges else self.required_input_count
                logger.info(
                    "[Vertex:%s] Incoming edge '%s' aborted (completed: %d, aborted: %d, total: %d)",
                    self.id, edge_id,
                    len(self.completed_incoming_edges),
                    len(self.aborted_incoming_edges),
                    total,
                )

                total_settled = len(self.completed_incoming_edges) + len(self.aborted_incoming_edges)
                if total > 0 and total_settled >= total:
                    if len(self.completed_incoming_edges) > 0:
                        self.state = VertexState.READY
                    else:
                        self.abort_reason = payload or f"All {total} incoming edges aborted"
                        self.state = VertexState.ABORTED

        elif signal == EdgeSignal.COMPLETED:
            data = payload
            key = self._make_key(channel)
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
                    self.state = VertexState.READY
            return True

    async def get_all_data(self) -> Dict[Tuple[str, Tuple[str, ...]], Any]:
        """Return a copy of the entire data store."""
        async with self._lock:
            return dict(self._data_store)

    async def prepare_outputs(self):
        """Run the script's ``on_ready`` hook to consolidate data.

        Called by the executor right before outgoing edges fire.
        The hook receives all stored data and the vertex settings, and
        should return a dict of ``{(data_id, (tags,...)): value}`` that
        will be merged into the data store.
        """
        if hasattr(self, "on_ready") and callable(getattr(self, "on_ready")):
            logger.debug("[Vertex:%s] Running self.on_ready hook", self.id)
            all_data = dict(self._data_store)
            try:
                outputs = self.on_ready(all_data, self.settings)
                if outputs and isinstance(outputs, dict):
                    async with self._lock:
                        for key, value in outputs.items():
                            store_key = self._make_key(str(key))
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
                            store_key = self._make_key(str(key))
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
        self.completed_incoming_edges.clear()
        self.aborted_incoming_edges.clear()
        self._received_input_count = 0
        self.error_message = None
        self.abort_reason = None
        logger.debug("[Vertex:%s] Reset to IDLE", self.id)

    def __repr__(self):
        return (
            f"Vertex(id={self.id!r}, state={self._state.value}, "
            f"in={len(self.incoming_edges)}, out={len(self.outgoing_edges)})"
        )
