"""Edge module — Routing connection between vertices in the graph.

An Edge defines *where* data flows (source → destination via channel).
The *how* — guard evaluation, hooks, LLM computation, and delivery — is
fully encapsulated in :class:`~framework.pipeline.EdgePipeline`.

This separation follows the Single Responsibility Principle:

* ``Edge``         — routing config + public extension surface
* ``EdgePipeline`` — 5-stage computation logic
"""

import logging
from typing import Any, Dict, Optional

from .pipeline import EdgePipeline

logger = logging.getLogger("vertex_edge_agent.edge")


class Edge:
    """Directed edge connecting a source vertex to a destination vertex.

    Attributes:
        id:              Unique edge identifier.
        source_id:       Source vertex ID.
        destination_id:  Destination vertex ID.
        channel:         Data channel for reading and writing data.
        prompt:          Prompt sent to the LLM Agent.
        model:           Model identifier for the LLM Agent.
        settings:        Arbitrary settings dict passed to agent & scripts.
        script_path:     Optional path to an external Python script.

    Subclass hooks (override in a subclass to customise pipeline behaviour):
        condition(data, settings) -> bool   — guard stage
        pre_process(data, settings) -> data — pre-process stage
        post_process(data, settings) -> data — post-process stage
    """

    def __init__(
        self,
        edge_id: str,
        source_id: str,
        destination_id: str,
        channel: str = "default",
        prompt: str = "",
        model: str = "default",
        settings: Optional[Dict] = None,
        script_path: Optional[str] = None,
        max_iterations: int = 0,
    ):
        # --- Routing config (Edge's sole responsibility) ---
        self.id = edge_id
        self.source_id = source_id
        self.destination_id = destination_id
        self.channel = channel
        self.script_path = script_path
        # Loop support: when > 0, this edge is a back-edge that allows the
        # destination vertex to re-enter READY state up to max_iterations times.
        self.max_iterations = max_iterations

        # Keep prompt/model/settings as direct attributes so graph loaders
        # and tests can read them without going through the pipeline.
        self.prompt = prompt
        self.model = model
        self.settings = settings or {}

        # --- Computation pipeline ---
        # Pass `self` as hook_provider so any subclass-defined hooks
        # (condition, pre_process, post_process) are discovered naturally
        # via hasattr inside the pipeline.
        self._pipeline = EdgePipeline(
            prompt=prompt,
            model=model,
            settings=self.settings,
            hook_provider=self,
        )

        logger.info(
            "[Edge:%s] Created %s -> %s | channel=%s model=%s",
            self.id, source_id, destination_id, self.channel, model,
        )

    # ------------------------------------------------------------------
    # Execution state — delegate to pipeline (backward-compatible)
    # ------------------------------------------------------------------
    @property
    def completed(self) -> bool:
        return self._pipeline.completed

    @completed.setter
    def completed(self, value: bool) -> None:
        self._pipeline.completed = value

    @property
    def aborted(self) -> bool:
        return self._pipeline.aborted

    @aborted.setter
    def aborted(self, value: bool) -> None:
        self._pipeline.aborted = value

    @property
    def abort_reason(self) -> Optional[str]:
        return self._pipeline.abort_reason

    @abort_reason.setter
    def abort_reason(self, value: Optional[str]) -> None:
        self._pipeline.abort_reason = value

    @property
    def result(self) -> Any:
        return self._pipeline.result

    @result.setter
    def result(self, value: Any) -> None:
        self._pipeline.result = value

    @property
    def error(self) -> Optional[str]:
        return self._pipeline.error

    @error.setter
    def error(self, value: Optional[str]) -> None:
        self._pipeline.error = value

    # ------------------------------------------------------------------
    # Script
    # ------------------------------------------------------------------
    @property
    def _script_module(self):
        """Proxy to the pipeline's script module (backward-compatible)."""
        return self._pipeline._script_module

    @_script_module.setter
    def _script_module(self, module) -> None:
        self._pipeline._script_module = module

    def set_script_module(self, module) -> None:
        """Attach a loaded external script module (forwarded to pipeline)."""
        self._pipeline.set_script_module(module)
        logger.debug("[Edge:%s] Script module attached: %s", self.id, module)

    # ------------------------------------------------------------------
    # Guard — public proxy so tests / external callers still work
    # ------------------------------------------------------------------
    def evaluate_condition(self, data: Any, settings: Optional[Dict] = None) -> bool:
        """Evaluate whether the guard condition is satisfied.

        Delegates to :meth:`EdgePipeline.evaluate_condition`.  The optional
        *settings* argument is accepted for backward compatibility but
        ignored — the pipeline always uses its own ``self.settings``.
        """
        return self._pipeline.evaluate_condition(data)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def execute(self, source_vertex, dest_vertex, agents, **kwargs) -> Any:
        """Run the 5-stage pipeline for this edge.

        Delegates entirely to :meth:`EdgePipeline.run`.
        """
        return await self._pipeline.run(
            edge_id=self.id,
            source_id=self.source_id,
            destination_id=self.destination_id,
            channel=self.channel,
            source_vertex=source_vertex,
            dest_vertex=dest_vertex,
            agents=agents,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset edge (and its pipeline) execution state for re-runs."""
        self._pipeline.reset()

    def __repr__(self) -> str:
        status = (
            "✓" if self.completed
            else ("⊘" if self.aborted
            else ("✗" if self.error else "·"))
        )
        return (
            f"{self.__class__.__name__}"
            f"({self.id} {self.source_id}->{self.destination_id} [{status}])"
        )
