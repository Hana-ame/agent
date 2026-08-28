"""Edge module — Routing connection between vertices in the graph.

An Edge defines *where* data flows (source → destination via channel).
The *how* — guard evaluation, hooks, LLM computation, and delivery — is
fully encapsulated in :class:`~framework.pipeline.Pipeline`.

This separation follows the Single Responsibility Principle:

* ``Edge``         — routing config + public extension surface
* ``Pipeline`` — 5-stage computation logic
"""

import logging
from typing import Any, Dict, Optional, Union

from .pipeline import Pipeline

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
        max_iterations: int = 0,
        agent: Optional[Union[str, 'BaseAgent', Dict]] = None,
        concurrency_type: str = "default",
    ):
        # --- Routing config (Edge's sole responsibility) ---
        self.id = edge_id
        self.source_id = source_id
        self.destination_id = destination_id
        self.channel = channel
        self.concurrency_type = concurrency_type
        self._pipeline_module = None
        self.completed: bool = False
        self.aborted: bool = False
        self.abort_reason: Optional[str] = None
        self.result: Any = None
        self.error: Optional[str] = None
        # Loop support: when > 0, this edge is a back-edge that allows the
        # destination vertex to re-enter READY state up to max_iterations times.
        self.max_iterations = max_iterations

        # Keep prompt/model/settings as direct attributes so graph loaders
        # and tests can read them without going through the pipeline.
        self.prompt = prompt
        self.model = model
        from .agents import get_agent
        self.agent = get_agent(agent)
        self.settings = settings or {}


        logger.debug(
            "[Edge:%s] Created %s -> %s | channel=%s model=%s",
            self.id, source_id, destination_id, self.channel, model,
        )

    # ------------------------------------------------------------------
    # Execution state — delegate to pipeline (backward-compatible)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Script
    # ------------------------------------------------------------------
    def set_pipeline_module(self, module) -> None:
        self._pipeline_module = module
        logger.debug("[Edge:%s] Script module attached: %s", self.id, module)
    # ------------------------------------------------------------------
    # Guard — public proxy so tests / external callers still work
    # ------------------------------------------------------------------
    def evaluate_condition(self, data: Any, settings: Optional[Dict] = None) -> bool:
        from .pipeline import Pipeline
        pipeline = Pipeline(
            prompt=self.prompt,
            model=self.model,
            settings=self.settings,
            pipeline_module=self._pipeline_module,
            hook_provider=self,
            agent=self.agent,
            log_id=self.id
        )
        return pipeline.evaluate_condition(data)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def execute(self, source_vertex, dest_vertex, agents, **kwargs) -> Any:
        from .pipeline import Pipeline, AbortPipeline
        from .vertex import EdgeSignal
        
        # 1. Fetch data from source vertex
        data = await source_vertex.fetch_data(channel=self.channel)
        
        # 2. Build stateless pipeline
        pipeline = Pipeline(
            prompt=self.prompt,
            model=self.model,
            settings=self.settings,
            pipeline_module=self._pipeline_module,
            hook_provider=self,
            agent=self.agent,
            log_id=self.id
        )
        
        # 3. Run pipeline
        try:
            result = await pipeline.run(data, agents=agents, **kwargs)
        except AbortPipeline as e:
            self.aborted = True
            self.abort_reason = e.reason
            await dest_vertex.receive_signal(self.id, EdgeSignal.ABORTED, payload=e.reason)
            return None
        except Exception as exc:
            self.error = str(exc)
            await dest_vertex.receive_signal(self.id, EdgeSignal.FAILED, payload=str(exc))
            raise
            
        # 4. Deliver result
        self.completed = True
        self.result = result
        await dest_vertex.receive_signal(
            self.id, EdgeSignal.COMPLETED, payload=result, channel=self.channel
        )
        return result

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset edge (and its pipeline) execution state for re-runs."""
        self.completed = False
        self.aborted = False
        self.abort_reason = None
        self.result = None
        self.error = None

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
