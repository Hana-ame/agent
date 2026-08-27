"""Edge module - Connection between vertices in the graph.

An Edge reads data from a source vertex (via ``get``), processes it through
a PI Agent, and writes the result to the destination vertex (via ``set``).
External scripts can pre-process data before the PI Agent or post-process
the result before delivery.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vertex_edge_agent.edge")


class Edge:
    """Directed edge connecting a source vertex to a destination vertex.

    Attributes:
        id:              Unique edge identifier.
        source_id:       Source vertex ID.
        destination_id:  Destination vertex ID.
        data_id:         Data key used for ``get`` / ``set``.
        tags:            Tag list used for ``get`` / ``set``.
        prompt:          Prompt sent to the PI Agent.
        model:           Model identifier for the PI Agent.
        settings:        Arbitrary settings dict passed to agent & scripts.
        script_path:     Optional path to an external Python script.
    """

    def __init__(
        self,
        edge_id: str,
        source_id: str,
        destination_id: str,
        data_id: str = "default",
        tags: Optional[List[str]] = None,
        prompt: str = "",
        model: str = "default",
        settings: Optional[Dict] = None,
        script_path: Optional[str] = None,
    ):
        self.id = edge_id
        self.source_id = source_id
        self.destination_id = destination_id
        self.data_id = data_id
        self.tags = tags or []
        self.prompt = prompt
        self.model = model
        self.settings = settings or {}
        self.script_path = script_path
        self._script_module = None

        # Execution state
        self.completed: bool = False
        self.result: Any = None
        self.error: Optional[str] = None

        logger.info(
            "[Edge:%s] Created %s -> %s | data_id=%s tags=%s model=%s",
            self.id, source_id, destination_id, data_id, self.tags, model,
        )

    def set_script_module(self, module):
        """Attach a loaded external script module."""
        self._script_module = module
        logger.debug("[Edge:%s] Script module attached: %s", self.id, module)

    async def execute(self, source_vertex, dest_vertex, pi_agent) -> Any:
        """Execute the edge pipeline.

        Steps:
            1. ``source_vertex.get(data_id, tags)``  → raw data
            2. Script ``pre_process(data, settings)`` (optional)
            3. ``pi_agent.process(data, prompt, model, settings)``
            4. Script ``post_process(result, settings)`` (optional)
            5. ``dest_vertex.set(result, data_id, tags)``

        Returns the final result written to the destination vertex.
        """
        logger.info(
            "[Edge:%s] EXECUTE  %s -[%s:%s]-> %s",
            self.id, self.source_id, self.data_id, self.tags, self.destination_id,
        )

        try:
            # 1 — read from source
            data = await source_vertex.get(self.data_id, self.tags)
            logger.debug("[Edge:%s] Source data: %s", self.id, repr(data)[:200])
            if data is None:
                logger.warning(
                    "[Edge:%s] Source vertex '%s' returned None for key=(%s, %s)",
                    self.id, self.source_id, self.data_id, self.tags,
                )

            # 2 — pre-process
            if hasattr(self, "pre_process") and callable(getattr(self, "pre_process")):
                data = self.pre_process(data, self.settings)
                logger.debug("[Edge:%s] After self.pre_process: %s", self.id, repr(data)[:200])
            elif self._script_module and hasattr(self._script_module, "pre_process"):
                data = self._script_module.pre_process(data, self.settings)
                logger.debug("[Edge:%s] After module pre_process: %s", self.id, repr(data)[:200])

            # 3 — PI Agent
            result = await pi_agent.process(
                data=data,
                prompt=self.prompt,
                model=self.model,
                settings=self.settings,
            )
            logger.debug("[Edge:%s] PI Agent result: %s", self.id, repr(result)[:200])

            # 4 — post-process
            if hasattr(self, "post_process") and callable(getattr(self, "post_process")):
                result = self.post_process(result, self.settings)
                logger.debug("[Edge:%s] After self.post_process: %s", self.id, repr(result)[:200])
            elif self._script_module and hasattr(self._script_module, "post_process"):
                result = self._script_module.post_process(result, self.settings)
                logger.debug("[Edge:%s] After module post_process: %s", self.id, repr(result)[:200])

            # 5 — write to destination
            await dest_vertex.set(result, self.data_id, self.tags, edge_id=self.id)
            logger.info(
                "[Edge:%s] Delivered to '%s' | key=(%s, %s)",
                self.id, self.destination_id, self.data_id, self.tags,
            )

            self.completed = True
            self.result = result
            return result

        except Exception as exc:
            self.error = str(exc)
            logger.error("[Edge:%s] FAILED: %s", self.id, exc, exc_info=True)
            # Propagate error to destination vertex to prevent deadlocks
            await dest_vertex.mark_edge_failed(self.id, str(exc))
            raise

    def reset(self):
        """Reset edge state for re-execution."""
        self.completed = False
        self.result = None
        self.error = None

    def __repr__(self):
        status = "✓" if self.completed else ("✗" if self.error else "·")
        return f"Edge({self.id} {self.source_id}->{self.destination_id} [{status}])"
