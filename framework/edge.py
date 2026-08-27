"""Edge module - Connection between vertices in the graph.

An Edge represents a 5-stage pipeline: Guard -> Pre-process -> Compute -> Post-process -> Deliver.
It communicates with vertices via the unified ``handle_edge_signal`` method using ``EdgeSignal``.
"""

import logging
from typing import Any, Dict, List, Optional
from .vertex import VertexState, EdgeSignal

logger = logging.getLogger("vertex_edge_agent.edge")


class Edge:
    """Directed edge connecting a source vertex to a destination vertex.

    Attributes:
        id:              Unique edge identifier.
        source_id:       Source vertex ID.
        destination_id:  Destination vertex ID.
        channel:         Data channel for reading and writing data.
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
        channel: str = "default",
        prompt: str = "",
        model: str = "default",
        settings: Optional[Dict] = None,
        script_path: Optional[str] = None,
    ):
        self.id = edge_id
        self.source_id = source_id
        self.destination_id = destination_id
        self.channel = channel
        self.prompt = prompt
        self.model = model
        self.settings = settings or {}
        self.script_path = script_path
        self._script_module = None

        # Execution state
        self.completed: bool = False
        self.aborted: bool = False
        self.abort_reason: Optional[str] = None
        self.result: Any = None
        self.error: Optional[str] = None

        logger.info(
            "[Edge:%s] Created %s -> %s | channel=%s model=%s",
            self.id, source_id, destination_id, self.channel, model,
        )

    def set_script_module(self, module):
        """Attach a loaded external script module."""
        self._script_module = module
        logger.debug("[Edge:%s] Script module attached: %s", self.id, module)

    def evaluate_condition(self, data: Any, settings: Dict) -> bool:
        """Evaluate whether the guard condition is satisfied."""
        # 1. Custom method on subclass or instance
        if hasattr(self, "condition") and callable(getattr(self, "condition")):
            return bool(self.condition(data, settings))

        # 2. Script module hook
        if self._script_module:
            for hook in ("evaluate_condition", "condition", "on_gate", "guard"):
                if hasattr(self._script_module, hook) and callable(getattr(self._script_module, hook)):
                    return bool(getattr(self._script_module, hook)(data, settings))

        # 3. Declarative settings
        if not settings:
            return True  # Default to True if no settings (no guard)

        if "condition" in settings and callable(settings["condition"]):
            return bool(settings["condition"](data))

        if "match" in settings and isinstance(settings["match"], dict) and isinstance(data, dict):
            return all(data.get(k) == v for k, v in settings["match"].items())

        if "threshold" in settings:
            threshold = settings["threshold"]
            op = str(settings.get("operator", "==")).lower()
            val = data
            if isinstance(data, dict) and "field" in settings:
                val = data.get(settings["field"])

            try:
                if op in (">", "gt"):
                    return val > threshold
                elif op in (">=", "gte", "ge"):
                    return val >= threshold
                elif op in ("<", "lt"):
                    return val < threshold
                elif op in ("<=", "lte", "le"):
                    return val <= threshold
                elif op in ("==", "eq"):
                    return val == threshold
                elif op in ("!=", "ne"):
                    return val != threshold
                elif op == "in":
                    return val in threshold
                elif op == "contains":
                    return threshold in val
            except Exception as exc:
                logger.warning("[Edge:%s] Threshold comparison failed: %s", self.id, exc)
                return False

        return True  # If settings exist but no guard condition is specified, pass

    async def execute(self, source_vertex, dest_vertex, agents) -> Any:
        """Execute the edge pipeline.

        Steps:
            1. Guard (`evaluate_condition`) -> If false, Abort.
            2. Pre-process (via script hook)
            3. Compute (LLM process OR transparent pass-through)
            4. Post-process (via script hook)
            5. Deliver to destination vertex.

        Returns the final result written to the destination vertex.
        """
        logger.info(
            "[Edge:%s] EXECUTE  %s -[%s:%s]-> %s",
            self.id, self.source_id, self.channel, self.destination_id,
        )

        try:
            # 0 — Check source vertex abort state
            if hasattr(source_vertex, "state") and source_vertex.state == VertexState.ABORTED:
                self.aborted = True
                self.abort_reason = f"Upstream source vertex '{self.source_id}' is ABORTED"
                logger.info("[Edge:%s] Source '%s' is ABORTED -> Aborting edge and notifying '%s'", self.id, self.source_id, self.destination_id)
                await dest_vertex.receive_signal(self.id, EdgeSignal.ABORTED, payload=self.abort_reason)
                return None

            # 1 — Read from source
            data = await source_vertex.fetch_data(channel=self.channel)
            logger.debug("[Edge:%s] Source data: %s", self.id, repr(data)[:200])

            # 1.5 — Guard (evaluate condition)
            if not self.evaluate_condition(data, self.settings):
                self.aborted = True
                self.abort_reason = f"Guard condition not satisfied on edge '{self.id}'"
                logger.info(
                    "[Edge:%s] Guard condition NOT satisfied -> ABORTING (dest: '%s')",
                    self.id, self.destination_id,
                )
                await dest_vertex.receive_signal(self.id, EdgeSignal.ABORTED, payload=self.abort_reason)
                return None

            # 2 — Pre-process
            if hasattr(self, "pre_process") and callable(getattr(self, "pre_process")):
                data = self.pre_process(data, self.settings)
                logger.debug("[Edge:%s] After self.pre_process: %s", self.id, repr(data)[:200])
            elif self._script_module and hasattr(self._script_module, "pre_process"):
                data = self._script_module.pre_process(data, self.settings)
                logger.debug("[Edge:%s] After module pre_process: %s", self.id, repr(data)[:200])

            # 3 — Compute (PI Agent or Pass-through)
            if self.prompt or (self.model and self.model != "default"):
                result = await agents.process(
                    data=data,
                    prompt=self.prompt,
                    model=self.model,
                    settings=self.settings,
                )
                logger.debug("[Edge:%s] PI Agent result: %s", self.id, repr(result)[:200])
            else:
                result = data
                logger.debug("[Edge:%s] Pass-through result: %s", self.id, repr(result)[:200])

            # 4 — Post-process
            if hasattr(self, "post_process") and callable(getattr(self, "post_process")):
                result = self.post_process(result, self.settings)
                logger.debug("[Edge:%s] After self.post_process: %s", self.id, repr(result)[:200])
            elif self._script_module and hasattr(self._script_module, "post_process"):
                result = self._script_module.post_process(result, self.settings)
                logger.debug("[Edge:%s] After module post_process: %s", self.id, repr(result)[:200])

            # 5 — Write to destination
            await dest_vertex.receive_signal(self.id, EdgeSignal.COMPLETED, payload=result, channel=self.channel)
            logger.info(
                "[Edge:%s] Delivered to '%s' | key=(%s, %s)",
                self.id, self.destination_id, self.channel,
            )

            self.completed = True
            self.result = result
            return result

        except Exception as exc:
            self.error = str(exc)
            logger.error("[Edge:%s] FAILED: %s", self.id, exc, exc_info=True)
            # Propagate error to destination vertex to prevent deadlocks
            await dest_vertex.receive_signal(self.id, EdgeSignal.FAILED, payload=str(exc))
            raise

    def reset(self):
        """Reset edge state for re-execution."""
        self.completed = False
        self.aborted = False
        self.abort_reason = None
        self.result = None
        self.error = None

    def __repr__(self):
        status = "✓" if self.completed else ("⊘" if self.aborted else ("✗" if self.error else "·"))
        return f"{self.__class__.__name__}({self.id} {self.source_id}->{self.destination_id} [{status}])"




