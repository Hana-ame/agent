"""EdgePipeline module — 5-stage computation pipeline for an Edge.

Separates execution responsibility from routing configuration (Edge),
following the Single Responsibility Principle.

Stages (run in order by :meth:`run`):
    0. Upstream abort check
    1. Fetch data from source vertex
    2. Guard  (``evaluate_condition``)
    3. Pre-process hook
    4. Compute  (LLM agent or transparent pass-through)
    5. Post-process hook
    6. Deliver to destination vertex
"""

import logging
from typing import Any, Dict, Optional

from .vertex import VertexState, EdgeSignal

logger = logging.getLogger("vertex_edge_agent.pipeline")


class EdgePipeline:
    """Encapsulates the 5-stage data transformation pipeline for an Edge.

    Args:
        prompt:         Prompt forwarded to the LLM agent.
        model:          Model identifier (e.g. ``"gemini-pro"``).
        settings:       Arbitrary settings dict (guards, hooks, LLM kwargs …).
        script_module:  Optional loaded external Python module with hook
                        functions (``evaluate_condition``, ``pre_process``,
                        ``post_process``).
        hook_provider:  Optional object (typically the owning :class:`Edge`
                        subclass instance) that may define overridable hook
                        methods: ``condition``, ``pre_process``,
                        ``post_process``.  Checked *before* the script module.

    Attributes:
        completed:     ``True`` if the pipeline completed successfully.
        aborted:       ``True`` if aborted by a guard or upstream abort.
        abort_reason:  Human-readable description of the abort cause.
        result:        Final value delivered to the destination vertex.
        error:         Error message string if the pipeline raised.
    """

    def __init__(
        self,
        prompt: str,
        model: str,
        settings: Dict,
        script_module=None,
        hook_provider=None,
    ):
        self.prompt = prompt
        self.model = model
        self.settings = settings

        self._script_module = script_module
        # hook_provider is checked for: condition, pre_process, post_process
        self._hook_provider = hook_provider

        # Execution state
        self.completed: bool = False
        self.aborted: bool = False
        self.abort_reason: Optional[str] = None
        self.result: Any = None
        self.error: Optional[str] = None

    # ------------------------------------------------------------------
    # Script module
    # ------------------------------------------------------------------
    def set_script_module(self, module) -> None:
        """Attach a loaded external script module."""
        self._script_module = module
        logger.debug("[Pipeline] Script module attached: %s", module)

    # ------------------------------------------------------------------
    # Stage 2 — Guard
    # ------------------------------------------------------------------
    def evaluate_condition(self, data: Any) -> bool:
        """Return ``True`` if the guard condition allows *data* through.

        Resolution order:
        1. ``hook_provider.condition(data, settings)`` — subclass override.
        2. Script module functions searched in order:
           ``evaluate_condition``, ``condition``, ``on_gate``, ``guard``.
        3. Declarative settings keys: ``condition`` callable, ``match`` dict,
           or ``threshold`` / ``operator`` / ``field`` combination.
        4. Default: ``True`` (no guard configured).
        """
        # 1. Hook provider (typically an Edge subclass defining `condition`)
        if self._hook_provider is not None:
            hook = getattr(self._hook_provider, "condition", None)
            if callable(hook):
                return bool(hook(data, self.settings))

        # 2. Script module hooks
        if self._script_module:
            for hook_name in ("evaluate_condition", "condition", "on_gate", "guard"):
                hook = getattr(self._script_module, hook_name, None)
                if callable(hook):
                    return bool(hook(data, self.settings))

        # 3. Declarative settings
        if not self.settings:
            return True

        if "condition" in self.settings and callable(self.settings["condition"]):
            return bool(self.settings["condition"](data))

        if (
            "match" in self.settings
            and isinstance(self.settings["match"], dict)
            and isinstance(data, dict)
        ):
            return all(data.get(k) == v for k, v in self.settings["match"].items())

        if "threshold" in self.settings:
            threshold = self.settings["threshold"]
            op = str(self.settings.get("operator", "==")).lower()
            val = data
            if isinstance(data, dict) and "field" in self.settings:
                val = data.get(self.settings["field"])

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
                logger.warning("[Pipeline] Threshold comparison failed: %s", exc)
                return False

        # Settings present but no guard condition matched → pass through
        return True

    # ------------------------------------------------------------------
    # Stage 3 — Pre-process helper
    # ------------------------------------------------------------------
    def _run_pre_process(self, edge_id: str, data: Any) -> Any:
        """Dispatch the pre-process hook (hook_provider → script_module)."""
        if self._hook_provider is not None:
            hook = getattr(self._hook_provider, "pre_process", None)
            if callable(hook):
                data = hook(data, self.settings)
                logger.debug(
                    "[Pipeline:%s] After hook_provider.pre_process: %s",
                    edge_id, repr(data)[:200],
                )
                return data

        if self._script_module:
            hook = getattr(self._script_module, "pre_process", None)
            if callable(hook):
                data = hook(data, self.settings)
                logger.debug(
                    "[Pipeline:%s] After module.pre_process: %s",
                    edge_id, repr(data)[:200],
                )

        return data

    # ------------------------------------------------------------------
    # Stage 5 — Post-process helper
    # ------------------------------------------------------------------
    def _run_post_process(self, edge_id: str, result: Any) -> Any:
        """Dispatch the post-process hook (hook_provider → script_module)."""
        if self._hook_provider is not None:
            hook = getattr(self._hook_provider, "post_process", None)
            if callable(hook):
                result = hook(result, self.settings)
                logger.debug(
                    "[Pipeline:%s] After hook_provider.post_process: %s",
                    edge_id, repr(result)[:200],
                )
                return result

        if self._script_module:
            hook = getattr(self._script_module, "post_process", None)
            if callable(hook):
                result = hook(result, self.settings)
                logger.debug(
                    "[Pipeline:%s] After module.post_process: %s",
                    edge_id, repr(result)[:200],
                )

        return result

    # ------------------------------------------------------------------
    # Main runner — all 6 stages in sequence
    # ------------------------------------------------------------------
    async def run(
        self,
        edge_id: str,
        source_id: str,
        destination_id: str,
        channel: str,
        source_vertex,
        dest_vertex,
        agents,
    ) -> Any:
        """Execute all pipeline stages sequentially and return the final result.

        Returns:
            The final result delivered to *dest_vertex*, or ``None`` if aborted.

        Raises:
            Any exception raised during compute or hook stages (after recording
            it in :attr:`error` and signalling FAILED to *dest_vertex*).
        """
        logger.info(
            "[Pipeline:%s] RUN  %s -[%s:%s]-> %s",
            edge_id, source_id, channel, destination_id,
        )

        try:
            # 0 — Upstream abort propagation
            if (
                hasattr(source_vertex, "state")
                and source_vertex.state == VertexState.ABORTED
            ):
                self.aborted = True
                self.abort_reason = (
                    f"Upstream source vertex '{source_id}' is ABORTED"
                )
                logger.info(
                    "[Pipeline:%s] Source ABORTED -> notifying '%s'",
                    edge_id, destination_id,
                )
                await dest_vertex.receive_signal(
                    edge_id, EdgeSignal.ABORTED, payload=self.abort_reason
                )
                return None

            # 1 — Fetch data from source
            data = await source_vertex.fetch_data(channel=channel)
            logger.debug(
                "[Pipeline:%s] Source data: %s", edge_id, repr(data)[:200]
            )

            # 2 — Guard
            if not self.evaluate_condition(data):
                self.aborted = True
                self.abort_reason = (
                    f"Guard condition not satisfied on edge '{edge_id}'"
                )
                logger.info(
                    "[Pipeline:%s] Guard NOT satisfied -> ABORTING (dest: '%s')",
                    edge_id, destination_id,
                )
                await dest_vertex.receive_signal(
                    edge_id, EdgeSignal.ABORTED, payload=self.abort_reason
                )
                return None

            # 3 — Pre-process
            data = self._run_pre_process(edge_id, data)

            # 4 — Compute  (LLM agent or transparent pass-through)
            if self.prompt or (self.model and self.model != "default"):
                result = await agents.process(
                    data=data,
                    prompt=self.prompt,
                    model=self.model,
                    settings=self.settings,
                )
                logger.debug(
                    "[Pipeline:%s] LLM result: %s", edge_id, repr(result)[:200]
                )
            else:
                result = data
                logger.debug(
                    "[Pipeline:%s] Pass-through: %s", edge_id, repr(result)[:200]
                )

            # 5 — Post-process
            result = self._run_post_process(edge_id, result)

            # 6 — Deliver
            await dest_vertex.receive_signal(
                edge_id, EdgeSignal.COMPLETED, payload=result, channel=channel
            )
            logger.info(
                "[Pipeline:%s] Delivered to '%s' via channel='%s'",
                edge_id, destination_id, channel,
            )

            self.completed = True
            self.result = result
            return result

        except Exception as exc:
            self.error = str(exc)
            logger.error(
                "[Pipeline:%s] FAILED: %s", edge_id, exc, exc_info=True
            )
            await dest_vertex.receive_signal(
                edge_id, EdgeSignal.FAILED, payload=str(exc)
            )
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset execution state so this pipeline can be re-run."""
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
        return f"EdgePipeline(model={self.model!r} [{status}])"
