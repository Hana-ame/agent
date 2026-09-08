"""ReflexiveEdgeV4 implementation: Self-loop error recovery edge with circuit breaker."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Union

from framework.edges.base import EdgeResultV4, EdgeV4, AgentProtocol, _resolve_script_callable
from framework.edges.registry import register_edge_type
from framework.utils.script_loader import ScriptNotAllowedError
from framework.vertex_v4 import VertexStateV4, VertexStoreV4

logger = logging.getLogger("vertex_edge_agent.edges.reflexive")


@register_edge_type("reflexive", "recovery")
class ReflexiveEdgeV4(EdgeV4):
    """Reflexive recovery edge (input == output) designed for reset on reject."""

    IS_RECOVERY = True

    def __init__(
        self,
        edge_id: str,
        vertex_name: str,
        trigger_state: str = VertexStateV4.REJECT.value,
        target_state: str = VertexStateV4.TODO_URGENT.value,
        max_retries: int = 3,
        script: Optional[Union[str, Callable]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_group: Optional[str] = "reflexive",
        priority: int = 10,
        timeout: Optional[float] = None,
    ):
        edge_settings = dict(settings or {})
        edge_settings["trigger_state"] = trigger_state
        edge_settings["target_state"] = target_state
        edge_settings["max_retries"] = int(max_retries)
        if script:
            edge_settings["script"] = script
        super().__init__(
            edge_id=edge_id,
            input_vertex=vertex_name,
            output_vertex=vertex_name,
            edge_type="reflexive",
            settings=edge_settings,
            concurrency_limit=concurrency_limit,
            concurrency_group=concurrency_group or "reflexive",
            priority=priority,
            timeout=timeout,
        )
        self.trigger_state = trigger_state
        self.target_state = target_state
        self.max_retries = int(max_retries)
        self.script = script
        self._callable: Optional[Callable] = None
        if callable(script):
            self._callable = script

    @classmethod
    def from_config_dict(cls, data: Dict[str, Any], base_dir: Optional[str] = None) -> "ReflexiveEdgeV4":
        """Build a reflexive recovery edge from a config dict."""
        from framework.edges.base import (
            _config_common_kwargs,
            _config_edge_id,
            _config_endpoint,
            _config_script,
        )

        settings = dict(data.get("settings") or {})
        node = _config_endpoint(data, "input") or _config_endpoint(data, "output")
        if not node:
            raise ValueError("Target vertex ('input' or 'output') is required for reflexive edge")
        trigger_state = data.get("trigger_state") or settings.get(
            "trigger_state", VertexStateV4.REJECT.value
        )
        target_state = data.get("target_state") or settings.get(
            "target_state", VertexStateV4.TODO_URGENT.value
        )
        raw_retries = data.get("max_retries") or settings.get("max_retries", 3)
        return cls(
            edge_id=_config_edge_id(data),
            vertex_name=str(node),
            trigger_state=trigger_state,
            target_state=target_state,
            max_retries=int(raw_retries),
            script=_config_script(data, base_dir),
            **_config_common_kwargs(data),
        )

    def _resolve_callable(self) -> Optional[Callable]:
        """Resolve optional recovery script to callable function."""
        if self._callable is not None:
            return self._callable

        if callable(self.script):
            self._callable = self.script
            return self._callable

        if isinstance(self.script, str):
            stripped = self.script.strip()
            if stripped.startswith("lambda ") or stripped.startswith("lambda:"):
                if not self.settings.get("allow_inline_script"):
                    raise ScriptNotAllowedError(
                        "Inline lambda scripts are disabled. Set settings['allow_inline_script']=true "
                        "in a trusted local configuration, or reference a script path instead."
                    )
                logger.warning(
                    "[ReflexiveEdgeV4:%s] Evaluating inline lambda script (allow_inline_script=True).",
                    self.id,
                )
                try:
                    self._callable = eval(stripped, {"__builtins__": __builtins__})
                    return self._callable
                except Exception as exc:
                    logger.warning("Failed to evaluate inline lambda '%s': %s", stripped, exc)
            self._callable = _resolve_script_callable(self.script, ["recover", "retry", "reset", "transform"])
            return self._callable

        return None

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[AgentProtocol] = None,
        **kwargs: Any,
    ) -> EdgeResultV4:
        """Execute reflexive recovery upon target vertex entering trigger state."""
        target_v = store.get_vertex(session_id, self.output_vertex)
        if target_v is None:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=f"Target vertex '{self.output_vertex}' not found",
            )

        if target_v.state != self.trigger_state:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=f"Target state is '{target_v.state}', waiting for '{self.trigger_state}'",
            )

        try:
            # Retrieve diagnostics from staging table
            latest_error = store.get_latest_staged_for_vertex(
                session_id=session_id,
                vertex_name=self.output_vertex,
                key="error_feedback",
            )
            error_info = latest_error.value if latest_error else "No staged error diagnostic"

            # Circuit breaker: Check if maximum iterations reached
            if target_v.processed_count >= self.max_retries:
                logger.warning(
                    "[ReflexiveEdgeV4:%s] Max retries (%d) reached for vertex '%s'. Locking to 'forbidden'.",
                    self.id,
                    self.max_retries,
                    self.output_vertex,
                )
                store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.FORBIDDEN.value)
                store.stage_output(
                    session_id=session_id,
                    edge_id=self.id,
                    key="retry_exhausted",
                    value=f"Exceeded max retries: {self.max_retries}",
                    vertex_name=self.output_vertex,
                )
                return EdgeResultV4(
                    edge_id=self.id,
                    success=False,
                    error="Max retries exceeded",
                    reason=f"Attempted {target_v.processed_count} retries out of {self.max_retries}",
                )

            # Apply optional recovery transformer
            new_content = target_v.content
            fn = self._resolve_callable()
            if fn:
                try:
                    if asyncio.iscoroutinefunction(fn):
                        new_content = await fn(target_v.content, error_info, self.settings)
                    else:
                        new_content = fn(target_v.content, error_info, self.settings)
                except Exception as rec_err:
                    logger.error("[ReflexiveEdgeV4:%s] Recovery script failed: %s", self.id, rec_err)
                    # Recovery script failed — count this toward circuit breaker
                    store.increment_processed_count(session_id, target_v.name)
                    store.stage_output(
                        session_id=session_id,
                        edge_id=self.id,
                        key="error_feedback",
                        value=f"Recovery script failed: {rec_err}",
                        vertex_name=self.output_vertex,
                        metadata={"exception_type": type(rec_err).__name__},
                    )
                    return EdgeResultV4(
                        edge_id=self.id,
                        success=False,
                        error=f"Recovery script failed: {rec_err}",
                        reason="Recovery transform threw exception",
                    )

            # Reset state back to target_state ('todo urgent' or 'todo') and increment processed_count
            store.update_vertex_content(
                session_id=session_id,
                name=self.output_vertex,
                content=str(new_content),
                state=self.target_state,
                increment_count=True,
            )

            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="reflexive_reset",
                value=f"Reset vertex to '{self.target_state}', attempt #{target_v.processed_count + 1}",
                vertex_name=self.output_vertex,
                metadata={"diagnostic": error_info},
            )

            return EdgeResultV4(
                edge_id=self.id,
                success=True,
                output=new_content,
                metadata={
                    "attempt": target_v.processed_count + 1,
                    "reset_to": self.target_state,
                },
            )
        except Exception as exc:
            logger.error("[ReflexiveEdgeV4:%s] Unhandled error: %s", self.id, exc)
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                error=str(exc),
                reason="Reflexive recovery encountered unhandled exception",
            )
