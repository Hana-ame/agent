"""CodeEdgeV4 implementation: Python transformation logic and script execution."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Union

from framework.edges.base import EdgeResultV4, EdgeV4, AgentProtocol, _resolve_script_callable
from framework.vertex_v4 import VertexStateV4, VertexStoreV4

logger = logging.getLogger("vertex_edge_agent.edges.code")


class CodeEdgeV4(EdgeV4):
    """Executes Python transformation logic or scripts between vertices."""

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        script: Optional[Union[str, Callable]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_group: Optional[str] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
    ):
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            edge_type="code",
            settings=settings,
            concurrency_limit=concurrency_limit,
            concurrency_group=concurrency_group,
            priority=priority,
            timeout=timeout,
        )
        self.script = script
        self._callable: Optional[Callable] = None
        if callable(script):
            self._callable = script

    def _resolve_callable(self) -> Callable:
        """Resolve script to callable function."""
        if self._callable is not None:
            return self._callable

        if callable(self.script):
            self._callable = self.script
            return self._callable

        if isinstance(self.script, str):
            stripped = self.script.strip()
            if stripped.startswith("lambda ") or stripped.startswith("lambda:"):
                try:
                    self._callable = eval(stripped, {"__builtins__": __builtins__})
                    return self._callable
                except Exception as exc:
                    logger.warning("Failed to evaluate inline lambda '%s': %s", stripped, exc)
            self._callable = _resolve_script_callable(self.script, ["execute", "process", "run", "transform"])
            return self._callable

        # Default passthrough transformation
        self._callable = lambda content, settings, staging=None: content
        return self._callable

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[AgentProtocol] = None,
        auto_transition: bool = True,
        **kwargs: Any,
    ) -> EdgeResultV4:
        """Execute code edge following the two-sided handshake contract."""
        satisfied, reason, in_v, out_v = self.check_handshake(session_id, store)
        if not satisfied or in_v is None or out_v is None:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=reason,
            )

        try:
            fn = self._resolve_callable()
            # Retrieve any latest staged records for context
            latest_staging = store.get_latest_staged_for_vertex(session_id, self.output_vertex)
            staging_dict = latest_staging.to_dict() if latest_staging else {}

            import inspect
            try:
                sig = inspect.signature(fn)
                pos_params = [
                    p for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                has_var = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
                if len(pos_params) == 1 and not has_var:
                    args = (in_v.content,)
                elif len(pos_params) == 2 and not has_var:
                    args = (in_v.content, self.settings)
                else:
                    args = (in_v.content, self.settings, staging_dict)
            except Exception:
                args = (in_v.content, self.settings, staging_dict)

            if asyncio.iscoroutinefunction(fn):
                res = await fn(*args)
            else:
                res = fn(*args)

            output_str = str(res) if res is not None else ""

            # Handshake fulfillment: Apply merge strategy
            merge_strategy = self.settings.get("merge_strategy", "overwrite")
            reducer_script_path = self.settings.get("reducer_script")
            reducer_fn = None
            if reducer_script_path:
                try:
                    reducer_fn = _resolve_script_callable(reducer_script_path, ["reduce", "merge"])
                except Exception as e:
                    logger.warning("Failed to resolve reducer script %s: %s", reducer_script_path, e)
            
            store.apply_merge_strategy(
                session_id=session_id,
                name=self.output_vertex,
                incoming_content=output_str,
                strategy=merge_strategy,
                reducer_fn=reducer_fn,
            )

            # Handshake fulfillment: transition to 'data ready' and increment count
            if auto_transition:
                store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.DATA_READY.value)
                store.increment_processed_count(session_id, self.output_vertex)

            return EdgeResultV4(
                edge_id=self.id,
                success=True,
                output=output_str,
                metadata={"length": len(output_str)},
            )

        except Exception as exc:
            err_msg = str(exc)
            logger.error("[CodeEdgeV4:%s] Execution failed: %s", self.id, err_msg)
            # Stage failure diagnostic attributed to this edge
            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="error_feedback",
                value=err_msg,
                vertex_name=self.output_vertex,
                metadata={"exception_type": type(exc).__name__},
            )
            # Transition downstream state to 'reject'
            store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.REJECT.value)

            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                error=err_msg,
                reason="Execution threw exception",
            )
