"""V4 Backward-Compatibility Façade for Workflow Execution and SSE Streaming.

Delegates core DAG workflow orchestration to WorkflowExecutorV4 and
HTTP/SSE protocol formatting to framework.server.sse.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, Union

from framework.graph_v4 import GraphV4
from framework.server.manager import SessionGraphManagerV4
from framework.server.sse import (
    SSE_DONE,
    ToolCallEcho,
    format_result_as_harness_echo,
    format_sse_line,
    stream_workflow_as_sse,
)
from framework.vertex_v4 import VertexStoreV4
from framework.workflow_executor_v4 import WorkflowExecutorV4

logger = logging.getLogger("vertex_edge_agent.sse_executor_v4")

# Re-export ToolCallEcho for 100% backward compatibility
__all__ = ["ToolCallEcho", "SSEExecutorV4", "SSE_DONE", "format_sse_line", "stream_workflow_as_sse"]


class SSEExecutorV4:
    """Backward-compatible wrapper integrating WorkflowExecutorV4 with server SSE streaming."""

    def __init__(
        self,
        manager: SessionGraphManagerV4,
        store: Optional[VertexStoreV4] = None,
        default_manifest: Optional[Union[str, Path]] = None,
    ):
        self._orchestrator = WorkflowExecutorV4(
            manager=manager,
            store=store,
            default_manifest=default_manifest,
        )

    @property
    def manager(self) -> SessionGraphManagerV4:
        return self._orchestrator.manager

    @property
    def store(self) -> VertexStoreV4:
        return self._orchestrator.store

    @property
    def default_manifest(self) -> Optional[Union[str, Path]]:
        return self._orchestrator.default_manifest

    def resolve_session_and_graph(
        self,
        session_id: Optional[str] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[str, GraphV4]:
        return self._orchestrator.resolve_session_and_graph(session_id, manifest_path)

    def resolve_subgraph_vertices(self, session_id: str, graph: GraphV4) -> None:
        return self._orchestrator.resolve_subgraph_vertices(session_id, graph)

    def _inject_input_payload(
        self,
        session_id: str,
        graph: GraphV4,
        input_payload: Optional[Union[str, Dict[str, Any]]],
    ) -> None:
        return self._orchestrator._inject_input_payload(session_id, graph, input_payload)

    async def execute_and_stream(
        self,
        session_id: Optional[str] = None,
        input_payload: Optional[Union[str, Dict[str, Any]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        max_concurrency: int = 4,
        timeout: float = 120.0,
    ) -> AsyncGenerator[str, None]:
        """Execute workflow and yield SSE chunks."""
        workflow_stream = self._orchestrator.stream(
            session_id=session_id,
            input_payload=input_payload,
            manifest_path=manifest_path,
            max_concurrency=max_concurrency,
            timeout=timeout,
        )
        async for chunk in stream_workflow_as_sse(workflow_stream):
            yield chunk

    async def execute_harness_call(
        self,
        session_id: Optional[str] = None,
        input_payload: Optional[Union[str, Dict[str, Any]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        max_concurrency: int = 4,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Execute workflow and return OpenAI tool-call payload."""
        call_id = f"call_{uuid.uuid4().hex[:10]}"
        try:
            result = await self._orchestrator.run(
                session_id=session_id,
                input_payload=input_payload,
                manifest_path=manifest_path,
                max_concurrency=max_concurrency,
                timeout=timeout,
            )
            return format_result_as_harness_echo(result, call_id=call_id)
        except Exception as exc:
            logger.error("[SSEExecutorV4] execute_harness_call failed: %s", exc)
            echo = ToolCallEcho(
                call_id=call_id,
                function_name="echo",
                arguments={
                    "info": {
                        "session_id": session_id,
                        "success": False,
                        "error": str(exc),
                    }
                },
            )
            return echo.to_openai_dict()
