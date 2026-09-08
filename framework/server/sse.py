"""Server-Sent Events (SSE) and OpenAI ToolCall Echo Presentation Adapter.

Handles HTTP-layer streaming formatting (text/event-stream protocol) and
OpenAI-compatible chat completion chunk payload packaging.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional

from framework.workflow_executor_v4 import WorkflowEventV4, WorkflowResultV4


@dataclass
class ToolCallEcho:
    """Tool call payload formatted for agent harness frameworks (OpenAI compatible)."""

    call_id: str
    function_name: str = "echo"
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_openai_dict(self) -> Dict[str, Any]:
        """Format as standard OpenAI tool_call response dictionary."""
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.function_name,
                "arguments": json.dumps(self.arguments),
            },
        }

    def to_sse_chunk(self, chunk_id: str, finish_reason: Optional[str] = None) -> str:
        """Format as OpenAI streaming chunk SSE data line (data: ...\\n\\n)."""
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "vea-v4-sse-executor",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": self.call_id,
                                "type": "function",
                                "function": {
                                    "name": self.function_name,
                                    "arguments": json.dumps(self.arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"


SSE_DONE = "data: [DONE]\n\n"


def format_sse_line(data: Any) -> str:
    """Format an arbitrary object or string as an SSE data frame."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"data: {payload}\n\n"


async def stream_workflow_as_sse(
    workflow_stream: AsyncGenerator[WorkflowEventV4, None],
    call_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Adapter translating pure WorkflowEventV4 events into SSE text/event-stream chunks."""
    c_id = chunk_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    cl_id = call_id or f"call_{uuid.uuid4().hex[:10]}"

    try:
        async for event in workflow_stream:
            if event.event_type == "workflow_started":
                info: Dict[str, Any] = {
                    "event": "workflow_started",
                    "session_id": event.session_id,
                    "concurrency": event.metadata.get("concurrency", 4),
                    "graph_vertices": event.metadata.get("graph_vertices", []),
                }
                finish_reason = None
            elif event.event_type == "workflow_finished":
                info = {
                    "event": "workflow_finished",
                    "session_id": event.session_id,
                    **(dict(event.payload) if isinstance(event.payload, dict) else {}),
                }
                finish_reason = "tool_calls"
            elif event.event_type in ("setup_error", "stream_error"):
                info = {
                    "event": event.event_type,
                    "error": event.payload.get("error", str(event.payload)) if isinstance(event.payload, dict) else str(event.payload),
                }
                finish_reason = "tool_calls"
            else:
                info = {
                    "event": event.event_type,
                    "session_id": event.session_id,
                    "edge_id": event.edge_id,
                    "vertex_name": event.vertex_name,
                    "payload": event.payload,
                }
                finish_reason = None

            echo_chunk = ToolCallEcho(
                call_id=cl_id,
                function_name="echo",
                arguments={"info": info},
            )
            yield echo_chunk.to_sse_chunk(c_id, finish_reason=finish_reason)
    finally:
        yield SSE_DONE


def format_result_as_harness_echo(
    result: WorkflowResultV4,
    call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Format a WorkflowResultV4 as a standard non-streaming OpenAI tool call dictionary."""
    cl_id = call_id or f"call_{uuid.uuid4().hex[:10]}"
    echo = ToolCallEcho(
        call_id=cl_id,
        function_name="echo",
        arguments={
            "info": {
                "session_id": result.session_id,
                "success": result.success,
                "execution_time": result.execution_time,
                "completed_edges": result.completed_edges,
                "vertex_states": result.vertex_states,
                "vertex_contents": result.vertex_contents,
                "errors": result.errors,
            }
        },
    )
    return echo.to_openai_dict()
