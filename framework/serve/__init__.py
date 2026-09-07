"""OpenAI-compatible HTTP serve layer for VEA graphs.

This module provides a FastAPI application that exposes VEA graph execution
as an OpenAI-compatible ``/v1/chat/completions`` endpoint.  It is purely
additive: no existing VEA files are modified.

Key features:
- **Multi-graph routing** — different ``model`` names map to different
  VEA graph configs + agent configs.
- **SSE streaming** — ``stream: true`` yields OpenAI-format ``data:``
  chunks via Server-Sent Events.
- **Fallback** — on graph/LLM failure, transparently retries with a
  fallback model (configurable per-route).
- **Graph observability** — ``Executor.stream()`` events are relayed
  as SSE chunks in graph mode; token-level streaming is available in
  direct mode via ``HttpLLMAgent.stream_process()``.

Usage::

    from framework.serve import create_app

    app = create_app(
        graphs={
            "default": {"config": graph_config_dict, "agent": agent_config},
            "quick":   {"config": quick_config_dict, "agent": quick_agent_config},
        },
    )

    # Run with uvicorn:
    #   uvicorn framework.serve.app:app --port 8000
"""

from .app import GraphRegistry, create_app
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)

__all__ = [
    "GraphRegistry",
    "create_app",
    "ChatCompletionChunk",
    "ChatCompletionChoice",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "UsageInfo",
]
