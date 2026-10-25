"""OpenAI-compatible request/response schemas for the VEA serve layer.

These Pydantic models mirror the OpenAI Chat Completions API v1 format so
that any OpenAI client library can talk to a VEA graph without modification.

Reference: https://platform.openai.com/docs/api-reference/chat
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def _new_id(prefix: str = "chatcmpl") -> str:
    """Generate a unique request id (OpenAI-style ``chatcmpl-xxxxxxxx``)."""
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single chat message in an OpenAI chat completion request."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    ``model`` selects the VEA graph to execute.  Unknown model names
    fall back to ``"default"``.
    """
    model: str = "default"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[str | List[str]] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None

    # VEA-specific extensions (ignored by OpenAI clients, safe to omit)
    graph_id: Optional[str] = None          # explicit graph override
    stream_mode: Optional[str] = None       # "events" | "tokens" | "auto"


# ---------------------------------------------------------------------------
# Response (non-streaming)
# ---------------------------------------------------------------------------

class ChatCompletionChoice(BaseModel):
    """A single completion choice in an OpenAI response."""
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = "stop"
    logprobs: Optional[Any] = None


class UsageInfo(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response (non-streaming)."""
    id: str = Field(default_factory=_new_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: List[ChatCompletionChoice] = Field(default_factory=list)
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


# ---------------------------------------------------------------------------
# Response (streaming — SSE chunks)
# ---------------------------------------------------------------------------

class ChunkDelta(BaseModel):
    """Incremental content delta for a streaming chunk."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChunkChoice(BaseModel):
    """A single streaming choice."""
    index: int = 0
    delta: ChunkDelta
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk (one SSE ``data:`` line)."""
    id: str = Field(default_factory=_new_id)
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: List[ChunkChoice] = Field(default_factory=list)

    def to_sse(self) -> str:
        """Serialise to a single SSE ``data:`` line (no trailing ``[DONE]``)."""
        return f"data: {self.model_dump_json()}\n\n"
