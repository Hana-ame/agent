"""Tests for the VEA OpenAI-compatible serve layer.

Covers:
- Schema validation (ChatCompletionRequest, ChatCompletionResponse, Chunk)
- GraphRegistry add/resolve/get
- create_app health endpoint
- /v1/chat/completions non-streaming (MockAgent)
- /v1/chat/completions streaming (MockAgent)
- Multi-model routing (different model → different graph)
- Unknown model → default graph
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from framework import Graph, Executor, MockAgent, Vertex, Edge
from framework.serve import (
    GraphRegistry,
    create_app,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatCompletionChoice,
    UsageInfo,
)
from framework.serve.app import _build_agent, _build_sse_chunk
from framework.serve.schemas import _new_id


# =====================================================================
# Schema tests
# =====================================================================

class TestSchemas:
    """Validate OpenAI-compatible Pydantic models."""

    def test_request_minimal(self):
        """Minimal request with just model + messages."""
        req = ChatCompletionRequest(
            model="default",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.model == "default"
        assert req.messages[0].role == "user"
        assert req.stream is False

    def test_request_streaming(self):
        """Streaming request has stream=True."""
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        assert req.stream is True

    def test_request_optional_params(self):
        """Optional params are default None."""
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.temperature is None
        assert req.max_tokens is None
        assert req.top_p is None
        assert req.graph_id is None
        assert req.stream_mode is None

    def test_response_minimal(self):
        """Response serialises correctly."""
        resp = ChatCompletionResponse(
            id="test-id",
            model="default",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": "hello"},
                    finish_reason="stop",
                )
            ],
        )
        data = resp.model_dump()
        assert data["id"] == "test-id"
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "hello"

    def test_response_with_usage(self):
        """Response includes usage stats."""
        resp = ChatCompletionResponse(
            id="test-id",
            model="default",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": "hello"},
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        data = resp.model_dump()
        assert data["usage"]["total_tokens"] == 15

    def test_chunk_to_sse(self):
        """Chunk serialises to SSE data line."""
        chunk = ChatCompletionChunk(
            id="test-id",
            model="default",
            choices=[
                {"index": 0, "delta": {"content": "hi"}, "finish_reason": None}
            ],
        )
        sse = chunk.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        payload = json.loads(sse[6:].strip())
        assert payload["object"] == "chat.completion.chunk"
        assert payload["choices"][0]["delta"]["content"] == "hi"

    def test_new_id_format(self):
        """_new_id returns chatcmpl- prefixed id."""
        id_ = _new_id()
        assert id_.startswith("chatcmpl-")
        assert len(id_) > 10


# =====================================================================
# GraphRegistry tests
# =====================================================================

class TestGraphRegistry:
    """Test the named graph registry."""

    def _sample_config(self) -> Dict:
        return {
            "metadata": {"name": "test"},
            "vertices": [
                {"id": "src", "settings": {"type": "source"},
                 "initial_data": [{"channel": "text", "value": ""}]},
                {"id": "sink", "settings": {"type": "sink"}},
            ],
            "edges": [
                {"id": "e1", "source": "src", "destination": "sink",
                 "channel": "text", "settings": {"prompt": "say", "model": "mock"}},
            ],
        }

    def test_add_and_get(self):
        reg = GraphRegistry()
        reg.add("m1", self._sample_config())
        entry = reg.get("m1")
        assert entry is not None
        assert entry["config"]["metadata"]["name"] == "test"

    def test_resolve_fallback(self):
        reg = GraphRegistry(default_model="m1")
        reg.add("m1", self._sample_config())
        entry = reg.resolve("unknown")
        assert entry["config"]["metadata"]["name"] == "test"

    def test_resolve_exact(self):
        reg = GraphRegistry(default_model="m1")
        reg.add("m1", self._sample_config())
        reg.add("m2", self._sample_config())
        entry = reg.resolve("m2")
        assert entry["config"]["metadata"]["name"] == "test"

    def test_model_names(self):
        reg = GraphRegistry()
        reg.add("a", self._sample_config())
        reg.add("b", self._sample_config())
        names = reg.model_names
        assert "a" in names
        assert "b" in names

    def test_empty_registry(self):
        reg = GraphRegistry()
        assert not reg
        assert len(reg) == 0

    def test_non_empty_registry(self):
        reg = GraphRegistry()
        reg.add("a", self._sample_config())
        assert reg
        assert len(reg) == 1

    def test_resolve_no_default_raises(self):
        reg = GraphRegistry()
        with pytest.raises(KeyError):
            reg.resolve("anything")

    def test_add_with_agent_config(self):
        reg = GraphRegistry()
        reg.add("m1", self._sample_config(),
                agent={"base_url": "http://localhost:1234"},
                fallback=["fallback_model"],
                stream="tokens")
        entry = reg.get("m1")
        assert entry["agent"]["base_url"] == "http://localhost:1234"
        assert entry["fallback"] == ["fallback_model"]
        assert entry["stream"] == "tokens"

    def test_add_from_json(self, tmp_path):
        config = self._sample_config()
        path = tmp_path / "graph.json"
        with open(path, "w") as f:
            json.dump(config, f)
        reg = GraphRegistry()
        reg.add_from_json("m1", str(path))
        entry = reg.get("m1")
        assert entry["config"]["metadata"]["name"] == "test"
        assert entry["base_dir"] == str(tmp_path)


# =====================================================================
# App integration tests (using MockAgent)
# =====================================================================

def _mock_config(prompt: str = "Say hi") -> Dict:
    """Build a simple single-edge graph config for testing."""
    return {
        "metadata": {"name": "test_graph"},
        "vertices": [
            {"id": "src", "settings": {"type": "source"},
             "initial_data": [{"channel": "text", "value": ""}]},
            {"id": "sink", "settings": {"type": "sink"}},
        ],
        "edges": [
            {"id": "e1", "source": "src", "destination": "sink",
             "channel": "text", "concurrency_type": "llm",
             "settings": {"prompt": prompt, "model": "mock"}},
        ],
    }


def _echo_agent() -> MockAgent:
    """MockAgent that returns the input data as a string."""
    return MockAgent(response_fn=lambda d, p, m, s: f"{p}: {d}" if isinstance(d, (str, dict)) else str(d))


class TestAppIntegration:
    """Integration tests for the FastAPI application using MockAgent."""

    def _make_app(self, **kwargs) -> TestClient:
        reg = GraphRegistry(default_model="default", agent_defaults={
            "base_url": "http://mock",
        })
        reg.add("default", _mock_config(),
                agent_cls=MockAgent, stream="events")
        app = create_app(registry=reg)
        return TestClient(app)

    def test_health(self):
        client = self._make_app()
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "default" in data["models"]

    def test_non_streaming(self):
        client = self._make_app()
        resp = client.post("/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        # MockAgent returns "[model] data" for string input
        content = data["choices"][0]["message"]["content"]
        assert "Hello" in content

    def test_non_streaming_with_usage(self):
        client = self._make_app()
        resp = client.post("/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        data = resp.json()
        assert "usage" in data
        # MockAgent doesn't log real usage, so tokens should be 0
        assert data["usage"]["total_tokens"] == 0

    def test_streaming(self):
        client = self._make_app()
        resp = client.post("/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/event-stream" in ct
        # Should contain SSE data lines
        body = resp.text
        assert "data:" in body

    def test_unknown_model_falls_back(self):
        client = self._make_app()
        resp = client.post("/v1/chat/completions", json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"

    def test_empty_messages_rejected(self):
        client = self._make_app()
        resp = client.post("/v1/chat/completions", json={
            "model": "default",
            "messages": [],
        })
        assert resp.status_code == 422


# =====================================================================
# Multi-model routing tests
# =====================================================================

class TestMultiModelRouting:
    """Test that different model names route to different graphs."""

    def _make_multi_app(self) -> TestClient:
        reg = GraphRegistry(default_model="fast", agent_defaults={
            "base_url": "http://mock",
        })
        # Fast graph: single edge with distinct prompt
        reg.add("fast", _mock_config("Fast answer"),
                agent_cls=MockAgent, stream="events")
        # Slow graph: two edges
        slow_config = {
            "metadata": {"name": "slow"},
            "vertices": [
                {"id": "src", "settings": {"type": "source"},
                 "initial_data": [{"channel": "text", "value": ""}]},
                {"id": "mid", "settings": {}},
                {"id": "sink", "settings": {"type": "sink"}},
            ],
            "edges": [
                {"id": "e1", "source": "src", "destination": "mid",
                 "channel": "text",
                 "settings": {"prompt": "Preprocess", "model": "mock"}},
                {"id": "e2", "source": "mid", "destination": "sink",
                 "channel": "text",
                 "settings": {"prompt": "Answer", "model": "mock"}},
            ],
        }
        reg.add("slow", slow_config,
                agent_cls=MockAgent, stream="events")
        app = create_app(registry=reg)
        return TestClient(app)

    def test_fast_model(self):
        client = self._make_multi_app()
        resp = client.post("/v1/chat/completions", json={
            "model": "fast",
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # MockAgent returns "[model] data" for string input
        assert "test" in content

    def test_slow_model(self):
        client = self._make_multi_app()
        resp = client.post("/v1/chat/completions", json={
            "model": "slow",
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_default_model_is_fast(self):
        client = self._make_multi_app()
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert "test" in content


# =====================================================================
# Fallback tests
# =====================================================================

class TestFallback:
    """Test fallback behavior when primary model fails."""

    def test_fallback_field_in_registry(self):
        reg = GraphRegistry()
        reg.add("primary", {"vertices": [], "edges": []},
                fallback=["backup", "last_resort"])
        entry = reg.get("primary")
        assert entry["fallback"] == ["backup", "last_resort"]


# =====================================================================
# Edge case tests
# =====================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def _make_app(self, **kwargs) -> TestClient:
        reg = GraphRegistry(default_model="default", agent_defaults={
            "base_url": "http://mock",
        })
        reg.add("default", _mock_config(),
                agent_cls=MockAgent, stream="events")
        app = create_app(registry=reg)
        return TestClient(app)

    def test_no_graphs_registered(self):
        reg = GraphRegistry()
        app = create_app(registry=reg)
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 404

    def test_graph_id_override(self):
        reg = GraphRegistry(default_model="default", agent_defaults={
            "base_url": "http://mock",
        })
        reg.add("g1", _mock_config(), agent_cls=MockAgent)
        reg.add("g2", _mock_config(), agent_cls=MockAgent)
        app = create_app(registry=reg)
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={
            "model": "g1",
            "graph_id": "g2",
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 200

    def test_system_message_included(self):
        reg = GraphRegistry(default_model="default", agent_defaults={
            "base_url": "http://mock",
        })
        reg.add("default", _mock_config("Be a cat girl"),
                agent_cls=MockAgent)
        app = create_app(registry=reg)
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={
            "model": "default",
            "messages": [
                {"role": "system", "content": "You are a cat girl"},
                {"role": "user", "content": "Hello"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert "Hello" in content

    def test_response_created_timestamp(self):
        client = self._make_app()
        resp = client.post("/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": "hi"}],
        })
        data = resp.json()
        assert "created" in data
        assert isinstance(data["created"], int)
        assert data["created"] > 1_000_000_000

    def test_sse_chunk_format(self):
        chunk = _build_sse_chunk("test-id", "model-1", content="hello")
        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")
        payload = json.loads(chunk[6:].strip())
        assert payload["id"] == "test-id"
        assert payload["model"] == "model-1"
        assert payload["choices"][0]["delta"]["content"] == "hello"

    def test_sse_done_chunk(self):
        chunk = _build_sse_chunk("test-id", "model-1", finish_reason="stop")
        payload = json.loads(chunk[6:].strip())
        assert payload["choices"][0]["finish_reason"] == "stop"
