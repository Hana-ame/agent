"""Tests for SenseNovaEdgeV4 — the V4 SenseNova 6.8 Flash Lite dedicated edge.

Tests cover:
1. Construction-time API key validation (raises if SENSENOVA_API_KEY missing).
2. Handshake compliance (skips when upstream not ready, downstream not demanded).
3. Successful execution with a mock agent.
4. Error handling: rejects downstream and stages error_feedback.
5. JSON validation when downstream vertex has JSON attribute.
6. Streaming mode with a mock agent.
7. Standalone CLI (--help at minimum).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from framework.edge_v4 import EdgeResultV4
from framework.sensenova_edge_v4 import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    SensenovaEdgeV4,
)
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexStateV4,
    VertexStoreV4,
)


@pytest.fixture
def mem_store() -> VertexStoreV4:
    """Fixture providing an in-memory SQLite store."""
    store = VertexStoreV4(":memory:")
    yield store
    store.close()


@pytest.fixture
def env_key(monkeypatch):
    """Ensure SENSENOVA_API_KEY is set for construction tests."""
    monkeypatch.setenv("SENSENOVA_API_KEY", "sk-test-key-12345")
    yield "sk-test-key-12345"


# =====================================================================
# 1. Construction & Key Validation
# =====================================================================


class TestConstruction:
    """Verify that the edge validates its API key at init time."""

    def test_missing_key_raises(self, monkeypatch):
        """ValueError when SENSENOVA_API_KEY is absent."""
        monkeypatch.delenv("SENSENOVA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SENSENOVA_API_KEY"):
            SensenovaEdgeV4(
                edge_id="e1",
                input_vertex="a",
                output_vertex="b",
            )

    def test_empty_key_raises(self, monkeypatch):
        """ValueError when SENSENOVA_API_KEY is empty string."""
        monkeypatch.setenv("SENSENOVA_API_KEY", "")
        with pytest.raises(ValueError, match="SENSENOVA_API_KEY"):
            SensenovaEdgeV4(
                edge_id="e1",
                input_vertex="a",
                output_vertex="b",
            )

    def test_valid_key_constructs(self, env_key):
        """Edge builds successfully when key is present."""
        edge = SensenovaEdgeV4(
            edge_id="e1",
            input_vertex="a",
            output_vertex="b",
        )
        assert edge.id == "e1"
        assert edge.type == "sensenova"
        assert edge.api_key == "sk-test-key-12345"
        assert edge.base_url == DEFAULT_BASE_URL
        assert edge.model == DEFAULT_MODEL

    def test_custom_model_from_settings(self, env_key):
        """Settings override the default model."""
        edge = SensenovaEdgeV4(
            edge_id="e1",
            input_vertex="a",
            output_vertex="b",
            settings={"model": "sensenova-7.0-pro"},
        )
        assert edge.model == "sensenova-7.0-pro"
        assert edge.settings["model"] == "sensenova-7.0-pro"

    def test_custom_base_url_from_settings(self, env_key):
        """Settings override the default base_url."""
        edge = SensenovaEdgeV4(
            edge_id="e1",
            input_vertex="a",
            output_vertex="b",
            settings={"base_url": "https://custom.moonchan.xyz/v1/chat/completions"},
        )
        assert edge.base_url == "https://custom.moonchan.xyz/v1/chat/completions"

    def test_prompt_template_stored_in_settings(self, env_key):
        """prompt_template argument is written into settings['prompt']."""
        edge = SensenovaEdgeV4(
            edge_id="e1",
            input_vertex="a",
            output_vertex="b",
            prompt_template="You are a translator. Translate: {input}",
        )
        assert edge.settings["prompt"] == "You are a translator. Translate: {input}"

    def test_api_key_not_in_settings(self, env_key):
        """The API key must never leak into edge.settings."""
        edge = SensenovaEdgeV4(
            edge_id="e1",
            input_vertex="a",
            output_vertex="b",
        )
        assert "api_key" not in edge.settings
        assert "api_key" not in str(edge.settings)


# =====================================================================
# 2. Handshake Compliance
# =====================================================================


class TestHandshake:
    """The edge must skip cleanly when the two-sided handshake is not met."""

    async def test_skips_when_input_not_data_ready(self, env_key, mem_store):
        """Upstream is 'todo' → edge returns skipped, no state change."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="",
            state=VertexStateV4.TODO,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")
        result = await edge.run("s1", mem_store)
        assert result.skipped
        assert result.success is False
        assert "Upstream" in (result.reason or "")

    async def test_skips_when_output_not_todo(self, env_key, mem_store):
        """Downstream is 'data ready' → edge returns skipped."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="hello",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.DATA_READY,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")
        result = await edge.run("s1", mem_store)
        assert result.skipped
        assert result.success is False
        assert "Downstream" in (result.reason or "")

    async def test_skips_when_output_forbidden(self, env_key, mem_store):
        """Downstream is 'forbidden' → edge returns skipped."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="hello",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.FORBIDDEN,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")
        result = await edge.run("s1", mem_store)
        assert result.skipped
        assert "Downstream" in (result.reason or "")

    async def test_skips_when_output_reject(self, env_key, mem_store):
        """Downstream is 'reject' → edge returns skipped."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="hello",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.REJECT,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")
        result = await edge.run("s1", mem_store)
        assert result.skipped
        assert "Downstream" in (result.reason or "")

    async def test_skips_when_vertex_missing(self, env_key, mem_store):
        """Missing vertex → skipped with not-found reason."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="hello",
            state=VertexStateV4.DATA_READY,
        )
        edge = SensenovaEdgeV4("e1", "up", "ghost")
        result = await edge.run("s1", mem_store)
        assert result.skipped
        assert "not found" in (result.reason or "")


# =====================================================================
# 3. Successful Execution (mock agent)
# =====================================================================


class TestExecution:
    """Test the edge with a mock agent that returns a known response."""

    async def test_success_with_mock_agent(self, env_key, mem_store):
        """Mock agent returns text → downstream becomes data ready."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="What is 1+1?",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        mock_agent = AsyncMock()
        mock_agent.process.return_value = "2"

        result = await edge.run("s1", mem_store, agent=mock_agent)

        assert result.success
        assert result.output == "2"
        assert result.edge_id == "e1"

        # Downstream vertex updated
        out_v = mem_store.get_vertex("s1", "down")
        assert out_v.state == VertexStateV4.DATA_READY.value
        assert out_v.content == "2"
        assert out_v.processed_count == 1

        # Agent was called with correct arguments
        mock_agent.process.assert_called_once()
        call_args = mock_agent.process.call_args
        assert call_args[0][0] == "What is 1+1?"
        assert "What is 1+1?" in call_args[0][1]  # rendered prompt

    async def test_prompt_template_rendering(self, env_key, mem_store):
        """{input} in prompt template is replaced with upstream content."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="hello world",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4(
            "e1", "up", "down",
            prompt_template="Translate this: {input}",
        )

        mock_agent = AsyncMock()
        mock_agent.process.return_value = "hola mundo"

        result = await edge.run("s1", mem_store, agent=mock_agent)

        assert result.success
        call_args = mock_agent.process.call_args
        assert "Translate this: hello world" in call_args[0][1]

    async def test_prompt_staged_for_observability(self, env_key, mem_store):
        """The rendered prompt is staged in session_staging."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="data",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        mock_agent = AsyncMock()
        mock_agent.process.return_value = "result"

        await edge.run("s1", mem_store, agent=mock_agent)

        staged = mem_store.get_latest_staged_for_vertex(
            session_id="s1", vertex_name="down", key="rendered_prompt",
        )
        assert staged is not None
        assert "data" in staged.value

    async def test_agent_with_generate_method(self, env_key, mem_store):
        """Fallback to agent.generate() when process() is absent."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="hi",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        class GenAgent:
            async def generate(self, prompt, model=None, temperature=None):
                return "generated response"

        result = await edge.run("s1", mem_store, agent=GenAgent())
        assert result.success
        assert result.output == "generated response"

    async def test_agent_with_chat_method(self, env_key, mem_store):
        """Fallback to agent.chat() when process()/generate() are absent."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="hi",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        class ChatAgent:
            async def chat(self, messages, model=None, temperature=None):
                return "chat response"

        result = await edge.run("s1", mem_store, agent=ChatAgent())
        assert result.success
        assert result.output == "chat response"


# =====================================================================
# 4. Error Handling
# =====================================================================


class TestErrorHandling:
    """Verify error → reject transition + staged diagnostic."""

    async def test_failure_sets_reject(self, env_key, mem_store):
        """Agent raises → downstream state becomes 'reject'."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="data",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        mock_agent = AsyncMock()
        mock_agent.process.side_effect = RuntimeError("upstream 500")

        result = await edge.run("s1", mem_store, agent=mock_agent)

        assert result.success is False
        assert "upstream 500" in (result.error or "")
        assert "SenseNova" in (result.reason or "")

        out_v = mem_store.get_vertex("s1", "down")
        assert out_v.state == VertexStateV4.REJECT.value

    async def test_failure_stages_error_feedback(self, env_key, mem_store):
        """Error is staged in session_staging with metadata."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="data",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        mock_agent = AsyncMock()
        mock_agent.process.side_effect = ValueError("bad model")

        result = await edge.run("s1", mem_store, agent=mock_agent)

        staged = mem_store.get_latest_staged_for_vertex(
            session_id="s1", vertex_name="down", key="error_feedback",
        )
        assert staged is not None
        assert "bad model" in staged.value
        assert staged.metadata.get("exception") == "ValueError"

    async def test_failure_does_not_crash(self, env_key, mem_store):
        """Edge returns EdgeResultV4 (not raises) on failure."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="data",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        mock_agent = AsyncMock()
        mock_agent.process.side_effect = ConnectionError("network down")

        result = await edge.run("s1", mem_store, agent=mock_agent)
        assert isinstance(result, EdgeResultV4)
        assert result.success is False


# =====================================================================
# 5. JSON Validation
# =====================================================================


class TestJSONValidation:
    """When downstream vertex has JSON attribute, output must be valid JSON."""

    async def test_json_validated(self, env_key, mem_store):
        """Valid JSON passes through cleanly."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="data",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            attributes=[VertexAttributeV4.JSON],
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        mock_agent = AsyncMock()
        mock_agent.process.return_value = '{"key": "value"}'

        result = await edge.run("s1", mem_store, agent=mock_agent)
        assert result.success
        assert result.output == '{"key": "value"}'

    async def test_json_strips_markdown_fence(self, env_key, mem_store):
        """```json fence is stripped before validation."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="data",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            attributes=[VertexAttributeV4.JSON],
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        mock_agent = AsyncMock()
        mock_agent.process.return_value = '```json\n{"a": 1}\n```'

        result = await edge.run("s1", mem_store, agent=mock_agent)
        assert result.success
        assert result.output == '{"a": 1}'

    async def test_malformed_json_raises_reject(self, env_key, mem_store):
        """Invalid JSON → ValueError → reject state."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="data",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            attributes=[VertexAttributeV4.JSON],
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        mock_agent = AsyncMock()
        mock_agent.process.return_value = "not json at all"

        result = await edge.run("s1", mem_store, agent=mock_agent)
        assert result.success is False
        assert "Malformed JSON" in (result.error or "")
        out_v = mem_store.get_vertex("s1", "down")
        assert out_v.state == VertexStateV4.REJECT.value


# =====================================================================
# 6. Streaming
# =====================================================================


class TestStreaming:
    """Test run_stream() with a mock streaming agent."""

    async def test_stream_yields_chunks(self, env_key, mem_store):
        """run_stream yields individual deltas and writes full output."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="data",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        async def fake_stream(*args, **kwargs):
            for chunk in ["Hello", ", ", "world"]:
                yield chunk

        mock_agent = AsyncMock()
        mock_agent.stream_process = fake_stream

        chunks = []
        async for chunk in edge.run_stream("s1", mem_store, agent=mock_agent):
            chunks.append(chunk)

        assert chunks == ["Hello", ", ", "world"]

        out_v = mem_store.get_vertex("s1", "down")
        assert out_v.content == "Hello, world"
        assert out_v.state == VertexStateV4.DATA_READY.value

    async def test_stream_failure_raises(self, env_key, mem_store):
        """Stream error raises, transitions downstream to reject."""
        mem_store.save_vertex(
            session_id="s1", name="up", content="data",
            state=VertexStateV4.DATA_READY,
        )
        mem_store.save_vertex(
            session_id="s1", name="down", content="",
            state=VertexStateV4.TODO,
        )
        edge = SensenovaEdgeV4("e1", "up", "down")

        async def broken_stream(*args, **kwargs):
            yield "partial"
            raise RuntimeError("stream broke")

        mock_agent = AsyncMock()
        mock_agent.stream_process = broken_stream

        with pytest.raises(RuntimeError, match="stream broke"):
            async for _ in edge.run_stream("s1", mem_store, agent=mock_agent):
                pass

        out_v = mem_store.get_vertex("s1", "down")
        assert out_v.state == VertexStateV4.REJECT.value


# =====================================================================
# 7. CLI
# =====================================================================


class TestCLI:
    """Smoke tests for the standalone CLI entrypoint."""

    def test_cli_help(self):
        """--help exits cleanly with usage text."""
        result = subprocess.run(
            [sys.executable, "-m", "framework.sensenova_edge_v4", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "SenseNova" in result.stdout or "sense" in result.stdout.lower()

    def test_cli_missing_key_fails(self):
        """Without SENSENOVA_API_KEY, CLI exits with error."""
        env = {k: v for k, v in os.environ.items() if k != "SENSENOVA_API_KEY"}
        result = subprocess.run(
            [sys.executable, "-m", "framework.sensenova_edge_v4",
             "--session", "test", "--input", "a", "--output", "b"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert result.returncode != 0
        assert "SENSENOVA_API_KEY" in (result.stderr + result.stdout)

    def test_cli_missing_required_args(self):
        """Missing --session/--input/--output → argparse error."""
        env = dict(os.environ)
        env["SENSENOVA_API_KEY"] = "sk-test"
        result = subprocess.run(
            [sys.executable, "-m", "framework.sensenova_edge_v4"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert result.returncode != 0
        assert "required" in (result.stderr + result.stdout).lower()
