"""Tests for framework.edge."""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.edge import Edge
from framework.vertex import Vertex, DataRejectedError, EdgeSignal
from framework.agents import MockAgent


# ── construction ─────────────────────────────────────────────────
class TestEdgeConstruction:
    def test_defaults(self):
        e = Edge("e1", "src", "dst")
        assert e.id == "e1"
        assert e.source_id == "src"
        assert e.destination_id == "dst"
        assert e.channel == "default"
        assert e.completed is False
        assert e.error is None

    def test_custom_fields(self):
        e = Edge("e2", "a", "b", channel="msg", 
                 settings={"k": "v", "prompt": "do it", "model": "gpt-4"})
        assert e.channel == "msg"
        assert e.prompt == "do it"
        assert e.model == "gpt-4"
        assert e.settings["k"] == "v"


# ── execution ────────────────────────────────────────────────────
class TestEdgeExecution:
    @pytest.mark.asyncio
    async def test_basic_execute(self, mock_agent):
        src = Vertex("src", initial_data=[{"data_id": "d", "tags": [], "value": "hi"}])
        dst = Vertex("dst")
        dst.required_input_count = 1
        dst.incoming_edges = ["e1"]

        e = Edge("e1", "src", "dst", channel="d", settings={"prompt": "process", "model": "mock"})
        result = await e.execute(src, dst, mock_agent)

        assert e.completed
        assert result is not None
        assert await dst.fetch_data(channel="d") is not None

    @pytest.mark.asyncio
    async def test_none_data_propagates(self, mock_agent):
        """Edge should still work when source returns None."""
        src = Vertex("src")
        dst = Vertex("dst")
        dst.required_input_count = 1
        dst.incoming_edges = ["e"]

        e = Edge("e", "src", "dst", channel="missing")
        result = await e.execute(src, dst, mock_agent)
        assert e.completed

    @pytest.mark.asyncio
    async def test_execute_with_dict_data(self, mock_agent):
        src = Vertex("src", initial_data=[
            {"data_id": "j", "tags": [], "value": {"key": "val"}}
        ])
        dst = Vertex("dst")
        dst.required_input_count = 1
        dst.incoming_edges = ["e"]

        e = Edge("e", "src", "dst", channel="j", settings={"prompt": "p", "model": "m"})
        result = await e.execute(src, dst, mock_agent)
        assert e.completed
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_execute_failure_sets_error(self):
        """When the agent raises, the edge should record the error."""
        def boom(d, p, m, s):
            raise RuntimeError("agent error")

        src = Vertex("src", initial_data=[{"data_id": "d", "value": "x"}])
        dst = Vertex("dst")
        agent = MockAgent(response_fn=boom)
        e = Edge("e", "src", "dst", channel="d", settings={"prompt": "trigger"})

        with pytest.raises(RuntimeError, match="agent error"):
            await e.execute(src, dst, agent)

        assert not e.completed
        assert "agent error" in e.error


# ── script hooks ─────────────────────────────────────────────────
class TestEdgeScripts:
    @pytest.mark.asyncio
    async def test_pre_post_process(self, echo_agent, tmp_path):
        class WrapEdge(Edge):
            def pre_process(self, data, settings):
                return f"PRE:{data}"
            def post_process(self, result, settings):
                return f"{result}:POST"

        src = Vertex("src", initial_data=[{"data_id": "d", "value": "x"}])
        dst = Vertex("dst")
        dst.required_input_count = 1
        dst.incoming_edges = ["e"]

        e = WrapEdge("e", "src", "dst", channel="d")

        result = await e.execute(src, dst, echo_agent)
        # echo_agent returns data unchanged, so result = post_process(pre_process("x"))
        assert result == "PRE:x:POST"
        assert await dst.fetch_data(channel="d") == "PRE:x:POST"


# ── reset ────────────────────────────────────────────────────────
class TestEdgeReset:
    def test_reset_clears_state(self):
        e = Edge("e", "a", "b")
        e.completed = True
        e.result = "something"
        e.error = "err"
        e.reset()
        assert not e.completed
        assert e.result is None
        assert e.error is None


# ── per-edge agent override ───────────────────────────────────
class TestEdgeAgentOverride:
    """``settings.agent`` overrides the executor-level agent (most specific wins)."""

    @pytest.mark.asyncio
    async def test_per_edge_agent_wins_over_executor_agent(self):
        from framework.agents import MockAgent

        executor_agent = MockAgent(response_fn=lambda d, p, m, s: f"EXEC:{d}")
        edge_agent = MockAgent(response_fn=lambda d, p, m, s: f"EDGE:{d}")

        src = Vertex("src", initial_data=[{"data_id": "d", "value": "hi"}])
        dst = Vertex("dst")
        dst.required_input_count = 1
        dst.incoming_edges = ["e"]

        e = Edge("e", "src", "dst", channel="d",
                 settings={"prompt": "p", "model": "m", "agent": edge_agent})
        assert e.agent is edge_agent
        result = await e.execute(src, dst, executor_agent)
        assert result == "EDGE:hi"  # per-edge override wins

    @pytest.mark.asyncio
    async def test_executor_agent_used_when_no_override(self):
        from framework.agents import MockAgent

        executor_agent = MockAgent(response_fn=lambda d, p, m, s: f"EXEC:{d}")
        src = Vertex("src", initial_data=[{"data_id": "d", "value": "hi"}])
        dst = Vertex("dst")
        dst.required_input_count = 1
        dst.incoming_edges = ["e"]

        e = Edge("e", "src", "dst", channel="d",
                 settings={"prompt": "p", "model": "m"})
        assert e.agent is None
        result = await e.execute(src, dst, executor_agent)
        assert result == "EXEC:hi"

    @pytest.mark.asyncio
    async def test_string_agent_spec_resolved(self):
        """``"agent": "mock"`` in settings resolves to a live agent."""
        src = Vertex("src", initial_data=[{"data_id": "d", "value": "hi"}])
        dst = Vertex("dst")
        dst.required_input_count = 1
        dst.incoming_edges = ["e"]

        e = Edge("e", "src", "dst", channel="d",
                 settings={"prompt": "p", "model": "m", "agent": "mock"})
        assert isinstance(e.agent, MockAgent)
        # No executor agent passed — per-edge resolved agent is used.
        result = await e.execute(src, dst, None)
        assert result == "[m] hi"
