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
                 prompt="do it", model="gpt-4", settings={"k": "v"})
        assert e.channel == "msg"
        assert e.prompt == "do it"
        assert e.model == "gpt-4"
        assert e.settings == {"k": "v"}


# ── execution ────────────────────────────────────────────────────
class TestEdgeExecution:
    @pytest.mark.asyncio
    async def test_basic_execute(self, mock_agent):
        src = Vertex("src", initial_data=[{"data_id": "d", "tags": [], "value": "hi"}])
        dst = Vertex("dst")
        dst.required_input_count = 1
        dst.incoming_edges = ["e1"]

        e = Edge("e1", "src", "dst", channel="d", prompt="process", model="mock")
        result = await e.execute(src, dst, mock_agent)

        assert e.completed
        assert result is not None
        assert await dst.handle_edge_signal("", EdgeSignal.READ, channel="d") is not None

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

        e = Edge("e", "src", "dst", channel="j", prompt="p", model="m")
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
        e = Edge("e", "src", "dst", channel="d", prompt="trigger")

        with pytest.raises(RuntimeError, match="agent error"):
            await e.execute(src, dst, agent)

        assert not e.completed
        assert "agent error" in e.error


# ── script hooks ─────────────────────────────────────────────────
class TestEdgeScripts:
    @pytest.mark.asyncio
    async def test_pre_post_process(self, echo_agent, tmp_path):
        script = tmp_path / "wrap.py"
        script.write_text(
            "def pre_process(data, settings):\n"
            "    return f'PRE:{data}'\n"
            "\n"
            "def post_process(data, settings):\n"
            "    return f'{data}:POST'\n"
        )
        from framework.script_loader import load_script

        src = Vertex("src", initial_data=[{"data_id": "d", "value": "x"}])
        dst = Vertex("dst")
        dst.required_input_count = 1
        dst.incoming_edges = ["e"]

        e = Edge("e", "src", "dst", channel="d")
        e.set_script_module(load_script(str(script)))

        result = await e.execute(src, dst, echo_agent)
        # echo_agent returns data unchanged, so result = post_process(pre_process("x"))
        assert result == "PRE:x:POST"
        assert await dst.handle_edge_signal("", EdgeSignal.READ, channel="d") == "PRE:x:POST"


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
