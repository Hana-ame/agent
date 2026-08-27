"""Tests for framework.vertex."""

import asyncio
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.vertex import Vertex, VertexState, DataRejectedError


# ── state machine ────────────────────────────────────────────────
class TestVertexState:
    def test_initial_state_idle(self, empty_vertex):
        assert empty_vertex.state == VertexState.IDLE

    def test_state_transition(self, empty_vertex):
        empty_vertex.state = VertexState.READY
        assert empty_vertex.state == VertexState.READY

    def test_reset(self, empty_vertex):
        empty_vertex.state = VertexState.DONE
        empty_vertex.reset()
        assert empty_vertex.state == VertexState.IDLE

    def test_all_states_reachable(self, empty_vertex):
        for st in VertexState:
            empty_vertex.state = st
            assert empty_vertex.state == st


# ── initial data ─────────────────────────────────────────────────
class TestVertexInitialData:
    def test_initial_data_loaded(self, source_vertex):
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(source_vertex.get("text", ["en"]))
        assert data == "Hello world"

    def test_missing_key_returns_none(self, source_vertex):
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(source_vertex.get("missing"))
        assert data is None


# ── get / set ────────────────────────────────────────────────────
class TestVertexGetSet:
    @pytest.mark.asyncio
    async def test_set_and_get(self, empty_vertex):
        await empty_vertex.set("value1", "key1", ["tag_a"])
        result = await empty_vertex.get("key1", ["tag_a"])
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_overwrite(self, empty_vertex):
        await empty_vertex.set("old", "k")
        await empty_vertex.set("new", "k")
        assert await empty_vertex.get("k") == "new"

    @pytest.mark.asyncio
    async def test_tag_order_irrelevant(self, empty_vertex):
        await empty_vertex.set("data", "id", ["b", "a"])
        result = await empty_vertex.get("id", ["a", "b"])
        assert result == "data"

    @pytest.mark.asyncio
    async def test_get_all_data(self, empty_vertex):
        await empty_vertex.set("v1", "k1", ["t"])
        await empty_vertex.set("v2", "k2", ["t"])
        all_data = await empty_vertex.get_all_data()
        assert len(all_data) == 2


# ── readiness semaphore ──────────────────────────────────────────
class TestVertexReadiness:
    @pytest.mark.asyncio
    async def test_becomes_ready_after_all_inputs(self):
        v = Vertex("v1")
        v.required_input_count = 2
        v.incoming_edges = ["e1", "e2"]

        await v.set("a", "d1", edge_id="e1")
        assert v.state == VertexState.IDLE  # only 1 of 2

        await v.set("b", "d2", edge_id="e2")
        assert v.state == VertexState.READY  # 2 of 2

    @pytest.mark.asyncio
    async def test_source_vertex_has_no_required_inputs(self, source_vertex):
        assert source_vertex.required_input_count == 0
        assert source_vertex.is_source()


# ── external script hooks ────────────────────────────────────────
class TestVertexScript:
    @pytest.mark.asyncio
    async def test_on_receive_transforms(self, empty_vertex, tmp_path):
        script = tmp_path / "upper.py"
        script.write_text(
            "def on_receive(data, data_id, tags, settings):\n"
            "    return data.upper() if isinstance(data, str) else data\n"
        )
        from framework.script_loader import load_script
        empty_vertex.set_script_module(load_script(str(script)))

        await empty_vertex.set("hello", "k")
        assert await empty_vertex.get("k") == "HELLO"

    @pytest.mark.asyncio
    async def test_on_receive_rejects(self, empty_vertex, tmp_path):
        script = tmp_path / "reject.py"
        script.write_text(
            "def on_receive(data, data_id, tags, settings):\n"
            "    raise ValueError('rejected')\n"
        )
        from framework.script_loader import load_script
        empty_vertex.set_script_module(load_script(str(script)))

        with pytest.raises(DataRejectedError, match="rejected"):
            await empty_vertex.set("anything", "k")

    @pytest.mark.asyncio
    async def test_on_ready_hook(self, empty_vertex, tmp_path):
        script = tmp_path / "ready.py"
        script.write_text(
            "def on_ready(all_data, settings):\n"
            "    return {('out', ('merged',)): 'merged-data'}\n"
        )
        from framework.script_loader import load_script
        empty_vertex.set_script_module(load_script(str(script)))

        await empty_vertex.set("raw", "in")
        await empty_vertex.prepare_outputs()

        assert await empty_vertex.get("out", ["merged"]) == "merged-data"


# ── helpers ──────────────────────────────────────────────────────
class TestVertexHelpers:
    def test_is_source(self, source_vertex):
        assert source_vertex.is_source()

    def test_is_sink(self, sink_vertex):
        assert sink_vertex.is_sink()

    def test_repr(self, source_vertex):
        r = repr(source_vertex)
        assert "src" in r
        assert "idle" in r
