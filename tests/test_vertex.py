"""Tests for framework.vertex (data keyed by source edge ID)."""

import os
import sys

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
    @pytest.mark.asyncio
    async def test_initial_data_loaded(self, source_vertex):
        # 初始数据存保留键 __init__，get() 读主数据应取到
        data = await source_vertex.get()
        assert data == "Hello world"

    @pytest.mark.asyncio
    async def test_missing_edge_slot_returns_none(self, source_vertex):
        # 指定一个不存在的来源边 ID，返回 None
        data = await source_vertex.get("no_such_edge")
        assert data is None


# ── get / set (keyed by edge ID) ─────────────────────────────────
class TestVertexGetSet:
    @pytest.mark.asyncio
    async def test_set_and_get_by_edge(self, empty_vertex):
        await empty_vertex.set("value1", edge_id="e1")
        result = await empty_vertex.get("e1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_same_edge_overwrites(self, empty_vertex):
        await empty_vertex.set("old", edge_id="e")
        await empty_vertex.set("new", edge_id="e")
        assert await empty_vertex.get("e") == "new"

    @pytest.mark.asyncio
    async def test_different_edges_do_not_overwrite(self, empty_vertex):
        # fan-in：不同来源边各占一槽，天然不覆盖
        await empty_vertex.set("a", edge_id="e1")
        await empty_vertex.set("b", edge_id="e2")
        assert await empty_vertex.get("e1") == "a"
        assert await empty_vertex.get("e2") == "b"

    @pytest.mark.asyncio
    async def test_get_all_data(self, empty_vertex):
        await empty_vertex.set("v1", edge_id="e1")
        await empty_vertex.set("v2", edge_id="e2")
        all_data = await empty_vertex.get_all_data()
        assert len(all_data) == 2
        assert all_data["e1"] == "v1"
        assert all_data["e2"] == "v2"


# ── readiness ────────────────────────────────────────────────────
class TestVertexReadiness:
    @pytest.mark.asyncio
    async def test_becomes_ready_after_all_inputs(self):
        v = Vertex("v1")
        v.required_input_count = 2
        v.incoming_edges = ["e1", "e2"]

        await v.set("a", edge_id="e1")
        assert v.state == VertexState.IDLE  # only 1 of 2

        await v.set("b", edge_id="e2")
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
            "def on_receive(data, edge_id, tags, settings):\n"
            "    return data.upper() if isinstance(data, str) else data\n"
        )
        from framework.script_loader import load_script
        empty_vertex.set_script_module(load_script(str(script)))

        await empty_vertex.set("hello", edge_id="e")
        assert await empty_vertex.get("e") == "HELLO"

    @pytest.mark.asyncio
    async def test_on_receive_rejects(self, empty_vertex, tmp_path):
        script = tmp_path / "reject.py"
        script.write_text(
            "def on_receive(data, edge_id, tags, settings):\n"
            "    raise ValueError('rejected')\n"
        )
        from framework.script_loader import load_script
        empty_vertex.set_script_module(load_script(str(script)))

        with pytest.raises(DataRejectedError, match="rejected"):
            await empty_vertex.set("anything", edge_id="e")

    @pytest.mark.asyncio
    async def test_on_ready_hook(self, empty_vertex, tmp_path):
        script = tmp_path / "ready.py"
        script.write_text(
            "def on_ready(all_data, settings):\n"
            "    return 'merged-data'\n"
        )
        from framework.script_loader import load_script
        empty_vertex.set_script_module(load_script(str(script)))

        await empty_vertex.set("raw", edge_id="e")
        await empty_vertex.prepare_outputs()

        # on_ready 合并结果存到主数据(__self__)，get() 应读到
        assert await empty_vertex.get() == "merged-data"


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
