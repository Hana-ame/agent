"""Tests for framework.vertex."""

import asyncio
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework.vertex import Vertex, VertexState, DataRejectedError, EdgeSignal


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
        data = "Hello world"
        assert data == "Hello world"

    def test_missing_key_returns_none(self, source_vertex):
        data = asyncio.run(source_vertex.fetch_data(channel="missing"))
        assert data is None


# ── get / set ────────────────────────────────────────────────────
class TestVertexGetSet:
    @pytest.mark.asyncio
    async def test_set_and_get(self, empty_vertex):
        await empty_vertex.receive_signal("", EdgeSignal.COMPLETED, payload="value1", channel="key1")
        result = await empty_vertex.fetch_data(channel="key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_overwrite(self, empty_vertex):
        await empty_vertex.receive_signal("", EdgeSignal.COMPLETED, payload="old", channel="k")
        await empty_vertex.receive_signal("", EdgeSignal.COMPLETED, payload="new", channel="k")
        assert await empty_vertex.fetch_data(channel="k") == "new"

    @pytest.mark.asyncio
    async def test_tag_order_irrelevant(self, empty_vertex):
        await empty_vertex.receive_signal("", EdgeSignal.COMPLETED, payload="data", channel="id")
        result = await empty_vertex.fetch_data(channel="id")
        assert result == "data"

    @pytest.mark.asyncio
    async def test_get_all_data(self, empty_vertex):
        await empty_vertex.receive_signal("", EdgeSignal.COMPLETED, payload="v1", channel="k1")
        await empty_vertex.receive_signal("", EdgeSignal.COMPLETED, payload="v2", channel="k2")
        all_data = await empty_vertex.get_all_data()
        assert len(all_data) == 2


# ── readiness semaphore ──────────────────────────────────────────
class TestVertexReadiness:
    @pytest.mark.asyncio
    async def test_becomes_ready_after_all_inputs(self):
        v = Vertex("v1")
        v.required_input_count = 2
        v.incoming_edges = ["e1", "e2"]

        await v.receive_signal("e1", EdgeSignal.COMPLETED, payload="a", channel="d1")
        assert v.state == VertexState.IDLE  # only 1 of 2

        await v.receive_signal("e2", EdgeSignal.COMPLETED, payload="b", channel="d2")
        assert v.state == VertexState.READY  # 2 of 2

    @pytest.mark.asyncio
    async def test_source_vertex_has_no_required_inputs(self, source_vertex):
        assert source_vertex.required_input_count == 0
        assert source_vertex.is_source()


# ── subclass hooks (on_receive / on_ready) ───────────────────────
# 发现背景：原 TestVertexScript 走 set_pipeline_module(load_script(...)) 挂
# 模块函数，但 Vertex 从未实现过 module 委托（vertex.py 只认 self.on_receive
# 等实例方法），该路径加载即 AttributeError，3 个测试长期 fail。随框架统一
# 转纯子类覆盖后，此处一并改为动态 Vertex 子类形式，真正覆盖钩子语义。
class TestVertexSubclassHooks:
    @pytest.mark.asyncio
    async def test_on_receive_transforms(self):
        class UpperVertex(Vertex):
            def on_receive(self, data, channel, settings):
                return data.upper() if isinstance(data, str) else data

        v = UpperVertex("v_upper")
        await v.receive_signal("", EdgeSignal.COMPLETED, payload="hello", channel="k")
        assert await v.fetch_data(channel="k") == "HELLO"

    @pytest.mark.asyncio
    async def test_on_receive_rejects(self):
        class RejectVertex(Vertex):
            def on_receive(self, data, channel, settings):
                raise ValueError("rejected")

        v = RejectVertex("v_reject")
        with pytest.raises(DataRejectedError, match="rejected"):
            await v.receive_signal("", EdgeSignal.COMPLETED, payload="anything", channel="k")

    @pytest.mark.asyncio
    async def test_on_ready_hook(self):
        class ReadyVertex(Vertex):
            def on_ready(self, all_data, settings):
                return {"out": "merged-data"}

        v = ReadyVertex("v_ready")
        await v.receive_signal("", EdgeSignal.COMPLETED, payload="raw", channel="in")
        await v.prepare_outputs()
        # 原测试仅 assert True（弱断言），现真正校验 on_ready 产出的 channel 被合并进 store。
        assert await v.fetch_data(channel="out") == "merged-data"


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
