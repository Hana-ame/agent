"""Regression tests for the edge-type registry and serialization fixes.

Covers review findings H3 (llm_tool serialize/restore + tools loss), H4
(settings divergence) and H5 (mutation API rejected most edge types).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from framework.chat_llm_edge_v4 import ChatLLMEdgeV4
from framework.callable_llm_edge_v4 import CallableLLMEdgeV4
from framework.edges.base import EdgeV4
from framework.edges.code import CodeEdgeV4
from framework.edges.llm import LLMEdgeV4
from framework.edges.reflexive import ReflexiveEdgeV4
from framework.edges.registry import (
    EDGE_REGISTRY,
    edge_type_choices,
    get_edge_class,
    is_registered_edge_type,
    is_tool_edge,
    register_edge_type,
)
from framework.edges.tool import LLMToolEdgeV4, ToolEdgeV4
from framework.generate_llm_edge_v4 import GenerateLLMEdgeV4
from framework.graph_v4 import DiscreteGraphLoaderV4, GraphV4
from framework.process_llm_edge_v4 import ProcessLLMEdgeV4
from framework.server.app import create_v4_server
from framework.vertex_v4 import VertexStoreV4

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "run"}}]


def _all_edges() -> dict:
    return {
        "code": CodeEdgeV4(edge_id="e", input_vertex="a", output_vertex="b"),
        "tool": ToolEdgeV4(edge_id="e", input_vertex="a", output_vertex="b",
                           tool_name="bash", arguments={"command": "ls"}),
        "llm": LLMEdgeV4(edge_id="e", input_vertex="a", output_vertex="b"),
        "llm_chat": ChatLLMEdgeV4(edge_id="e", input_vertex="a", output_vertex="b"),
        "llm_generate": GenerateLLMEdgeV4(edge_id="e", input_vertex="a", output_vertex="b"),
        "llm_process": ProcessLLMEdgeV4(edge_id="e", input_vertex="a", output_vertex="b"),
        "llm_callable": CallableLLMEdgeV4(edge_id="e", input_vertex="a", output_vertex="b"),
        "llm_tool": LLMToolEdgeV4(edge_id="e", input_vertex="a", output_vertex="b", tools=TOOLS),
        "reflexive": ReflexiveEdgeV4(edge_id="e", vertex_name="a"),
    }


class TestRegistry:
    def test_every_edge_type_is_registered(self):
        for name, edge in _all_edges().items():
            assert is_registered_edge_type(name), f"{name} not registered"
            assert get_edge_class(name) is type(edge), f"{name} maps to the wrong class"

    def test_registry_has_no_hardcoded_dispatch_left(self):
        # The registry is the only source of truth for the type -> class mapping.
        assert set(_all_edges()) <= set(EDGE_REGISTRY)
        assert "code" in edge_type_choices()
        assert "llm_tool" in edge_type_choices()

    def test_capability_flags(self):
        assert is_tool_edge(ToolEdgeV4(edge_id="e", input_vertex="a", output_vertex="b"))
        assert is_tool_edge(LLMToolEdgeV4(edge_id="e", input_vertex="a", output_vertex="b"))
        assert not is_tool_edge(CodeEdgeV4(edge_id="e", input_vertex="a", output_vertex="b"))
        assert ReflexiveEdgeV4.IS_RECOVERY is True
        assert ToolEdgeV4.EMITS_TOOL_CALL is True

    def test_registering_same_class_twice_is_idempotent(self):
        @register_edge_type("dup_type")
        class _Dup(CodeEdgeV4):
            pass

        register_edge_type("dup_type")(_Dup)  # same class again: allowed
        with pytest.raises(ValueError):
            class _Other(CodeEdgeV4):
                pass

            register_edge_type("dup_type")(_Other)
        EDGE_REGISTRY.pop("dup_type", None)
        edge_type_choices().remove("dup_type") if "dup_type" in edge_type_choices() else None

    def test_unknown_type_raises_helpful_error(self):
        with pytest.raises(ValueError) as exc:
            EdgeV4.from_config({"id": "e", "type": "nope", "input_vertex": "a", "output_vertex": "b"})
        assert "registered type" in str(exc.value)


class TestRoundTrip:
    @pytest.mark.parametrize("name", list(_all_edges()))
    def test_to_dict_from_config_round_trip(self, name):
        edge = _all_edges()[name]
        rebuilt = EdgeV4.from_config(edge.to_dict())
        assert type(rebuilt) is type(edge)

    def test_llm_tool_tools_survive_serialization(self):
        edge = LLMToolEdgeV4(edge_id="e", input_vertex="a", output_vertex="b", tools=TOOLS)
        assert edge.settings["tools"] == TOOLS
        data = edge.to_dict()
        assert data["settings"]["tools"] == TOOLS
        rebuilt = EdgeV4.from_config(data)
        assert rebuilt.tools == TOOLS
        assert rebuilt.type == "llm_tool"

    def test_tool_edge_arguments_survive_serialization(self):
        edge = ToolEdgeV4(edge_id="e", input_vertex="a", output_vertex="b",
                          tool_name="bash", arguments={"command": "ls"})
        rebuilt = EdgeV4.from_config(edge.to_dict())
        assert rebuilt.tool_name == "bash"
        assert rebuilt.arguments == {"command": "ls"}

    def test_llm_tool_accepts_concurrency_kwargs(self):
        edge = LLMToolEdgeV4(
            edge_id="e", input_vertex="a", output_vertex="b",
            concurrency_limit=2, concurrency_group="g", timeout=5.0,
        )
        assert edge.concurrency_limit == 2
        assert edge.concurrency_group == "g"
        assert edge.timeout == 5.0

    def test_graph_dump_load_preserves_llm_tool(self):
        graph = GraphV4(session_id="s1")
        graph.add_vertex({"name": "a", "content": "x", "state": "data ready"})
        graph.add_vertex({"name": "b", "content": "", "state": "todo"})
        graph.add_edge(LLMToolEdgeV4(edge_id="e_lt", input_vertex="a", output_vertex="b", tools=TOOLS))
        reloaded = DiscreteGraphLoaderV4.load_from_dict(graph.dump(), override_session_id="s2")
        edge = reloaded.edges["e_lt"]
        assert edge.type == "llm_tool"
        assert edge.tools == TOOLS


class TestMutationApi:
    def _client(self) -> TestClient:
        app = create_v4_server(store_or_db=VertexStoreV4(":memory:"), snapshot_dir=None)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/api/sessions/s/graph/vertices", json={"name": "a", "content": "x", "state": "data ready"})
        client.post("/api/sessions/s/graph/vertices", json={"name": "b", "content": "", "state": "todo"})
        return client

    @pytest.mark.parametrize("edge_type", ["code", "llm", "llm_chat", "llm_generate",
                                           "llm_process", "llm_callable", "tool", "llm_tool", "reflexive"])
    def test_all_registered_types_accepted_by_api(self, edge_type):
        client = self._client()
        res = client.post(
            "/api/sessions/s/graph/edges",
            json={
                "id": f"e_{edge_type}",
                "type": edge_type,
                "input_vertex": "a",
                "output_vertex": "b",
                "settings": {"tool_name": "bash", "tools": TOOLS},
            },
        )
        assert res.status_code == 200, res.text

    def test_settings_reach_the_live_edge_not_just_sqlite(self):
        """H4: live edge settings must match what was persisted."""
        app = create_v4_server(store_or_db=VertexStoreV4(":memory:"), snapshot_dir=None)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/api/sessions/s/graph/vertices", json={"name": "a", "content": "x", "state": "data ready"})
        client.post("/api/sessions/s/graph/vertices", json={"name": "b", "content": "", "state": "todo"})
        res = client.post(
            "/api/sessions/s/graph/edges",
            json={
                "id": "e1",
                "type": "llm",
                "input_vertex": "a",
                "output_vertex": "b",
                "settings": {"temperature": 0.1, "merge_strategy": "json_merge",
                             "model": "m", "prompt": "P"},
            },
        )
        assert res.status_code == 200
        graph = app.state.manager.get_or_create_graph("s")
        live = graph.edges["e1"]
        persisted = {e.edge_id: e for e in app.state.store.list_edges("s")}["e1"]
        for key, value in (("temperature", 0.1), ("merge_strategy", "json_merge"),
                           ("model", "m"), ("prompt", "P")):
            assert live.settings.get(key) == value, f"{key} missing from live edge"
            assert persisted.settings.get(key) == value

    def test_llm_tool_added_via_api_round_trips_through_store(self):
        app = create_v4_server(store_or_db=VertexStoreV4(":memory:"), snapshot_dir=None)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/api/sessions/s/graph/vertices", json={"name": "a", "content": "x", "state": "data ready"})
        client.post("/api/sessions/s/graph/vertices", json={"name": "b", "content": "", "state": "todo"})
        res = client.post(
            "/api/sessions/s/graph/edges",
            json={
                "id": "e_lt",
                "type": "llm_tool",
                "input_vertex": "a",
                "output_vertex": "b",
                "settings": {"tools": TOOLS},
            },
        )
        assert res.status_code == 200
        reloaded = app.state.manager.load_graph_from_store("s")
        assert reloaded.edges["e_lt"].type == "llm_tool"
        assert reloaded.edges["e_lt"].tools == TOOLS
