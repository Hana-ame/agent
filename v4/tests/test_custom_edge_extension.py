"""Regression tests for zero-registration custom edges.

The intended extension workflow is:

1. write ``class MyEdge(EdgeV4)`` in ``my_edge.py``
2. reference it from graph/edge JSON as ``"type": "my_edge.py:MyEdge"``
3. run it — no framework edit, no registration call

These tests pin that contract, plus the persistence/API paths around it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from framework.edges.base import EdgeV4, is_dynamic_edge_type
from framework.edges.code import CodeEdgeV4
from framework.executor_v4 import ExecutorV4
from framework.graph_manager_v4 import SessionGraphManagerV4
from framework.graph_v4 import DiscreteGraphLoaderV4, GraphV4
from framework.server.app import create_v4_server
from framework.vertex_v4 import VertexStateV4, VertexStoreV4

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "examples" / "custom_edge"
EXAMPLE_MANIFEST = EXAMPLE_DIR / "config.json"
DYNAMIC_SPEC = "my_edges.py:WordCountEdge"


class TestResolution:
    def test_dynamic_type_detection(self):
        assert is_dynamic_edge_type("my_edge.py:MyEdge")
        assert is_dynamic_edge_type("edges/custom.py")
        assert not is_dynamic_edge_type("code")
        assert not is_dynamic_edge_type(None)

    def test_manifest_loads_custom_classes(self):
        graph = DiscreteGraphLoaderV4.load_from_manifest(
            str(EXAMPLE_MANIFEST), override_session_id="custom"
        )
        assert type(graph.edges["e_count"]).__name__ == "WordCountEdge"
        assert type(graph.edges["e_upper"]).__name__ == "UpperCaseEdge"
        # The declared spec is preserved on the instance.
        assert graph.edges["e_count"].type == DYNAMIC_SPEC

    def test_explicit_class_from_config(self):
        from examples.custom_edge.my_edges import WordCountEdge

        edge = WordCountEdge.from_config(
            {"id": "e", "type": DYNAMIC_SPEC, "input_vertex": "a", "output_vertex": "b"},
            base_dir=str(EXAMPLE_DIR),
        )
        assert type(edge).__name__ == "WordCountEdge"

    def test_unknown_class_name_reports_clearly(self):
        with pytest.raises(ValueError) as exc:
            EdgeV4.from_config(
                {
                    "id": "e",
                    "type": "examples/custom_edge/my_edges.py:DoesNotExist",
                    "input_vertex": "a",
                    "output_vertex": "b",
                }
            )
        assert "does not define an EdgeV4 subclass" in str(exc.value)


class TestExecution:
    @pytest.mark.asyncio
    async def test_custom_edges_execute_end_to_end(self):
        store = VertexStoreV4(":memory:")
        graph = DiscreteGraphLoaderV4.load_from_manifest(
            str(EXAMPLE_MANIFEST), override_session_id="custom_run"
        )
        DiscreteGraphLoaderV4.populate_store(graph, store)

        executor = ExecutorV4(graph=graph, store=store, max_concurrency=2, timeout=10.0)
        res = await executor.run()

        assert res.success is True, res.errors
        assert set(res.completed_edges) == {"e_count", "e_upper", "e_passthrough"}
        assert res.vertex_contents["v_count"] == "5 words"
        assert res.vertex_contents["v_out"] == "5 WORDS"

    @pytest.mark.asyncio
    async def test_custom_constructor_kwargs_are_filtered(self):
        """A custom __init__ that accepts only core kwargs still works."""
        from examples.custom_edge.my_edges import WordCountEdge

        edge = EdgeV4.from_config(
            {
                "id": "e",
                "type": DYNAMIC_SPEC,
                "input_vertex": "a",
                "output_vertex": "b",
                "settings": {"label": "tokens"},
                # Constructor does not accept these; they must be dropped.
                "concurrency_limit": 3,
                "timeout": 5.0,
                "priority": 7,
            },
            base_dir=str(EXAMPLE_DIR),
        )
        # A script loaded by path is a distinct module object from a normal
        # import, so compare by class name rather than identity.
        assert type(edge).__name__ == "WordCountEdge"
        assert edge.settings["label"] == "tokens"


class TestPersistence:
    def test_to_dict_round_trip_preserves_class(self):
        graph = DiscreteGraphLoaderV4.load_from_manifest(
            str(EXAMPLE_MANIFEST), override_session_id="rt"
        )
        data = graph.edges["e_count"].to_dict()
        assert data["type"] == DYNAMIC_SPEC
        rebuilt = EdgeV4.from_config(data)
        assert type(rebuilt).__name__ == "WordCountEdge"

    def test_store_round_trip_preserves_class(self):
        store = VertexStoreV4(":memory:")
        graph = DiscreteGraphLoaderV4.load_from_manifest(
            str(EXAMPLE_MANIFEST), override_session_id="store_rt"
        )
        DiscreteGraphLoaderV4.populate_store(graph, store)
        reloaded = GraphV4.load_from_store(store, "store_rt")
        assert type(reloaded.edges["e_count"]).__name__ == "WordCountEdge"
        assert type(reloaded.edges["e_upper"]).__name__ == "UpperCaseEdge"

    def test_snapshot_dump_round_trip_preserves_class(self):
        graph = DiscreteGraphLoaderV4.load_from_manifest(
            str(EXAMPLE_MANIFEST), override_session_id="snap_rt"
        )
        rebuilt = DiscreteGraphLoaderV4.load_from_dict(graph.dump(), override_session_id="snap_rt2")
        assert type(rebuilt.edges["e_count"]).__name__ == "WordCountEdge"


class TestApi:
    def _client(self) -> TestClient:
        app = create_v4_server(store_or_db=VertexStoreV4(":memory:"), snapshot_dir=None)
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/api/sessions/s/graph/vertices", json={"name": "a", "content": "one two three", "state": "data ready"})
        client.post("/api/sessions/s/graph/vertices", json={"name": "b", "content": "", "state": "todo"})
        return client

    def test_api_accepts_in_repo_dynamic_type(self):
        client = self._client()
        res = client.post(
            "/api/sessions/s/graph/edges",
            json={
                "id": "e_custom",
                "type": "examples/custom_edge/my_edges.py:WordCountEdge",
                "input_vertex": "a",
                "output_vertex": "b",
                "settings": {"label": "words"},
            },
        )
        assert res.status_code == 200, res.text
        run = client.post("/api/sessions/s/run", json={})
        assert run.json()["success"] is True
        assert run.json()["vertex_contents"]["b"] == "3 words"

    def test_api_rejects_out_of_root_dynamic_type(self):
        client = self._client()
        res = client.post(
            "/api/sessions/s/graph/edges",
            json={
                "id": "e_evil",
                "type": "/tmp/evil.py:EvilEdge",
                "input_vertex": "a",
                "output_vertex": "b",
            },
        )
        assert res.status_code == 400
        assert "outside the allowed roots" in json.dumps(res.json())

    def test_api_rejects_unknown_type_with_guidance(self):
        client = self._client()
        res = client.post(
            "/api/sessions/s/graph/edges",
            json={"id": "e_x", "type": "nope", "input_vertex": "a", "output_vertex": "b"},
        )
        assert res.status_code == 400
        assert "my_edge.py:MyEdge" in json.dumps(res.json())

    def test_builtin_types_still_work(self):
        client = self._client()
        res = client.post(
            "/api/sessions/s/graph/edges",
            json={"id": "e_code", "type": "code", "input_vertex": "a", "output_vertex": "b"},
        )
        assert res.status_code == 200
        app = client.app
        graph = app.state.manager.get_or_create_graph("s")
        assert isinstance(graph.edges["e_code"], CodeEdgeV4)
        assert graph.edges["e_code"].type == "code"
