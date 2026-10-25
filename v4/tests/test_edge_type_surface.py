"""Regression tests for the client-facing edge-type surface.

The Python side stopped hardcoding edge types in the registry refactor, but the
dashboard and the server factory still had gaps:

* ``dashboard.html`` offered a fixed ``<select>`` with four options, so
  ``llm_tool`` and custom script-spec edges could not be created — and editing
  such an edge reset its type to ``code`` because ``select.value`` silently
  fails for values that are not among the options.
* ``create_v4_server(script_roots=...)`` stored ``app.state.script_roots`` but
  never passed it to the graph manager, so the parameter did nothing and
  confinement always fell back to the repository root plus ``VEA_SCRIPT_ROOTS``.

These tests pin the fixed behaviour.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from framework.edges.base import EdgeV4
from framework.edges.registry import EDGE_REGISTRY, register_edge_type
from framework.graph_manager_v4 import SessionGraphManagerV4
from framework.server.app import create_v4_server, parse_server_args
from framework.vertex_v4 import VertexStoreV4

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "framework" / "templates"

EDGE_SCRIPT = """
from framework.edges.base import EdgeV4


class SurfaceProbeEdge(EdgeV4):
    async def run(self, session_id, store, agent=None, **kwargs):
        return None
"""


def _client(**kwargs) -> TestClient:
    app = create_v4_server(store_or_db=VertexStoreV4(":memory:"), snapshot_dir=None, **kwargs)
    return TestClient(app, raise_server_exceptions=False)


class TestEdgeTypeEndpoint:
    def test_lists_registered_types(self):
        res = _client().get("/api/edge-types")
        assert res.status_code == 200
        body = res.json()
        types = body["types"]
        assert "code" in types
        assert "llm_tool" in types
        assert body["script_spec_supported"] is True
        assert ":" in body["script_spec_example"]

    def test_reports_script_roots_configuration(self, tmp_path):
        assert _client().get("/api/edge-types").json()["script_roots_configured"] is False
        res = _client(script_roots=[tmp_path]).get("/api/edge-types")
        assert res.json()["script_roots_configured"] is True

    def test_custom_registered_type_is_advertised(self):
        class SurfaceAdvertisedEdge(EdgeV4):
            pass

        register_edge_type("surface_advertised", replace=True)(SurfaceAdvertisedEdge)
        try:
            types = _client().get("/api/edge-types").json()["types"]
            assert "surface_advertised" in types
        finally:
            EDGE_REGISTRY.pop("surface_advertised", None)


class TestScriptRootWiring:
    def test_manager_receives_script_roots(self, tmp_path):
        app = create_v4_server(
            store_or_db=VertexStoreV4(":memory:"), snapshot_dir=None, script_roots=[tmp_path]
        )
        assert app.state.manager.script_roots == [Path(tmp_path).resolve()]

    def test_manager_defaults_to_none(self):
        app = create_v4_server(store_or_db=VertexStoreV4(":memory:"), snapshot_dir=None)
        assert app.state.manager.script_roots is None

    def test_spec_outside_configured_root_is_rejected(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "probe.py").write_text(EDGE_SCRIPT, encoding="utf-8")

        client = _client(script_roots=[allowed])
        res = client.post(
            "/api/sessions/s/graph/edges",
            json={
                "id": "e1",
                "type": f"{outside / 'probe.py'}:SurfaceProbeEdge",
                "input_vertex": "a",
                "output_vertex": "b",
            },
        )
        assert res.status_code == 400
        assert "outside the allowed roots" in str(res.json()["detail"])

    def test_spec_inside_configured_root_is_accepted(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        (allowed / "probe.py").write_text(EDGE_SCRIPT, encoding="utf-8")

        client = _client(script_roots=[allowed])
        res = client.post(
            "/api/sessions/s/graph/edges",
            json={
                "id": "e1",
                "type": "probe.py:SurfaceProbeEdge",
                "input_vertex": "a",
                "output_vertex": "b",
            },
        )
        assert res.status_code == 200, res.text
        assert res.json()["edge"]["type"] == "probe.py:SurfaceProbeEdge"

    def test_manager_constructor_accepts_script_roots(self, tmp_path):
        manager = SessionGraphManagerV4(
            store=VertexStoreV4(":memory:"), snapshot_dir=None, script_roots=[tmp_path]
        )
        assert manager.script_roots == [Path(tmp_path).resolve()]

    def test_dynamic_edge_survives_store_rehydrate(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        (allowed / "probe.py").write_text(EDGE_SCRIPT, encoding="utf-8")

        client = _client(script_roots=[allowed])
        created = client.post(
            "/api/sessions/s/graph/edges",
            json={
                "id": "e1",
                "type": "probe.py:SurfaceProbeEdge",
                "input_vertex": "a",
                "output_vertex": "b",
            },
        )
        assert created.status_code == 200, created.text

        # Reading the graph rehydrates every edge from SQLite through
        # EdgeV4.from_config, which must resolve the relative spec again.
        graph = client.get("/api/sessions/s/graph")
        assert graph.status_code == 200, graph.text
        edges = graph.json()["edges"]
        assert edges["e1"]["type"] == "probe.py:SurfaceProbeEdge"

        manager = client.app.state.manager
        assert type(manager._graphs["s"].edges["e1"]).__name__ == "SurfaceProbeEdge"

    def test_graph_remembers_script_roots(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        (allowed / "probe.py").write_text(EDGE_SCRIPT, encoding="utf-8")
        client = _client(script_roots=[allowed])
        client.post(
            "/api/sessions/s/graph/edges",
            json={
                "id": "e1",
                "type": "probe.py:SurfaceProbeEdge",
                "input_vertex": "a",
                "output_vertex": "b",
            },
        )
        client.get("/api/sessions/s/graph")
        assert client.app.state.manager._graphs["s"].script_roots == [allowed.resolve()]


class TestCliScriptRoot:
    def test_script_root_flag_is_repeatable(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["vea", "--script-root", "edges_a", "--script-root", "edges_b"],
        )
        args = parse_server_args()
        assert args.script_roots == ["edges_a", "edges_b"]

    def test_script_root_defaults_to_none(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["vea"])
        assert parse_server_args().script_roots is None


class TestDashboardTemplate:
    def test_no_hardcoded_select(self):
        html = (TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")
        assert '<select id="eType"' not in html
        assert 'id="eType"' in html
        assert 'list="edgeTypeOptions"' in html
        assert 'id="edgeTypeOptions"' in html

    def test_js_loads_types_from_api(self):
        js = (TEMPLATE_DIR / "dashboard.js").read_text(encoding="utf-8")
        assert "/api/edge-types" in js
        assert "loadEdgeTypes" in js
        # The loader must run before the graph editor is usable.
        assert "await loadEdgeTypes();" in js


class TestDashboardVertexStates:
    def test_state_options_match_enum(self):
        from framework.vertex_v4 import VertexStateV4

        html = (TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")
        block = html.split('id="vState"', 1)[1].split("</select>", 1)[0]
        options = set(re.findall(r'<option value="([^"]+)"', block))
        assert options == {s.value for s in VertexStateV4}

    def test_js_never_drops_an_unlisted_state(self):
        js = (TEMPLATE_DIR / "dashboard.js").read_text(encoding="utf-8")
        # Assigning an unknown value to a <select> empties it; the helper appends
        # the missing option instead so a save cannot silently change the state.
        assert "function setSelectValue" in js
        assert 'setSelectValue("vState"' in js
        assert 'document.getElementById("vState").value =' not in js
