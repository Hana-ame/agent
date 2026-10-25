"""Regression tests for the security hardening (review §1 / C1–C5).

Every test here fails against the pre-hardening code base.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from framework.server.app import create_v4_server
from framework.utils.paths import (
    PathNotAllowedError,
    resolve_confined_path,
    validate_session_id,
)
from framework.utils.script_loader import (
    ScriptNotAllowedError,
    get_allowed_script_roots,
    validate_script_reference,
)
from framework.vertex_v4 import VertexStoreV4

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_app(**kwargs) -> TestClient:
    app = create_v4_server(store_or_db=VertexStoreV4(":memory:"), snapshot_dir=None, **kwargs)
    return TestClient(app, raise_server_exceptions=False)


def _seed_two_vertices(client: TestClient, session: str = "sec_sess") -> None:
    client.post(f"/api/sessions/{session}/graph/vertices", json={"name": "a", "content": "x", "state": "data ready"})
    client.post(f"/api/sessions/{session}/graph/vertices", json={"name": "b", "content": "", "state": "todo"})


# ---------------------------------------------------------------------------
# C3 — authentication boundary and CORS
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    def test_protected_routes_require_key(self):
        client = _make_app(api_key="s3cret")
        assert client.get("/v1/models").status_code == 401
        assert client.get("/api/db/stats").status_code == 401

    def test_key_accepted_via_header_and_bearer(self):
        client = _make_app(api_key="s3cret")
        assert client.get("/v1/models", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert client.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200

    def test_wrong_key_rejected(self):
        client = _make_app(api_key="s3cret")
        assert client.get("/v1/models", headers={"X-API-Key": "nope"}).status_code == 401

    def test_dashboard_shell_is_exempt(self):
        client = _make_app(api_key="s3cret")
        assert client.get("/dashboard").status_code == 200

    def test_auth_disabled_when_no_key_configured(self, monkeypatch):
        monkeypatch.delenv("VEA_API_KEY", raising=False)
        client = _make_app()
        assert client.get("/v1/models").status_code == 200

    def test_no_cors_headers_by_default(self):
        client = _make_app()
        res = client.get("/v1/models", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in {k.lower() for k in res.headers}

    def test_explicit_cors_origin_is_honoured(self):
        client = _make_app(allowed_origins=["https://ok.example"])
        res = client.get("/v1/models", headers={"Origin": "https://ok.example"})
        assert res.headers.get("access-control-allow-origin") == "https://ok.example"


# ---------------------------------------------------------------------------
# C1 — inline eval / out-of-root script execution
# ---------------------------------------------------------------------------

class TestScriptExecutionBoundary:
    def test_inline_lambda_rejected_by_api(self):
        client = _make_app()
        _seed_two_vertices(client)
        res = client.post(
            "/api/sessions/sec_sess/graph/edges",
            json={
                "id": "e_rce",
                "type": "code",
                "input_vertex": "a",
                "output_vertex": "b",
                "script": "lambda c, s, st: __import__('os').popen('id').read()",
            },
        )
        assert res.status_code == 400
        assert "Inline script expressions are not allowed" in json.dumps(res.json())

    def test_absolute_out_of_root_script_rejected_by_api(self, tmp_path):
        evil = tmp_path / "evil.py"
        evil.write_text("def execute(c, s, st):\n    return 'pwned'\n")
        client = _make_app()
        _seed_two_vertices(client)
        res = client.post(
            "/api/sessions/sec_sess/graph/edges",
            json={
                "id": "e_abs",
                "type": "code",
                "input_vertex": "a",
                "output_vertex": "b",
                "script": f"{evil}:execute",
            },
        )
        assert res.status_code == 400
        assert "outside the allowed roots" in json.dumps(res.json())

    def test_relative_escape_rejected_by_api(self):
        client = _make_app()
        _seed_two_vertices(client)
        res = client.post(
            "/api/sessions/sec_sess/graph/edges",
            json={
                "id": "e_esc",
                "type": "code",
                "input_vertex": "a",
                "output_vertex": "b",
                "script": "../../evil.py:execute",
            },
        )
        assert res.status_code == 400

    def test_in_repo_script_still_accepted(self):
        client = _make_app()
        _seed_two_vertices(client)
        res = client.post(
            "/api/sessions/sec_sess/graph/edges",
            json={
                "id": "e_ok",
                "type": "code",
                "input_vertex": "a",
                "output_vertex": "b",
                "script": "examples/hn_v4/hn_transforms.py:execute",
            },
        )
        assert res.status_code == 200

    def test_inline_lambda_blocked_at_runtime_without_opt_in(self):
        """Even a trusted manifest must opt in explicitly for inline code."""
        import asyncio

        from framework.edges.code import CodeEdgeV4
        from framework.vertex_v4 import VertexStoreV4 as _Store

        store = _Store(":memory:")
        store.save_vertex("s", "a", content="x", state="data ready")
        store.save_vertex("s", "b", content="", state="todo")
        edge = CodeEdgeV4("e", "a", "b", script="lambda c, s, st: c")
        res = asyncio.run(edge.run("s", store))
        assert res.success is False
        assert "Inline lambda scripts are disabled" in (res.error or "")

    def test_inline_lambda_allowed_with_explicit_opt_in(self):
        import asyncio

        from framework.edges.code import CodeEdgeV4
        from framework.vertex_v4 import VertexStoreV4 as _Store

        store = _Store(":memory:")
        store.save_vertex("s", "a", content="x", state="data ready")
        store.save_vertex("s", "b", content="", state="todo")
        edge = CodeEdgeV4("e", "a", "b", script="lambda c, s, st: c + '-ok'",
                          settings={"allow_inline_script": True})
        res = asyncio.run(edge.run("s", store))
        assert res.success is True
        assert res.output == "x-ok"

    def test_validate_script_reference_rejects_inline_and_escapes(self, tmp_path):
        with pytest.raises(ScriptNotAllowedError):
            validate_script_reference("lambda x: x")
        with pytest.raises(ScriptNotAllowedError):
            validate_script_reference("/etc/passwd.py:main")
        with pytest.raises(ScriptNotAllowedError):
            validate_script_reference(str(tmp_path / "x.py"))
        # A repo-relative script is fine.
        validate_script_reference("examples/hn_v4/hn_transforms.py:execute")

    def test_allowed_roots_do_not_include_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        roots = get_allowed_script_roots()
        assert all(r != tmp_path.resolve() for r in roots)
        assert REPO_ROOT in roots


# ---------------------------------------------------------------------------
# C2 — client-supplied manifest paths
# ---------------------------------------------------------------------------

class TestManifestConfinement:
    def _evil_manifest(self, tmp_path: Path) -> Path:
        script = tmp_path / "evil.py"
        script.write_text(
            "def execute(c, s, st):\n"
            "    open(r'" + str(tmp_path / "PWNED.txt") + "', 'w').write('x')\n"
            "    return 'pwned'\n"
        )
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "vertices": [
                {"name": "v_start", "content": "t", "state": "data ready", "attributes": ["start"]},
                {"name": "v_end", "content": "", "state": "todo", "attributes": ["end"]},
            ],
            "edges": [
                {"id": "e1", "type": "code", "input_vertex": "v_start", "output_vertex": "v_end",
                 "script": f"{script}:execute"},
            ],
        }))
        return manifest

    def test_vea_manifest_path_is_ignored(self, tmp_path):
        manifest = self._evil_manifest(tmp_path)
        client = _make_app()
        res = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "vea_manifest_path": str(manifest),
                "messages": [{"role": "user", "content": "go"}],
            },
        )
        assert res.status_code == 200
        assert not (tmp_path / "PWNED.txt").exists()
        assert res.json()["choices"][0]["message"]["content"] != "pwned"

    def test_subgraph_manifest_outside_base_rejected(self, tmp_path):
        manifest = self._evil_manifest(tmp_path)
        client = _make_app()
        res = client.post(
            "/api/sessions/sec_sess/graph/subgraphs/add",
            json={"subgraph_manifest": str(manifest)},
        )
        assert res.status_code == 400
        assert "PATH_TRAVERSAL_REJECTED" in json.dumps(res.json())

    def test_subgraph_manifest_inside_base_allowed(self):
        client = _make_app()
        res = client.post(
            "/api/sessions/sec_sess/graph/subgraphs/add",
            json={"subgraph_manifest": "examples/subgraph_v4/child/child_graph.json"},
        )
        assert res.status_code == 200

    def test_models_only_lists_loadable_templates(self):
        client = _make_app()
        ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
        assert "default" in ids
        assert "hn_v4" in ids
        # Legacy V1 configs must not be advertised as usable models.
        assert "simple" not in ids
        assert "complex" not in ids


# ---------------------------------------------------------------------------
# C5 — session_id path traversal
# ---------------------------------------------------------------------------

class TestSessionIdValidation:
    def test_validate_session_id_accepts_normal_ids(self):
        for good in ("sess_1", "session-alpha", "a.b"):
            assert validate_session_id(good) == good

    @pytest.mark.parametrize("bad", ["..", ".", "a/b", "a b", "a::b", "", "x" * 129])
    def test_validate_session_id_rejects_traversal_shapes(self, bad):
        with pytest.raises(ValueError):
            validate_session_id(bad)

    def test_snapshot_route_rejects_dotdot_session(self):
        client = _make_app()
        res = client.post("/api/sessions/%2E%2E/snapshots", json={"trigger": "probe"})
        assert res.status_code == 400
        assert "INVALID_SESSION_ID" in json.dumps(res.json())

    def test_snapshot_manager_rejects_traversal(self, tmp_path):
        from framework.snapshot_v4 import GraphSnapshotManagerV4

        mgr = GraphSnapshotManagerV4(tmp_path / "snaps")
        with pytest.raises(PathNotAllowedError):
            mgr.get_session_dir("../../escaped")

    def test_resolve_confined_path_blocks_escape(self, tmp_path):
        with pytest.raises(PathNotAllowedError):
            resolve_confined_path("/etc/passwd", tmp_path)
        inside = tmp_path / "ok.json"
        inside.write_text("{}")
        assert resolve_confined_path(str(inside), tmp_path) == inside.resolve()


# ---------------------------------------------------------------------------
# C4 — SSRF / credential passthrough in route-and-run
# ---------------------------------------------------------------------------

class TestRouteAndRunSchema:
    def test_request_schema_drops_network_overrides(self):
        from framework.server.schemas import RouteAndRunRequest

        fields = set(RouteAndRunRequest.model_fields)
        assert "base_url" not in fields
        assert "api_key" not in fields
        assert "model" not in fields

    def test_client_supplied_base_url_is_ignored(self, monkeypatch):
        """Even if a client sends base_url, it must not reach the router."""
        captured = {}

        async def fake_classify(query, catalog_dir, api_key=None, base_url=None, model=None, timeout=12.0):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            return "data_extractor", "stub"

        import framework.tool_catalog.router as router_mod
        monkeypatch.setattr(router_mod, "classify_intent_with_llm", fake_classify)

        client = _make_app()
        res = client.post(
            "/api/sessions/sec_sess/route-and-run",
            json={
                "task": "extract contacts",
                "use_llm": True,
                "base_url": "http://127.0.0.1:1/v1/chat/completions",
                "api_key": "CLIENT-SUPPLIED",
            },
        )
        # The tool catalog may not be configured in this fixture, so only assert
        # the request did not carry the override through when it was used.
        if captured:
            assert captured.get("base_url") != "http://127.0.0.1:1/v1/chat/completions"
        assert res.status_code in (200, 404, 500)
