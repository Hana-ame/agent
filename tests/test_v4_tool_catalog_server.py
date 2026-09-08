"""Automated test suite for V4 Dynamic Tool Catalog and Router Server Endpoints."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from framework.server_v4 import create_v4_server
from framework.vertex_v4 import VertexStoreV4


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Fixture providing a TestClient with a persistent test SQLite DB and snapshot dir."""
    db_file = tmp_path / "test_tool_server_v4.db"
    snapshot_dir = tmp_path / "snapshots"
    store = VertexStoreV4(str(db_file))
    app = create_v4_server(
        store_or_db=store,
        snapshot_dir=snapshot_dir,
    )
    test_client = TestClient(app)
    yield test_client
    store.close()


def test_tool_catalog_listing(client: TestClient):
    """Test GET /api/tool-catalog returns all registered tools."""
    res = client.get("/api/tool-catalog")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 3
    tool_ids = [t["id"] for t in data["tools"]]
    assert "code_analyzer" in tool_ids
    assert "finance_calculator" in tool_ids
    assert "data_extractor" in tool_ids

    # Verify tool metadata fields
    for t in data["tools"]:
        assert "name" in t
        assert "description" in t
        assert "entry_vertex" in t
        assert "exit_vertex" in t
        assert "vertex_count" in t
        assert "edge_count" in t


def test_tool_catalog_get_manifest(client: TestClient):
    """Test GET /api/tool-catalog/{tool_id} fetches specific manifest."""
    res = client.get("/api/tool-catalog/code_analyzer")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "code_analyzer"
    manifest = data["manifest"]
    assert "vertices" in manifest
    assert "edges" in manifest
    assert len(manifest["vertices"]) == 3
    assert len(manifest["edges"]) == 2

    # Test non-existent tool returns 404
    err_res = client.get("/api/tool-catalog/non_existent_tool_xyz")
    assert err_res.status_code == 404


def test_route_and_run_explicit_tool(client: TestClient):
    """Test POST /api/sessions/{session_id}/route-and-run with explicit tool override."""
    session_id = "sess_test_explicit"
    payload = {
        "task": "def calculate_bonus(salary): return eval('salary * 0.1')",
        "tool_id": "code_analyzer",
    }
    res = client.post(f"/api/sessions/{session_id}/route-and-run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == session_id
    assert data["routed_tool"] == "code_analyzer"
    assert data["success"] is True
    assert "Automated Code Review Report" in data["output"]
    assert "eval()" in data["output"]
    assert data["snapshot_count"] > 0
    assert len(data["inserted_vertices"]) == 3
    assert len(data["inserted_edges"]) >= 2


def test_route_and_run_heuristic_intent_routing(client: TestClient):
    """Test POST /api/sessions/{session_id}/route-and-run with heuristic classifier (use_llm=False)."""
    # 1. Code analyzer routing
    res_code = client.post(
        "/api/sessions/sess_code/route-and-run",
        json={
            "task": "def process_data(items): return [x.upper() for x in items]",
            "use_llm": False,
        },
    )
    assert res_code.status_code == 200
    assert res_code.json()["routed_tool"] == "code_analyzer"
    assert "Automated Code Review Report" in res_code.json()["output"]

    # 2. Finance calculator routing
    res_fin = client.post(
        "/api/sessions/sess_fin/route-and-run",
        json={
            "task": "Our quarterly revenue reached 4000000 with cost of 2500000.",
            "use_llm": False,
        },
    )
    assert res_fin.status_code == 200
    assert res_fin.json()["routed_tool"] == "finance_calculator"
    assert "Executive Financial Analysis" in res_fin.json()["output"]
    assert "$4,000,000.00" in res_fin.json()["output"]

    # 3. Data extractor routing
    res_data = client.post(
        "/api/sessions/sess_data/route-and-run",
        json={
            "task": "Please contact test_user@company.com or sales@enterprise.org for licenses.",
            "use_llm": False,
        },
    )
    assert res_data.status_code == 200
    assert res_data.json()["routed_tool"] == "data_extractor"
    assert "test_user@company.com" in res_data.json()["output"]


def test_openai_chat_completions_with_dynamic_route(client: TestClient):
    """Test /v1/chat/completions with vea_dynamic_route=True automatically selects tool and executes."""
    req = {
        "model": "dynamic-router",
        "messages": [
            {
                "role": "user",
                "content": "Company announced revenue of 10000000 and expenses of 7000000 for FY2026.",
            }
        ],
        "vea_dynamic_route": True,
    }
    res = client.post("/v1/chat/completions", json=req)
    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) == 1
    content = body["choices"][0]["message"]["content"]
    assert "Executive Financial Analysis" in content
    assert "$10,000,000.00" in content
