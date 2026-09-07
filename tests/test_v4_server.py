"""Automated test suite for V4 standalone server and online graph API.

Verifies:
1. Online graph manipulation (adding/updating/deleting vertices and edges).
2. Multi-session isolation (independent graphs per session).
3. Live workflow execution via API.
4. Database inspection endpoints (stats, vertices, session_staging).
5. Interactive dashboard HTML serving.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from framework.server_v4 import create_v4_server
from framework.vertex_v4 import VertexAttributeV4, VertexStateV4, VertexStoreV4


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Fixture providing a TestClient with a persistent test SQLite DB."""
    db_file = tmp_path / "test_server_v4.db"
    store = VertexStoreV4(str(db_file))
    app = create_v4_server(store_or_db=store)
    test_client = TestClient(app)
    yield test_client
    store.close()


def test_dashboard_endpoint(client: TestClient):
    """Test dashboard HTML is served properly."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "VEA v4 Dynamic Graph & Database Viewer" in res.text

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "Live Vertices (Click row to inspect/edit)" in res_dash.text


def test_online_graph_manipulation_and_validation(client: TestClient):
    """Test dynamically adding vertices and edges, validating DAG, and tier recalculation."""
    session_id = "online_graph_sess"

    # 1. Add vertices online
    res_v1 = client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={
            "name": "node_input",
            "content": "raw telemetry data",
            "attributes": ["start", "plain text"],
            "state": "data ready"
        }
    )
    assert res_v1.status_code == 200
    assert res_v1.json()["status"] == "saved"
    assert res_v1.json()["vertex"]["name"] == "node_input"

    res_v2 = client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={
            "name": "node_output",
            "content": "",
            "attributes": ["end"],
            "state": "todo"
        }
    )
    assert res_v2.status_code == 200

    # 2. Add forward edge online
    res_e1 = client.post(
        f"/api/sessions/{session_id}/graph/edges",
        json={
            "id": "e_forward_1",
            "type": "code",
            "input_vertex": "node_input",
            "output_vertex": "node_output",
            "settings": {}
        }
    )
    assert res_e1.status_code == 200
    data_e1 = res_e1.json()
    assert data_e1["status"] == "saved"
    assert data_e1["validation"]["valid"] is True
    assert data_e1["validation"]["tiers"]["e_forward_1"] == 0

    # 3. Retrieve graph structure
    res_graph = client.get(f"/api/sessions/{session_id}/graph")
    assert res_graph.status_code == 200
    graph_info = res_graph.json()
    assert graph_info["valid"] is True
    assert "node_input" in graph_info["vertices"]
    assert "node_output" in graph_info["vertices"]
    assert "e_forward_1" in graph_info["edges"]

    # 4. Delete an edge
    res_del_e = client.delete(f"/api/sessions/{session_id}/graph/edges/e_forward_1")
    assert res_del_e.status_code == 200
    assert res_del_e.json()["deleted"] is True

    # 5. Delete a vertex
    res_del_v = client.delete(f"/api/sessions/{session_id}/graph/vertices/node_output")
    assert res_del_v.status_code == 200
    assert res_del_v.json()["deleted"] is True


def test_session_isolation(client: TestClient):
    """Test that Session A and Session B maintain isolated graphs and DB records."""
    sess_a = "session_alpha"
    sess_b = "session_beta"

    # Add vertex to session A
    client.post(
        f"/api/sessions/{sess_a}/graph/vertices",
        json={"name": "alpha_node", "state": "data ready", "content": "alpha content"}
    )

    # Add vertex to session B
    client.post(
        f"/api/sessions/{sess_b}/graph/vertices",
        json={"name": "beta_node", "state": "todo", "content": "beta content"}
    )

    # Verify session A vertices
    res_a = client.get(f"/api/db/sessions/{sess_a}/vertices")
    assert res_a.status_code == 200
    names_a = [v["name"] for v in res_a.json()]
    assert "alpha_node" in names_a
    assert "beta_node" not in names_a

    # Verify session B vertices
    res_b = client.get(f"/api/db/sessions/{sess_b}/vertices")
    assert res_b.status_code == 200
    names_b = [v["name"] for v in res_b.json()]
    assert "beta_node" in names_b
    assert "alpha_node" not in names_b

    # Verify distinct sessions listed
    res_sessions = client.get("/api/db/sessions")
    assert res_sessions.status_code == 200
    sess_list = res_sessions.json()
    assert sess_a in sess_list
    assert sess_b in sess_list


def test_db_inspector_endpoints(client: TestClient):
    """Test viewing SQLite tables (vertices, session_staging, stats)."""
    session_id = "inspector_sess"

    # Seed vertex
    client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={"name": "v_test", "content": "inspect payload", "state": "data ready"}
    )

    # Stage an entry directly in store
    store: VertexStoreV4 = client.app.state.store
    store.stage_output(
        session_id=session_id,
        edge_id="e_debug_1",
        key="test_key",
        value="test_staged_value",
        vertex_name="v_test",
        metadata={"diag": "ok"}
    )

    # Test stats
    res_stats = client.get("/api/db/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert stats["total_vertices"] >= 1
    assert stats["total_staging"] >= 1

    # Test query staging
    res_staging = client.get(f"/api/db/sessions/{session_id}/staging")
    assert res_staging.status_code == 200
    staged_items = res_staging.json()
    assert len(staged_items) == 1
    assert staged_items[0]["key"] == "test_key"
    assert staged_items[0]["edge_id"] == "e_debug_1"

    # Clear session
    res_clear = client.post(f"/api/db/sessions/{session_id}/clear")
    assert res_clear.status_code == 200
    assert res_clear.json()["status"] == "cleared"

    res_v_after = client.get(f"/api/db/sessions/{session_id}/vertices")
    assert len(res_v_after.json()) == 0


def test_online_run_session_workflow(client: TestClient):
    """Test executing a pipeline through the online API."""
    session_id = "workflow_exec_sess"

    # Build 2-node graph online
    client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={"name": "v_in", "content": "dynamic execution", "attributes": ["start"], "state": "data ready"}
    )
    client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={"name": "v_out", "content": "", "attributes": ["end"], "state": "todo"}
    )
    client.post(
        f"/api/sessions/{session_id}/graph/edges",
        json={
            "id": "e_step",
            "type": "code",
            "input_vertex": "v_in",
            "output_vertex": "v_out",
            "settings": {}
        }
    )

    # Execute workflow online
    res_run = client.post(
        f"/api/sessions/{session_id}/run",
        json={"max_concurrency": 2, "timeout": 30.0}
    )
    assert res_run.status_code == 200
    result = res_run.json()
    assert result["success"] is True
    assert "e_step" in result["completed_edges"]
    assert result["vertex_states"]["v_out"] == "data ready"
    assert result["vertex_contents"]["v_out"] == "dynamic execution"

    # Check updated database content via inspector API
    res_v = client.get(f"/api/db/sessions/{session_id}/vertices")
    v_map = {item["name"]: item for item in res_v.json()}
    assert v_map["v_out"]["state"] == "data ready"
    assert v_map["v_out"]["processed_count"] == 1


def test_sse_executor_harness_echo_call(client: TestClient):
    """Test SSEExecutor non-streaming response returns standard tool call echo format."""
    session_id = "sse_echo_sess"

    client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={"name": "v_start", "content": "initial input", "attributes": ["start"], "state": "data ready"}
    )
    client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={"name": "v_end", "content": "", "attributes": ["end"], "state": "todo"}
    )
    client.post(
        f"/api/sessions/{session_id}/graph/edges",
        json={
            "id": "e_echo_step",
            "type": "code",
            "input_vertex": "v_start",
            "output_vertex": "v_end",
            "settings": {}
        }
    )

    # Invoke SSE executor non-streaming
    res = client.post(
        "/api/sse/execute",
        json={
            "session_id": session_id,
            "stream": False,
            "input_payload": "injected request payload"
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "function"
    assert data["function"]["name"] == "echo"

    args = json.loads(data["function"]["arguments"])
    assert "info" in args
    info = args["info"]
    assert info["session_id"] == session_id
    assert info["success"] is True
    assert "e_echo_step" in info["completed_edges"]
    assert info["vertex_states"]["v_end"] == "data ready"


def test_sse_executor_streaming_echo(client: TestClient):
    """Test SSEExecutor streaming response returns standard tool call echo SSE chunks."""
    session_id = "sse_stream_sess"

    client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={"name": "v_s", "content": "stream input", "attributes": ["start"], "state": "data ready"}
    )
    client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={"name": "v_e", "content": "", "attributes": ["end"], "state": "todo"}
    )
    client.post(
        f"/api/sessions/{session_id}/graph/edges",
        json={
            "id": "e_s",
            "type": "code",
            "input_vertex": "v_s",
            "output_vertex": "v_e",
            "settings": {}
        }
    )

    # Invoke SSE executor streaming
    res = client.post(
        "/api/sse/execute",
        json={
            "session_id": session_id,
            "stream": True,
        }
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    text = res.text
    assert "data: [DONE]" in text
    assert "vea-v4-sse-executor" in text
    assert '"function": {"name": "echo"' in text
