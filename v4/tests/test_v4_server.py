"""Automated test suite for V4 standalone server and online graph API.

Verifies:
1. Online graph manipulation (adding/updating/deleting vertices and edges).
2. Multi-session isolation (independent graphs per session).
3. Live workflow execution via API.
4. Database inspection endpoints (stats, vertices, session_staging).
5. Interactive dashboard HTML serving.
"""

from __future__ import annotations

import asyncio
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

    res_js = client.get("/dashboard/app.js")
    assert res_js.status_code == 200
    assert "application/javascript" in res_js.headers["content-type"]
    assert "renderDagGraph" in res_js.text

    res_css = client.get("/dashboard/app.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers["content-type"]
    assert "brand" in res_css.text


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


def test_edge_sqlite_persistence_and_hydration(tmp_path):
    """Test that edges created via online API are persisted into SQLite and properly hydrated on fresh manager."""
    from framework.vertex_v4 import VertexStoreV4
    from framework.server_v4 import SessionGraphManagerV4, create_v4_server
    from fastapi.testclient import TestClient

    db_file = tmp_path / "test_persistence.db"
    store = VertexStoreV4(db_file)
    manager = SessionGraphManagerV4(store)
    app = create_v4_server(store_or_db=store, manager=manager)
    client = TestClient(app)

    session_id = "sess_persist"
    client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={"name": "node_a", "content": "hello", "attributes": ["start"], "state": "data ready"}
    )
    client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={"name": "node_b", "content": "", "attributes": ["end"], "state": "todo"}
    )
    client.post(
        f"/api/sessions/{session_id}/graph/edges",
        json={
            "id": "edge_ab",
            "type": "code",
            "input_vertex": "node_a",
            "output_vertex": "node_b",
            "settings": {"k": "v"}
        }
    )

    # Verify edge exists in SQLite
    edges_in_db = store.list_edges(session_id)
    assert len(edges_in_db) == 1
    assert edges_in_db[0].edge_id == "edge_ab"
    assert edges_in_db[0].input_vertex == "node_a"
    assert edges_in_db[0].output_vertex == "node_b"

    # Now create a fresh manager simulating server restart
    fresh_manager = SessionGraphManagerV4(store)
    hydrated_graph = fresh_manager.get_or_create_graph(session_id)
    assert "node_a" in hydrated_graph.vertices
    assert "node_b" in hydrated_graph.vertices
    assert "edge_ab" in hydrated_graph.edges
    assert hydrated_graph.edges["edge_ab"].input_vertex == "node_a"


def test_online_graph_endpoint_reflects_runtime_db_state(client: TestClient):
    """Test that GET /api/sessions/{session_id}/graph returns current DB state after workflow run."""
    session_id = "sess_sync_check"
    client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={"name": "v_in", "content": "payload_val", "attributes": ["start"], "state": "data ready"}
    )
    client.post(
        f"/api/sessions/{session_id}/graph/vertices",
        json={"name": "v_out", "content": "", "attributes": ["end"], "state": "todo"}
    )
    client.post(
        f"/api/sessions/{session_id}/graph/edges",
        json={"id": "e1", "type": "code", "input_vertex": "v_in", "output_vertex": "v_out"}
    )

    # Run workflow
    res_run = client.post(f"/api/sessions/{session_id}/run")
    assert res_run.status_code == 200

    # Query GET /graph
    res_graph = client.get(f"/api/sessions/{session_id}/graph")
    assert res_graph.status_code == 200
    graph_data = res_graph.json()
    assert graph_data["vertices"]["v_out"]["state"] == "data ready"
    assert graph_data["vertices"]["v_out"]["content"] == "payload_val"
    assert graph_data["vertices"]["v_out"]["processed_count"] == 1


def test_subgraph_downstream_dataflow_closure(tmp_path):
    """Test end-to-end dataflow closure across nested subgraphs and parent downstream vertices."""
    import json
    from framework.vertex_v4 import VertexStoreV4
    from framework.server_v4 import SessionGraphManagerV4
    from framework.sse_executor_v4 import SSEExecutorV4
    from framework.graph_v4 import GraphV4
    from framework.edge_v4 import CodeEdgeV4
    from framework.vertex_v4 import VertexRecordV4, VertexStateV4, VertexAttributeV4

    # 1. Create a child subgraph manifest on disk
    sub_dir = tmp_path / "child_graph"
    sub_dir.mkdir(parents=True)
    v_start_file = sub_dir / "sub_in.json"
    v_start_file.write_text(json.dumps({
        "name": "sub_start",
        "content": "",
        "attributes": ["start"],
        "state": "idle"
    }))
    v_end_file = sub_dir / "sub_out.json"
    v_end_file.write_text(json.dumps({
        "name": "sub_end",
        "content": "",
        "attributes": ["end"],
        "state": "todo"
    }))

    # Custom script that transforms data inside the child subgraph
    script_file = sub_dir / "transform.py"
    script_file.write_text("def process(data, settings=None, staging=None):\n    return f'PROCESSED({data})'\n")

    e_sub_file = sub_dir / "e_inner.json"
    e_sub_file.write_text(json.dumps({
        "id": "e_inner",
        "type": "code",
        "input_vertex": "sub_start",
        "output_vertex": "sub_end",
        "script": f"{script_file}:process"
    }))

    sub_manifest = sub_dir / "graph.json"
    sub_manifest.write_text(json.dumps({
        "version": "4.0",
        "session_id": "child_sess",
        "metadata": {"name": "Child"},
        "vertices": ["sub_in.json", "sub_out.json"],
        "edges": ["e_inner.json"]
    }))

    # 2. Build parent graph: Root -> SubgraphBox -> Sink
    parent_session = "parent_sess_closure"
    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    parent_graph = manager.get_or_create_graph(parent_session)

    v_root = VertexRecordV4(
        id=0,
        session_id=parent_session,
        name="v_root",
        content="raw_user_input",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value,
    )
    v_box = VertexRecordV4(
        id=0,
        session_id=parent_session,
        name="v_box",
        content=json.dumps({"subgraph_manifest": str(sub_manifest)}),
        attributes=[VertexAttributeV4.SUBGRAPH.value],
        state=VertexStateV4.TODO.value,
    )
    v_sink = VertexRecordV4(
        id=0,
        session_id=parent_session,
        name="v_sink",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value,
    )
    parent_graph.add_vertex(v_root)
    parent_graph.add_vertex(v_box)
    parent_graph.add_vertex(v_sink)

    # Edges: v_root -> v_box, v_box -> v_sink
    e1 = CodeEdgeV4("e_to_box", "v_root", "v_box")
    e2 = CodeEdgeV4("e_from_box", "v_box", "v_sink")
    parent_graph.add_edge(e1)
    parent_graph.add_edge(e2)

    # 3. Execute via SSEExecutor
    executor = SSEExecutorV4(manager=manager, store=store)
    res = asyncio.run(executor.execute_harness_call(session_id=parent_session))

    assert res["type"] == "function"
    info = json.loads(res["function"]["arguments"])["info"]
    assert info["success"] is True

    # 4. Verify that downstream sink vertex received the transformed data from the child subgraph!
    sink_rec = store.get_vertex(parent_session, "v_sink")
    assert sink_rec is not None
    assert sink_rec.state == VertexStateV4.DATA_READY.value
    assert sink_rec.content == "PROCESSED(raw_user_input)"



def test_api_vertex_reentry_endpoint(client: TestClient):
    """Test /api/sessions/{session_id}/graph/vertices/{name}/reenter REST endpoint."""
    session_id = "api_reenter_sess"
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "a", "content": "c1", "state": "data ready"})
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "b", "content": "c2", "state": "data ready"})
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "c", "content": "c3", "state": "data ready"})
    client.post(f"/api/sessions/{session_id}/graph/edges", json={"id": "e1", "type": "code", "input_vertex": "a", "output_vertex": "b"})
    client.post(f"/api/sessions/{session_id}/graph/edges", json={"id": "e2", "type": "code", "input_vertex": "b", "output_vertex": "c"})

    # Trigger re-entry of b with new content
    res = client.post(
        f"/api/sessions/{session_id}/graph/vertices/b/reenter",
        json={"new_content": "new_b_content", "reset_state": "todo"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "reentered"
    assert data["reentered_vertex"] == "b"
    assert "c" in data["affected_downstream_vertices"]

    # Verify updated DB states
    res_b = client.get(f"/api/db/sessions/{session_id}/vertices")
    v_map = {item["name"]: item for item in res_b.json()}
    assert v_map["b"]["content"] == "new_b_content"
    assert v_map["b"]["state"] == "data ready"
    assert v_map["c"]["state"] == "todo"
    assert v_map["a"]["state"] == "data ready"


def test_api_edge_reconnect_endpoint(client: TestClient):
    """Test /api/sessions/{session_id}/graph/edges/{edge_id}/reconnect REST endpoint."""
    session_id = "api_reconnect_sess"
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "n1"})
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "n2"})
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "n3"})
    client.post(f"/api/sessions/{session_id}/graph/edges", json={"id": "e_rec", "type": "code", "input_vertex": "n1", "output_vertex": "n2"})

    # Reconnect e_rec to output to n3
    res = client.patch(
        f"/api/sessions/{session_id}/graph/edges/e_rec/reconnect",
        json={"new_output_vertex": "n3"}
    )
    assert res.status_code == 200
    assert res.json()["status"] == "reconnected"

    graph_res = client.get(f"/api/sessions/{session_id}/graph")
    edge_rec = graph_res.json()["edges"]["e_rec"]
    assert edge_rec["output_vertex"] == "n3"


def test_api_subgraph_splice_endpoint(client: TestClient):
    """Test /api/sessions/{session_id}/graph/subgraphs/splice REST endpoint."""
    session_id = "api_splice_sess"
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "src", "attributes": ["start"]})
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "placeholder"})
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "dst", "attributes": ["end"]})
    client.post(f"/api/sessions/{session_id}/graph/edges", json={"id": "e1", "type": "code", "input_vertex": "src", "output_vertex": "placeholder"})
    client.post(f"/api/sessions/{session_id}/graph/edges", json={"id": "e2", "type": "code", "input_vertex": "placeholder", "output_vertex": "dst"})

    subgraph_data = {
        "vertices": [
            {"name": "sub_in", "attributes": ["start"], "state": "todo"},
            {"name": "sub_out", "attributes": ["end"], "state": "todo"}
        ],
        "edges": [
            {"id": "e_internal", "type": "code", "input_vertex": "sub_in", "output_vertex": "sub_out"}
        ]
    }

    res = client.post(
        f"/api/sessions/{session_id}/graph/subgraphs/splice",
        json={
            "target_vertex": "placeholder",
            "subgraph_data": subgraph_data,
            "name_prefix": "pfx"
        }
    )
    assert res.status_code == 200
    assert res.json()["status"] == "spliced"

    # Verify placeholder is gone and replaced by pfx_sub_in and pfx_sub_out
    g_res = client.get(f"/api/sessions/{session_id}/graph")
    v_dict = g_res.json()["vertices"]
    assert "placeholder" not in v_dict
    assert "pfx_sub_in" in v_dict
    assert "pfx_sub_out" in v_dict
    assert g_res.json()["valid"] is True


def test_api_graph_loaded_nodes_and_relationships(client: TestClient):
    """Test REST API inspection of loaded nodes and graph relationships."""
    session_id = "test_api_rel_sess"

    # Add vertices: n1 -> n2 -> n3
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "n1", "state": "data ready"})
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "n2", "state": "todo"})
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "n3", "state": "todo"})

    # Add edges
    client.post(f"/api/sessions/{session_id}/graph/edges", json={"id": "e12", "type": "code", "input_vertex": "n1", "output_vertex": "n2"})
    client.post(f"/api/sessions/{session_id}/graph/edges", json={"id": "e23", "type": "code", "input_vertex": "n2", "output_vertex": "n3"})

    # 1. Test GET loaded nodes
    nodes_res = client.get(f"/api/sessions/{session_id}/graph/nodes")
    assert nodes_res.status_code == 200
    nodes_data = nodes_res.json()
    assert nodes_data["total"] == 3
    node_names = [n["name"] for n in nodes_data["nodes"]]
    assert set(node_names) == {"n1", "n2", "n3"}

    # 2. Test GET single node relationship
    n2_res = client.get(f"/api/sessions/{session_id}/graph/nodes/n2/relationships")
    assert n2_res.status_code == 200
    n2_data = n2_res.json()
    assert n2_data["predecessors"] == ["n1"]
    assert n2_data["successors"] == ["n3"]
    assert n2_data["in_degree"] == 1
    assert n2_data["out_degree"] == 1

    # 3. Test GET non-existent node relationship
    err_res = client.get(f"/api/sessions/{session_id}/graph/nodes/unknown/relationships")
    assert err_res.status_code == 404

    # 4. Test GET full graph relationships
    rel_res = client.get(f"/api/sessions/{session_id}/graph/relationships")
    assert rel_res.status_code == 200
    rel_data = rel_res.json()
    assert rel_data["roots"] == ["n1"]
    assert rel_data["sinks"] == ["n3"]
    assert rel_data["adjacency_list"]["n1"] == ["n2"]
    assert rel_data["adjacency_list"]["n2"] == ["n3"]


def test_api_add_subgraph_endpoint(client: TestClient):
    """Test /api/sessions/{session_id}/graph/subgraphs/add REST endpoint with persistence."""
    session_id = "test_api_add_sub_sess"

    # 1. Base graph: head and tail
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "head", "state": "data ready"})
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "tail", "state": "todo"})

    # 2. Subgraph payload
    subgraph_data = {
        "version": "4.0",
        "name": "worker_pipeline",
        "vertices": [
            {"name": "worker_1", "state": "idle", "content": "w1"},
            {"name": "worker_2", "state": "idle", "content": "w2"},
        ],
        "edges": [
            {"id": "w1_to_w2", "type": "code", "input_vertex": "worker_1", "output_vertex": "worker_2"}
        ],
    }

    # 3. Add subgraph via API
    add_res = client.post(
        f"/api/sessions/{session_id}/graph/subgraphs/add",
        json={
            "subgraph_data": subgraph_data,
            "name_prefix": "pkg",
            "incoming_bindings": {"head": "worker_1"},
            "connections": [{"from": "pkg_worker_2", "to": "tail", "type": "code"}],
            "source": "registry:worker_v4",
        },
    )
    assert add_res.status_code == 200
    res_body = add_res.json()
    assert res_body["status"] == "added"
    assert "pkg_worker_1" in res_body["result"]["added_vertices"]
    assert "pkg_worker_2" in res_body["result"]["added_vertices"]

    # 4. Verify graph and loaded nodes
    nodes_res = client.get(f"/api/sessions/{session_id}/graph/nodes")
    assert nodes_res.status_code == 200
    loaded_names = [n["name"] for n in nodes_res.json()["nodes"]]
    assert "pkg_worker_1" in loaded_names
    assert "pkg_worker_2" in loaded_names

    # 5. Verify relationships
    w1_rel = client.get(f"/api/sessions/{session_id}/graph/nodes/pkg_worker_1/relationships").json()
    assert w1_rel["predecessors"] == ["head"]
    assert w1_rel["successors"] == ["pkg_worker_2"]

    tail_rel = client.get(f"/api/sessions/{session_id}/graph/nodes/tail/relationships").json()
    assert tail_rel["predecessors"] == ["pkg_worker_2"]


def test_api_dump_graph_endpoint(client: TestClient, tmp_path: Path):
    """Test /api/sessions/{session_id}/graph/dump REST endpoint."""
    session_id = "test_dump_api_sess"
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "v_start", "state": "data ready"})
    client.post(f"/api/sessions/{session_id}/graph/vertices", json={"name": "v_stop", "state": "todo"})
    client.post(f"/api/sessions/{session_id}/graph/edges", json={"id": "e_flow", "type": "code", "input_vertex": "v_start", "output_vertex": "v_stop"})

    # 1. GET dump
    get_res = client.get(f"/api/sessions/{session_id}/graph/dump")
    assert get_res.status_code == 200
    dump_data = get_res.json()
    assert dump_data["status"] == "dumped"
    assert dump_data["session_id"] == session_id
    g = dump_data["graph"]
    assert g["version"] == "4.0"
    assert len(g["vertices"]) == 2
    assert len(g["edges"]) == 1
    assert g["relationships"]["roots"] == ["v_start"]
    assert g["relationships"]["sinks"] == ["v_stop"]

    # 2. POST dump to filesystem
    client.app.state.manifest_base_dir = tmp_path
    target_file = str(tmp_path / "dump_export.json")
    post_res = client.post(f"/api/sessions/{session_id}/graph/dump?path={target_file}")
    assert post_res.status_code == 200
    assert Path(target_file).exists()
    with open(target_file, "r", encoding="utf-8") as f:
        exported = json.load(f)
    assert exported["session_id"] == session_id
    assert len(exported["vertices"]) == 2



