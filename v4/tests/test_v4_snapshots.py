import json
import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from framework.graph_v4 import GraphV4, DiscreteGraphLoaderV4
from framework.vertex_v4 import VertexStoreV4, VertexStateV4
from framework.edge_v4 import CodeEdgeV4
from framework.executor_v4 import ExecutorV4
from framework.snapshot_v4 import GraphSnapshotManagerV4
from framework.server_v4 import SessionGraphManagerV4, create_v4_server


def dummy_step1(content: str, settings=None, staging=None) -> str:
    return "step1_output"


def dummy_step2(content: str, settings=None, staging=None) -> str:
    return f"{content}_step2_output"


def test_snapshot_manager_save_and_load(tmp_path: Path):
    """Test manual saving, listing, reading, and loading complete graph snapshots."""
    snap_dir = tmp_path / "test_snapshots"
    mgr = GraphSnapshotManagerV4(base_dir=snap_dir)
    session_id = "sess_snap_1"

    # Create a graph
    graph = GraphV4(session_id=session_id, name="demo_graph")
    graph.add_vertex("v_start", content="init_data", state=VertexStateV4.DATA_READY.value, attributes=["start"])
    graph.add_vertex("v_mid", content="", state=VertexStateV4.TODO.value)
    edge = CodeEdgeV4(edge_id="e1", input_vertex="v_start", output_vertex="v_mid", script="tests.test_v4_snapshots:dummy_step1")
    graph.add_edge(edge)

    # Save initial snapshot
    p0 = mgr.save_snapshot(graph, trigger="init")
    assert p0.exists()
    assert p0.name == "step_0000_init.json"

    # Verify JSON content is a complete graph representation
    with open(p0, "r", encoding="utf-8") as f:
        data0 = json.load(f)

    assert data0["version"] == "4.0"
    assert data0["session_id"] == session_id
    assert data0["snapshot_step"] == 0
    assert data0["snapshot_trigger"] == "init"
    assert len(data0["vertices"]) == 2
    assert len(data0["edges"]) == 1
    assert data0["vertices"][0]["name"] in ("v_start", "v_mid")

    # Mutate graph (e.g. advance state)
    graph.vertices["v_mid"].state = VertexStateV4.DATA_READY.value
    graph.vertices["v_mid"].content = "mid_data"

    # Save second snapshot
    p1 = mgr.save_snapshot(graph, trigger="edge_completed:e1")
    assert p1.exists()
    assert p1.name == "step_0001_edge_completed_e1.json"

    # List snapshots
    snaps = mgr.list_snapshots(session_id)
    assert len(snaps) == 2
    assert snaps[0]["step"] == 0
    assert snaps[0]["trigger"] == "init"
    assert snaps[1]["step"] == 1
    assert snaps[1]["trigger"] == "edge_completed:e1"

    # Retrieve specific snapshot data
    d0 = mgr.get_snapshot_data(session_id, 0)
    assert d0 is not None
    assert d0["snapshot_step"] == 0

    # Load as GraphV4
    loaded_g0 = mgr.load_snapshot_as_graph(session_id, 0)
    assert loaded_g0 is not None
    assert loaded_g0.session_id == session_id
    assert "v_start" in loaded_g0.vertices
    assert loaded_g0.vertices["v_mid"].state == VertexStateV4.TODO.value

    loaded_g1 = mgr.load_snapshot_as_graph(session_id, 1)
    assert loaded_g1 is not None
    assert loaded_g1.vertices["v_mid"].state == VertexStateV4.DATA_READY.value
    assert loaded_g1.vertices["v_mid"].content == "mid_data"


def test_snapshot_restore_to_store(tmp_path: Path):
    """Test time-travel rollback / restore from a historical snapshot to SQLite store."""
    snap_dir = tmp_path / "restore_snapshots"
    mgr = GraphSnapshotManagerV4(base_dir=snap_dir)
    session_id = "sess_rollback"

    store = VertexStoreV4(str(tmp_path / "rollback.db"))

    # Initial state (Step 0)
    graph0 = GraphV4(session_id=session_id, name="rollback_test")
    graph0.add_vertex("v1", content="version_1", state=VertexStateV4.DATA_READY.value)
    DiscreteGraphLoaderV4.populate_store(graph0, store)
    mgr.save_snapshot(graph0, trigger="v1_initial", store=store)

    # Later state (Step 1)
    store.save_vertex(session_id=session_id, name="v1", content="version_2_corrupted", state=VertexStateV4.REJECT.value)
    store.save_vertex(session_id=session_id, name="v2", content="added_later", state=VertexStateV4.TODO.value)
    graph1 = GraphV4.load_from_store(store, session_id)
    mgr.save_snapshot(graph1, trigger="v2_corrupted", store=store)

    # Verify store currently has version_2
    curr_v1 = store.get_vertex(session_id, "v1")
    assert curr_v1.content == "version_2_corrupted"
    assert len(store.list_vertices(session_id)) == 2

    # Perform Rollback to Step 0
    restored_graph = mgr.restore_snapshot(session_id, step=0, store=store)
    assert restored_graph is not None

    # Verify store is rolled back to Step 0 state
    v1_restored = store.get_vertex(session_id, "v1")
    assert v1_restored.content == "version_1"
    assert v1_restored.state == VertexStateV4.DATA_READY.value
    assert store.get_vertex(session_id, "v2") is None
    assert len(store.list_vertices(session_id)) == 1


@pytest.mark.asyncio
async def test_executor_automatic_snapshots(tmp_path: Path):
    """Test that ExecutorV4 automatically creates complete graph snapshots on start, edge finish, and completion."""
    snap_dir = tmp_path / "exec_snaps"
    session_id = "sess_auto_exec"

    store = VertexStoreV4(str(tmp_path / "exec.db"))
    graph = GraphV4(session_id=session_id, name="auto_graph")
    graph.add_vertex("v_in", content="start", state=VertexStateV4.DATA_READY.value, attributes=["start"])
    graph.add_vertex("v_mid", content="", state=VertexStateV4.TODO.value)
    graph.add_vertex("v_out", content="", state=VertexStateV4.TODO.value, attributes=["end"])

    e1 = CodeEdgeV4(edge_id="e1", input_vertex="v_in", output_vertex="v_mid", script=dummy_step1)
    e2 = CodeEdgeV4(edge_id="e2", input_vertex="v_mid", output_vertex="v_out", script=dummy_step2)
    graph.add_edge(e1)
    graph.add_edge(e2)

    DiscreteGraphLoaderV4.populate_store(graph, store)

    executor = ExecutorV4(
        graph=graph,
        store=store,
        snapshot_dir=snap_dir,
        max_concurrency=2,
    )

    result = await executor.run()
    assert result.success is True

    mgr = GraphSnapshotManagerV4(base_dir=snap_dir)
    snapshots = mgr.list_snapshots(session_id)

    # Expect: execution_start, edge_completed:e1, edge_completed:e2, execution_finished
    assert len(snapshots) >= 4
    triggers = [s["trigger"] for s in snapshots]
    assert triggers[0] == "execution_start"
    assert any("edge_completed:e1" in t for t in triggers)
    assert any("edge_completed:e2" in t for t in triggers)
    assert triggers[-1] == "execution_finished"

    # Verify final snapshot contains final output
    final_data = mgr.get_snapshot_data(session_id, snapshots[-1]["step"])
    out_vertex = next(v for v in final_data["vertices"] if v["name"] == "v_out")
    assert out_vertex["state"] == VertexStateV4.DATA_READY.value
    assert out_vertex["content"] == "step1_output_step2_output"


@pytest.mark.asyncio
async def test_server_snapshot_rest_api(tmp_path: Path):
    """Test HTTP REST endpoints for querying, creating, and restoring complete graph snapshots."""
    snap_dir = tmp_path / "server_snaps"
    session_id = "sess_api_test"

    store = VertexStoreV4(str(tmp_path / "server.db"))
    app = create_v4_server(
        store_or_db=store,
        snapshot_dir=snap_dir,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Add vertex via API -> should trigger snapshot
        res = await client.post(
            f"/api/sessions/{session_id}/graph/vertices",
            json={"name": "v_start", "content": "api_init", "state": "data ready", "attributes": ["start"]},
        )
        assert res.status_code == 200

        # 2. Add another vertex via API -> triggers snapshot
        res = await client.post(
            f"/api/sessions/{session_id}/graph/vertices",
            json={"name": "v_end", "content": "", "state": "todo", "attributes": ["end"]},
        )
        assert res.status_code == 200

        # 3. Add edge via API -> triggers snapshot
        res = await client.post(
            f"/api/sessions/{session_id}/graph/edges",
            json={
                "id": "e_api",
                "type": "code",
                "input_vertex": "v_start",
                "output_vertex": "v_end",
                "script": "tests.test_v4_snapshots:dummy_step1",
            },
        )
        assert res.status_code == 200

        # 4. List snapshots
        res = await client.get(f"/api/sessions/{session_id}/snapshots")
        assert res.status_code == 200
        snaps = res.json()["snapshots"]
        assert len(snaps) == 3
        assert snaps[0]["step"] == 0
        assert "vertex_saved:v_start" in snaps[0]["trigger"]
        assert "vertex_saved:v_end" in snaps[1]["trigger"]
        assert "edge_saved:e_api" in snaps[2]["trigger"]

        # 5. Get complete snapshot JSON by step
        res = await client.get(f"/api/sessions/{session_id}/snapshots/0")
        assert res.status_code == 200
        step0_data = res.json()
        assert step0_data["version"] == "4.0"
        assert len(step0_data["vertices"]) == 1

        # 6. Manually create explicit snapshot
        res = await client.post(
            f"/api/sessions/{session_id}/snapshots",
            json={"trigger": "manual_checkpoint_before_experiment"},
        )
        assert res.status_code == 200

        # 7. Add bad vertex and rollback
        await client.post(
            f"/api/sessions/{session_id}/graph/vertices",
            json={"name": "v_bad", "content": "corrupted", "state": "reject"},
        )

        # Restore to step 0
        res = await client.post(f"/api/sessions/{session_id}/snapshots/0/restore")
        assert res.status_code == 200
        assert res.json()["status"] == "restored"
        assert res.json()["vertices"] == 1

        # Check vertices after restore
        res = await client.get(f"/api/sessions/{session_id}/graph/relationships")
        assert res.status_code == 200
        assert "v_start" in res.json()["nodes"]
        assert "v_bad" not in res.json()["nodes"]
