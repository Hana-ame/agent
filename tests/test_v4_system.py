"""Comprehensive test suite for V4 architecture.

Tests:
1. VertexStoreV4 CRUD and session_staging attribution.
2. EdgeV4 standalone execution (Python API and CLI).
3. DiscreteGraphLoaderV4 manifest parsing and DAG validation.
4. ExecutorV4 concurrency control, topological DAG tier order, and reflexive recovery.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from framework.edge_v4 import CodeEdgeV4, EdgeResultV4, EdgeV4, LLMEdgeV4, ReflexiveEdgeV4
from framework.executor_v4 import ExecutionResultV4, ExecutorV4
from framework.graph_v4 import DiscreteGraphLoaderV4, GraphTopologyError, GraphV4
from framework.vertex_v4 import (
    StagingRecordV4,
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)


@pytest.fixture
def mem_store() -> VertexStoreV4:
    """Fixture providing an in-memory SQLite store."""
    store = VertexStoreV4(":memory:")
    yield store
    store.close()


# =====================================================================
# 1. Storage & Staging Tests
# =====================================================================


def test_vertex_store_crud(mem_store: VertexStoreV4):
    """Test creating, fetching, and updating vertex records in SQLite."""
    session_id = "test_crud_sess"

    v1 = mem_store.save_vertex(
        session_id=session_id,
        name="source_node",
        content="raw payload",
        attributes=[VertexAttributeV4.START, VertexAttributeV4.PLAIN_TEXT],
        state=VertexStateV4.DATA_READY,
    )
    assert v1.id > 0
    assert v1.name == "source_node"
    assert v1.content == "raw payload"
    assert v1.state == VertexStateV4.DATA_READY.value
    assert v1.has_attribute(VertexAttributeV4.START)

    # Fetch
    fetched = mem_store.get_vertex(session_id, "source_node")
    assert fetched is not None
    assert fetched.name == "source_node"

    # Update state
    ok = mem_store.update_vertex_state(session_id, "source_node", VertexStateV4.IDLE)
    assert ok is True
    assert mem_store.get_vertex(session_id, "source_node").state == VertexStateV4.IDLE.value

    # Update content and increment count
    ok = mem_store.update_vertex_content(
        session_id=session_id,
        name="source_node",
        content="modified payload",
        state=VertexStateV4.DATA_READY,
        increment_count=True,
    )
    assert ok is True
    updated = mem_store.get_vertex(session_id, "source_node")
    assert updated.content == "modified payload"
    assert updated.state == VertexStateV4.DATA_READY.value
    assert updated.processed_count == 1


def test_session_staging_attribution(mem_store: VertexStoreV4):
    """Test staging table records drafts and feedback tagged by producing edge."""
    session_id = "test_staging_sess"

    # Stage intermediate data
    s1_id = mem_store.stage_output(
        session_id=session_id,
        edge_id="edge_llm_1",
        key="prompt_draft",
        value="Summarize the quarterly results",
        vertex_name="report_out",
        metadata={"tokens": 15},
    )
    assert s1_id > 0

    s2_id = mem_store.stage_output(
        session_id=session_id,
        edge_id="edge_validator_2",
        key="error_feedback",
        value="Missing required section: Balance Sheet",
        vertex_name="report_out",
        metadata={"code": 422},
    )
    assert s2_id > s1_id

    # Query latest for vertex
    latest = mem_store.get_latest_staged_for_vertex(session_id, "report_out")
    assert latest is not None
    assert latest.key == "error_feedback"
    assert latest.edge_id == "edge_validator_2"

    # Query specifically for prompt_draft
    prompt_record = mem_store.get_latest_staged_for_vertex(session_id, "report_out", key="prompt_draft")
    assert prompt_record is not None
    assert prompt_record.value == "Summarize the quarterly results"

    # Query all staged by edge
    edge_records = mem_store.get_staged(session_id, edge_id="edge_llm_1")
    assert len(edge_records) == 1
    assert edge_records[0].key == "prompt_draft"


# =====================================================================
# 2. Standalone Edge Tests
# =====================================================================


@pytest.mark.asyncio
async def test_code_edge_handshake_and_execution(mem_store: VertexStoreV4):
    """Test CodeEdgeV4 enforces two-sided handshake and executes transformation."""
    session_id = "edge_handshake_sess"

    mem_store.save_vertex(session_id, "in_v", "hello world", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(session_id, "out_v", "", state=VertexStateV4.TODO)

    edge = CodeEdgeV4("edge_upper", "in_v", "out_v", script=lambda content, settings, staging: content.upper())

    res = await edge.run(session_id, mem_store)
    assert res.success is True
    assert res.output == "HELLO WORLD"

    out_v = mem_store.get_vertex(session_id, "out_v")
    assert out_v.content == "HELLO WORLD"
    assert out_v.state == VertexStateV4.DATA_READY.value
    assert out_v.processed_count == 1


@pytest.mark.asyncio
async def test_code_edge_skipped_when_handshake_unmet(mem_store: VertexStoreV4):
    """Edge must skip execution when upstream is not data ready or downstream is not todo."""
    session_id = "edge_skip_sess"

    # Upstream is idle (not data ready)
    mem_store.save_vertex(session_id, "in_v", "not ready", state=VertexStateV4.IDLE)
    mem_store.save_vertex(session_id, "out_v", "", state=VertexStateV4.TODO)

    edge = CodeEdgeV4("edge_skip", "in_v", "out_v", script=lambda c, s, st: c)
    res = await edge.run(session_id, mem_store)
    assert res.success is False
    assert res.skipped is True
    assert "required 'data ready'" in (res.reason or "")

    # Now upstream is data ready, but downstream is already data ready (not todo)
    mem_store.update_vertex_state(session_id, "in_v", VertexStateV4.DATA_READY)
    mem_store.update_vertex_state(session_id, "out_v", VertexStateV4.DATA_READY)

    res2 = await edge.run(session_id, mem_store)
    assert res2.success is False
    assert res2.skipped is True
    assert "required 'todo'" in (res2.reason or "")


@pytest.mark.asyncio
async def test_code_edge_failure_transitions_to_reject(mem_store: VertexStoreV4):
    """When edge throws exception, downstream transitions to reject and error is staged."""
    session_id = "edge_err_sess"

    mem_store.save_vertex(session_id, "in_v", "bad data", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(session_id, "out_v", "", state=VertexStateV4.TODO)

    def faulty_transform(content, settings, staging):
        raise ValueError("Invalid format in input data")

    edge = CodeEdgeV4("edge_faulty", "in_v", "out_v", script=faulty_transform)
    res = await edge.run(session_id, mem_store)

    assert res.success is False
    assert res.error == "Invalid format in input data"

    out_v = mem_store.get_vertex(session_id, "out_v")
    assert out_v.state == VertexStateV4.REJECT.value

    staged = mem_store.get_latest_staged_for_vertex(session_id, "out_v", key="error_feedback")
    assert staged is not None
    assert "Invalid format in input data" in staged.value


def test_edge_standalone_cli_execution(tmp_path: Path):
    """Test running edge_v4 directly from the command line."""
    db_file = tmp_path / "standalone_cli.db"
    store = VertexStoreV4(str(db_file))
    session_id = "cli_session"
    store.save_vertex(session_id, "src", "cli input", state=VertexStateV4.DATA_READY)
    store.save_vertex(session_id, "dst", "", state=VertexStateV4.TODO)
    store.close()

    cmd = [
        sys.executable,
        "-m",
        "framework.edge_v4",
        "--db",
        str(db_file),
        "--session",
        session_id,
        "--edge-id",
        "cli_edge_1",
        "--type",
        "code",
        "--input",
        "src",
        "--output",
        "dst",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI execution failed: {proc.stderr}"

    output_data = json.loads(proc.stdout)
    assert output_data["success"] is True
    assert output_data["edge_id"] == "cli_edge_1"
    assert output_data["output"] == "cli input"

    # Verify destination was updated in the DB
    verify_store = VertexStoreV4(str(db_file))
    dst_v = verify_store.get_vertex(session_id, "dst")
    assert dst_v.state == VertexStateV4.DATA_READY.value
    assert dst_v.content == "cli input"
    verify_store.close()


# =====================================================================
# 3. Discrete Graph Loader Tests
# =====================================================================


def test_discrete_graph_loader(tmp_path: Path):
    """Test loading a discrete graph layout from individual JSON files."""
    graph_dir = tmp_path / "test_graph"
    vertices_dir = graph_dir / "vertices"
    edges_dir = graph_dir / "edges"
    vertices_dir.mkdir(parents=True)
    edges_dir.mkdir(parents=True)

    # 1. Discrete vertices
    (vertices_dir / "v_in.json").write_text(
        json.dumps({
            "name": "v_in",
            "content": "initial data",
            "attributes": ["start", "plain text"],
            "state": "data ready"
        }),
        encoding="utf-8"
    )
    (vertices_dir / "v_mid.json").write_text(
        json.dumps({
            "name": "v_mid",
            "content": "",
            "attributes": [],
            "state": "todo"
        }),
        encoding="utf-8"
    )
    (vertices_dir / "v_out.json").write_text(
        json.dumps({
            "name": "v_out",
            "content": "",
            "attributes": ["end"],
            "state": "todo"
        }),
        encoding="utf-8"
    )

    # 2. Discrete edges
    (edges_dir / "e1.json").write_text(
        json.dumps({
            "id": "e1",
            "type": "code",
            "input_vertex": "v_in",
            "output_vertex": "v_mid",
            "settings": {}
        }),
        encoding="utf-8"
    )
    (edges_dir / "e2.json").write_text(
        json.dumps({
            "id": "e2",
            "type": "code",
            "input_vertex": "v_mid",
            "output_vertex": "v_out",
            "settings": {}
        }),
        encoding="utf-8"
    )
    (edges_dir / "e_recovery.json").write_text(
        json.dumps({
            "id": "e_recovery",
            "type": "reflexive",
            "input_vertex": "v_mid",
            "output_vertex": "v_mid",
            "trigger_state": "reject",
            "target_state": "todo urgent",
            "settings": {"max_retries": 3}
        }),
        encoding="utf-8"
    )

    # 3. Master manifest
    manifest_path = graph_dir / "graph.json"
    manifest_path.write_text(
        json.dumps({
            "version": "4.0",
            "session_id": "discrete_sess",
            "metadata": {"name": "Discrete Pipeline"},
            "vertices": [
                "vertices/v_in.json",
                "vertices/v_mid.json",
                "vertices/v_out.json"
            ],
            "edges": [
                "edges/e1.json",
                "edges/e2.json",
                "edges/e_recovery.json"
            ]
        }),
        encoding="utf-8"
    )

    # Load and validate
    graph = DiscreteGraphLoaderV4.load_from_manifest(manifest_path)
    assert graph.session_id == "discrete_sess"
    assert len(graph.vertices) == 3
    assert len(graph.edges) == 3

    # Check tiers: e1 is tier 0, e2 is tier 1, e_recovery is tier -1
    assert graph.edge_tiers["e1"] == 0
    assert graph.edge_tiers["e2"] == 1
    assert graph.edge_tiers["e_recovery"] == -1


def test_graph_dag_cycle_detection():
    """Verify forward edge cycles raise GraphTopologyError."""
    graph = GraphV4("cycle_test")
    v1 = VertexRecordV4(0, "cycle_test", "v1", state=VertexStateV4.DATA_READY)
    v2 = VertexRecordV4(0, "cycle_test", "v2", state=VertexStateV4.TODO)
    graph.add_vertex(v1)
    graph.add_vertex(v2)

    # Mutual cycle between v1 and v2
    e1 = CodeEdgeV4("e1", "v1", "v2")
    e2 = CodeEdgeV4("e2", "v2", "v1")
    graph.add_edge(e1)
    graph.add_edge(e2)

    with pytest.raises(GraphTopologyError, match="Cycle detected in forward edges"):
        graph.validate()


# =====================================================================
# 4. ExecutorV4 Concurrency, DAG Ordering & Reflexive Recovery Tests
# =====================================================================


@pytest.mark.asyncio
async def test_executor_dag_ordering_and_concurrency(mem_store: VertexStoreV4):
    """Test ExecutorV4 runs DAG tiers in sequence while observing concurrency."""
    session_id = "executor_dag_sess"
    graph = GraphV4(session_id)

    # Graph topology:
    # v_root -> v_branch1 -> v_join
    #        -> v_branch2 -> v_join2
    v_root = VertexRecordV4(0, session_id, "v_root", "seed", [VertexAttributeV4.START], VertexStateV4.DATA_READY)
    v_b1 = VertexRecordV4(0, session_id, "v_b1", "", [], VertexStateV4.TODO)
    v_b2 = VertexRecordV4(0, session_id, "v_b2", "", [], VertexStateV4.TODO)
    v_end1 = VertexRecordV4(0, session_id, "v_end1", "", [VertexAttributeV4.END], VertexStateV4.TODO)
    v_end2 = VertexRecordV4(0, session_id, "v_end2", "", [VertexAttributeV4.END], VertexStateV4.TODO)

    for v in (v_root, v_b1, v_b2, v_end1, v_end2):
        graph.add_vertex(v)

    execution_order = []

    def make_fn(name: str):
        def fn(content, settings, staging):
            execution_order.append(name)
            return content + f"_{name}"
        return fn

    e_root_b1 = CodeEdgeV4("e_root_b1", "v_root", "v_b1", script=make_fn("b1"))
    e_root_b2 = CodeEdgeV4("e_root_b2", "v_root", "v_b2", script=make_fn("b2"))
    e_b1_end = CodeEdgeV4("e_b1_end", "v_b1", "v_end1", script=make_fn("end1"))
    e_b2_end = CodeEdgeV4("e_b2_end", "v_b2", "v_end2", script=make_fn("end2"))

    for e in (e_root_b1, e_root_b2, e_b1_end, e_b2_end):
        graph.add_edge(e)

    graph.validate()

    # Verify tiers
    assert graph.edge_tiers["e_root_b1"] == 0
    assert graph.edge_tiers["e_root_b2"] == 0
    assert graph.edge_tiers["e_b1_end"] == 1
    assert graph.edge_tiers["e_b2_end"] == 1

    executor = ExecutorV4(graph=graph, store=mem_store, max_concurrency=2)
    result = await executor.run()

    assert result.success is True
    assert set(result.completed_edges) == {"e_root_b1", "e_root_b2", "e_b1_end", "e_b2_end"}

    # Both tier 0 edges must start and finish before tier 1 edges
    idx_b1 = execution_order.index("b1")
    idx_b2 = execution_order.index("b2")
    idx_end1 = execution_order.index("end1")
    idx_end2 = execution_order.index("end2")

    assert max(idx_b1, idx_b2) < min(idx_end1, idx_end2)

    assert result.vertex_contents["v_end1"] == "seed_b1_end1"
    assert result.vertex_contents["v_end2"] == "seed_b2_end2"


@pytest.mark.asyncio
async def test_reflexive_recovery_loop(mem_store: VertexStoreV4):
    """Test self-loop reflexive edge resets reject state and permits recovery."""
    session_id = "recovery_loop_sess"
    graph = GraphV4(session_id)

    v_start = VertexRecordV4(0, session_id, "v_start", "input_token", [VertexAttributeV4.START], VertexStateV4.DATA_READY)
    v_target = VertexRecordV4(0, session_id, "v_target", "", [VertexAttributeV4.END], VertexStateV4.TODO)

    graph.add_vertex(v_start)
    graph.add_vertex(v_target)

    attempt_counter = 0

    def flaky_script(content, settings, staging):
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter == 1:
            raise ValueError("Transient upstream failure")
        return "success_" + content

    e_forward = CodeEdgeV4("e_forward", "v_start", "v_target", script=flaky_script)
    e_reflexive = ReflexiveEdgeV4("e_reflexive", "v_target", max_retries=3)

    graph.add_edge(e_forward)
    graph.add_edge(e_reflexive)
    graph.validate()

    executor = ExecutorV4(graph=graph, store=mem_store, max_concurrency=1)
    result = await executor.run()

    assert result.success is True
    assert result.completed_edges == ["e_reflexive", "e_forward"]
    assert result.vertex_contents["v_target"] == "success_input_token"
    assert result.vertex_states["v_target"] == VertexStateV4.DATA_READY.value

    # Verify staging records
    staged_reset = mem_store.get_latest_staged_for_vertex(session_id, "v_target", key="reflexive_reset")
    assert staged_reset is not None
    assert "Reset vertex to 'todo urgent'" in staged_reset.value


@pytest.mark.asyncio
async def test_reflexive_retry_limit_circuit_breaker(mem_store: VertexStoreV4):
    """When retries exceed max_retries, target vertex is locked to forbidden."""
    session_id = "circuit_breaker_sess"
    graph = GraphV4(session_id)

    v_start = VertexRecordV4(0, session_id, "v_start", "permanent_failure", [VertexAttributeV4.START], VertexStateV4.DATA_READY)
    v_target = VertexRecordV4(0, session_id, "v_target", "", [VertexAttributeV4.END], VertexStateV4.TODO)

    graph.add_vertex(v_start)
    graph.add_vertex(v_target)

    def always_fail(content, settings, staging):
        raise RuntimeError("Fatal unrecoverable error")

    e_forward = CodeEdgeV4("e_forward", "v_start", "v_target", script=always_fail)
    e_reflexive = ReflexiveEdgeV4("e_reflexive", "v_target", max_retries=2)

    graph.add_edge(e_forward)
    graph.add_edge(e_reflexive)
    graph.validate()

    executor = ExecutorV4(graph=graph, store=mem_store, max_concurrency=1)
    result = await executor.run()

    # Workflow must terminate gracefully and fail
    assert result.success is False
    target_v = mem_store.get_vertex(session_id, "v_target")
    assert target_v.state == VertexStateV4.FORBIDDEN.value

    # Verify staging logged retry exhaustion
    staged_exhausted = mem_store.get_latest_staged_for_vertex(session_id, "v_target", key="retry_exhausted")
    assert staged_exhausted is not None
    assert "Exceeded max retries: 2" in staged_exhausted.value


def test_reflexive_edge_standalone_cli_execution(tmp_path: Path):
    """Test running a reflexive recovery edge via CLI without mandatory --input."""
    db_file = tmp_path / "reflexive_cli.db"
    store = VertexStoreV4(str(db_file))
    session_id = "refl_cli_session"
    store.save_vertex(session_id, "node_err", "faulty payload", state=VertexStateV4.REJECT, processed_count=0)
    store.close()

    cmd = [
        sys.executable,
        "-m",
        "framework.edge_v4",
        "--db",
        str(db_file),
        "--session",
        session_id,
        "--edge-id",
        "cli_refl_1",
        "--type",
        "reflexive",
        "--output",
        "node_err",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"Reflexive CLI failed: {proc.stderr}"

    output_data = json.loads(proc.stdout)
    assert output_data["success"] is True

    verify_store = VertexStoreV4(str(db_file))
    node_v = verify_store.get_vertex(session_id, "node_err")
    assert node_v.state == VertexStateV4.TODO_URGENT.value
    assert node_v.processed_count == 1
    verify_store.close()


@pytest.mark.asyncio
async def test_deep_recursive_subgraph_execution(tmp_path: Path):
    """Test deep recursive subgraphs where child subgraph itself invokes a grandchild subgraph."""
    from framework.server_v4 import SessionGraphManagerV4
    from framework.sse_executor_v4 import SSEExecutorV4

    # 1. Grandchild: receives input, appends _L2
    l2_dir = tmp_path / "l2_graph"
    l2_dir.mkdir(parents=True)
    (l2_dir / "l2_in.json").write_text(json.dumps({"name": "l2_start", "content": "", "attributes": ["start"], "state": "idle"}))
    (l2_dir / "l2_out.json").write_text(json.dumps({"name": "l2_end", "content": "", "attributes": ["end"], "state": "todo"}))
    l2_script = l2_dir / "l2_trans.py"
    l2_script.write_text("def process(data, settings=None, staging=None):\n    return f'{data}->L2'\n")
    (l2_dir / "l2_edge.json").write_text(json.dumps({
        "id": "e_l2", "type": "code", "input_vertex": "l2_start", "output_vertex": "l2_end", "script": f"{l2_script}:process"
    }))
    l2_manifest = l2_dir / "graph.json"
    l2_manifest.write_text(json.dumps({
        "version": "4.0", "session_id": "l2_sess", "metadata": {"name": "L2"},
        "vertices": ["l2_in.json", "l2_out.json"], "edges": ["l2_edge.json"]
    }))

    # 2. Child (L1): contains a subgraph vertex pointing to L2
    l1_dir = tmp_path / "l1_graph"
    l1_dir.mkdir(parents=True)
    (l1_dir / "l1_in.json").write_text(json.dumps({"name": "l1_start", "content": "", "attributes": ["start"], "state": "idle"}))
    (l1_dir / "l1_box.json").write_text(json.dumps({
        "name": "l1_subbox", "content": json.dumps({"subgraph_manifest": str(l2_manifest)}),
        "attributes": ["subgraph"], "state": "todo"
    }))
    (l1_dir / "l1_out.json").write_text(json.dumps({"name": "l1_end", "content": "", "attributes": ["end"], "state": "todo"}))
    (l1_dir / "e_l1_in.json").write_text(json.dumps({
        "id": "e_l1_in", "type": "code", "input_vertex": "l1_start", "output_vertex": "l1_subbox"
    }))
    (l1_dir / "e_l1_out.json").write_text(json.dumps({
        "id": "e_l1_out", "type": "code", "input_vertex": "l1_subbox", "output_vertex": "l1_end"
    }))
    l1_manifest = l1_dir / "graph.json"
    l1_manifest.write_text(json.dumps({
        "version": "4.0", "session_id": "l1_sess", "metadata": {"name": "L1"},
        "vertices": ["l1_in.json", "l1_box.json", "l1_out.json"], "edges": ["e_l1_in.json", "e_l1_out.json"]
    }))

    # 3. Parent (Root): Root -> L1Box -> Sink
    parent_sess = "parent_recursive_sess"
    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    parent_graph = manager.get_or_create_graph(parent_sess)

    parent_graph.add_vertex(VertexRecordV4(0, parent_sess, "root", "TOP_INPUT", [VertexAttributeV4.START.value], VertexStateV4.DATA_READY.value))
    parent_graph.add_vertex(VertexRecordV4(0, parent_sess, "l1_box", json.dumps({"subgraph_manifest": str(l1_manifest)}), [VertexAttributeV4.SUBGRAPH.value], VertexStateV4.TODO.value))
    parent_graph.add_vertex(VertexRecordV4(0, parent_sess, "sink", "", [VertexAttributeV4.END.value], VertexStateV4.TODO.value))
    parent_graph.add_edge(CodeEdgeV4("e_to_l1", "root", "l1_box"))
    parent_graph.add_edge(CodeEdgeV4("e_from_l1", "l1_box", "sink"))

    # 4. Execute via SSEExecutor
    executor = SSEExecutorV4(manager=manager, store=store)
    res = await executor.execute_harness_call(session_id=parent_sess)
    info = json.loads(res["function"]["arguments"])["info"]
    assert info["success"] is True

    sink_v = store.get_vertex(parent_sess, "sink")
    assert sink_v is not None
    assert sink_v.content == "TOP_INPUT->L2"
    assert sink_v.state == VertexStateV4.DATA_READY.value
