"""Tests for V4 Subgraph execution and Discrete JSON / Directory loading."""

import asyncio
import json
from pathlib import Path
import pytest

from framework.graph_v4 import DiscreteGraphLoaderV4, GraphV4
from framework.server_v4 import SessionGraphManagerV4
from framework.sse_executor_v4 import SSEExecutorV4
from framework.vertex_v4 import VertexStateV4, VertexStoreV4


def test_add_vertex_and_edge_from_file_path(tmp_path: Path):
    """Verify add_vertex and add_edge with direct JSON file paths."""
    v_file = tmp_path / "v1.json"
    v_file.write_text(json.dumps({
        "name": "node_a",
        "content": "initial data",
        "attributes": ["start"],
        "state": "data_ready",
    }))

    v2_file = tmp_path / "v2.json"
    v2_file.write_text(json.dumps({
        "name": "node_b",
        "content": "",
        "attributes": ["end"],
        "state": "todo",
    }))

    e_file = tmp_path / "e1.json"
    e_file.write_text(json.dumps({
        "id": "edge_ab",
        "type": "code",
        "input_vertex": "node_a",
        "output_vertex": "node_b",
    }))

    graph = GraphV4(session_id="test_file_loading")
    v1 = graph.add_vertex(v_file)
    v2 = graph.add_vertex(v2_file)
    e = graph.add_edge(e_file)

    assert v1.name == "node_a"
    assert v1.state == VertexStateV4.DATA_READY.value
    assert v2.name == "node_b"
    assert "edge_ab" in graph.edges
    assert graph.edges["edge_ab"].input_vertex == "node_a"
    assert graph.edges["edge_ab"].output_vertex == "node_b"


def test_add_vertex_and_edge_from_directory(tmp_path: Path):
    """Verify add_vertex and add_edge when passed directory paths."""
    v_dir = tmp_path / "vertices"
    v_dir.mkdir()
    (v_dir / "v1.json").write_text(json.dumps({"name": "v_alpha", "content": "1"}))
    (v_dir / "v2.json").write_text(json.dumps({"name": "v_beta", "content": "2"}))

    e_dir = tmp_path / "edges"
    e_dir.mkdir()
    (e_dir / "e1.json").write_text(json.dumps({
        "id": "e_alpha_beta",
        "input_vertex": "v_alpha",
        "output_vertex": "v_beta",
    }))

    graph = GraphV4(session_id="test_dir_loading")
    graph.add_vertex(v_dir)
    graph.add_edge(e_dir)

    assert "v_alpha" in graph.vertices
    assert "v_beta" in graph.vertices
    assert "e_alpha_beta" in graph.edges


def test_graph_from_directory_structured(tmp_path: Path):
    """Verify GraphV4.from_directory with vertices/ and edges/ subdirectories."""
    base_dir = tmp_path / "structured_graph"
    (base_dir / "vertices").mkdir(parents=True)
    (base_dir / "edges").mkdir(parents=True)

    (base_dir / "vertices" / "v1.json").write_text(json.dumps({"name": "root"}))
    (base_dir / "vertices" / "v2.json").write_text(json.dumps({"name": "leaf"}))
    (base_dir / "edges" / "e1.json").write_text(json.dumps({
        "id": "e_root_leaf",
        "input_vertex": "root",
        "output_vertex": "leaf",
    }))

    graph = GraphV4.from_directory(base_dir, session_id="structured_sess")
    assert "root" in graph.vertices
    assert "leaf" in graph.vertices
    assert "e_root_leaf" in graph.edges


def test_graph_from_directory_flat(tmp_path: Path):
    """Verify GraphV4.from_directory with flat JSON files."""
    flat_dir = tmp_path / "flat_graph"
    flat_dir.mkdir()

    (flat_dir / "v_input.json").write_text(json.dumps({"name": "in_node"}))
    (flat_dir / "v_output.json").write_text(json.dumps({"name": "out_node"}))
    (flat_dir / "e_connect.json").write_text(json.dumps({
        "id": "e_conn",
        "type": "code",
        "input_vertex": "in_node",
        "output_vertex": "out_node",
    }))

    graph = GraphV4.from_directory(flat_dir)
    assert "in_node" in graph.vertices
    assert "out_node" in graph.vertices
    assert "e_conn" in graph.edges


def test_file_not_found_errors(tmp_path: Path):
    """Verify appropriate FileNotFoundError when file or directory is missing."""
    graph = GraphV4(session_id="test_missing")
    with pytest.raises(FileNotFoundError):
        graph.add_vertex(tmp_path / "non_existent.json")

    with pytest.raises(FileNotFoundError):
        graph.add_edge(tmp_path / "non_existent_edge.json")

    with pytest.raises(FileNotFoundError):
        GraphV4.from_directory(tmp_path / "non_existent_dir")


@pytest.mark.asyncio
async def test_subgraph_discrete_paths_execution_e2e():
    """Verify end-to-end execution of the subgraph_v4 example using discrete paths."""
    demo_dir = Path(__file__).resolve().parent.parent / "examples" / "subgraph_v4"
    assert (demo_dir / "parent_in.json").exists()

    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    session_id = "test_subgraph_e2e_session"

    graph = manager.get_or_create_graph(session_id)
    graph.add_vertex(demo_dir / "parent_in.json")
    graph.add_vertex(demo_dir / "parent_subgraph.json")
    graph.add_vertex(demo_dir / "parent_out.json")
    graph.add_edge(demo_dir / "e_start_to_subgraph.json")
    graph.add_edge(demo_dir / "e_subgraph_to_output.json")

    DiscreteGraphLoaderV4.populate_store(graph, store)

    executor = SSEExecutorV4(manager=manager, store=store)
    result = await executor.execute_harness_call(session_id=session_id)
    info = json.loads(result["function"]["arguments"])["info"]

    assert info["success"] is True
    assert "e_in_to_subgraph" in info["completed_edges"]
    assert "bridge_subgraph_enrichment_box" in info["completed_edges"]
    assert "e_subgraph_to_out" in info["completed_edges"]

    out_content = json.loads(info["vertex_contents"]["final_output"])
    assert out_content["processed_text"] == "Hello Antigravity Nested Subgraph Processing"
    assert out_content["word_count"] == 5
    assert out_content["status"] == "enriched_by_child_subgraph"


@pytest.mark.asyncio
async def test_subgraph_manifest_execution_e2e():
    """Verify end-to-end execution of the parent manifest with child subgraph."""
    manifest_path = Path(__file__).resolve().parent.parent / "examples" / "subgraph_v4" / "parent_graph.json"
    assert manifest_path.exists()

    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)

    executor = SSEExecutorV4(manager=manager, store=store, default_manifest=manifest_path)
    result = await executor.execute_harness_call()
    info = json.loads(result["function"]["arguments"])["info"]

    assert info["success"] is True
    assert "bridge_subgraph_enrichment_box" in info["completed_edges"]
    out_content = json.loads(info["vertex_contents"]["final_output"])
    assert out_content["word_count"] == 5
