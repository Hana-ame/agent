"""Layer 3: Graph Topology, State Coloring, and Lifecycle Test Suite.

Validates:
1. Three-color state coloring DFS cycle detection and topological sorting.
2. Cycle path extraction and diagnostic formatting in GraphTopologyError.
3. Reflexive self-loop exemption from forward DAG cycle checks.
4. Orphan vertex lifecycle: default-inactive state, explicit and declarative activation.
5. Topological tier calculation (compute_dag_tiers) across various topologies.
6. Dynamic graph CRUD mutations and cascade cleanup.
7. DiscreteGraphLoaderV4 manifest loading and SQLite store hydration roundtrip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from framework.edge_v4 import CodeEdgeV4, ReflexiveEdgeV4
from framework.executor_v4 import ExecutorV4
from framework.graph_v4 import (
    DiscreteGraphLoaderV4,
    GraphTopologyError,
    GraphV4,
    NodeColor,
)
from framework.vertex_v4 import (
    TraversalColor,
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)


@pytest.fixture
def mem_store() -> VertexStoreV4:
    """Fixture providing an isolated in-memory SQLite store."""
    store = VertexStoreV4(":memory:")
    yield store
    store.close()


# =====================================================================
# 1. Three-Color State Coloring DFS & Cycle Detection
# =====================================================================


def test_node_color_enum_values():
    """Verify NodeColor is the dedicated traversal-colour enum, not a lifecycle state."""
    assert NodeColor is TraversalColor
    assert TraversalColor.WHITE == "white"
    assert TraversalColor.GRAY == "gray"
    assert TraversalColor.BLACK == "black"
    assert NodeColor.WHITE == 0
    assert NodeColor.GRAY == 1
    assert NodeColor.BLACK == 2
    # Traversal colours must not be lifecycle states.
    for color in ("white", "gray", "black"):
        with pytest.raises(ValueError):
            VertexStateV4(color)


def test_state_coloring_acyclic_dag_all_turn_black():
    """Verify DFS records colours in node_states only, leaving vertex.state untouched."""
    graph = GraphV4("acyclic_test")
    v1 = VertexRecordV4(0, "acyclic_test", "v1")
    v2 = VertexRecordV4(0, "acyclic_test", "v2")
    v3 = VertexRecordV4(0, "acyclic_test", "v3")
    graph.add_vertex(v1)
    graph.add_vertex(v2)
    graph.add_vertex(v3)

    graph.add_edge(CodeEdgeV4("e1", "v1", "v2"))
    graph.add_edge(CodeEdgeV4("e2", "v2", "v3"))

    topo_order = graph.detect_cycles_and_order()
    assert topo_order == ["v1", "v2", "v3"]
    assert graph.node_states["v1"] == TraversalColor.BLACK
    assert graph.node_states["v2"] == TraversalColor.BLACK
    assert graph.node_states["v3"] == TraversalColor.BLACK
    assert graph.node_colors["v1"] == NodeColor.BLACK
    assert graph.node_colors["v2"] == NodeColor.BLACK
    assert graph.node_colors["v3"] == NodeColor.BLACK
    # Lifecycle state is preserved: validation must never persist DFS colours.
    assert v1.state == VertexStateV4.IDLE.value
    assert v2.state == VertexStateV4.IDLE.value
    assert v3.state == VertexStateV4.IDLE.value


def test_state_coloring_direct_mutual_cycle():
    """Verify direct mutual cycle (v1 -> v2 -> v1) raises GraphTopologyError with path."""
    graph = GraphV4("mutual_cycle")
    v1 = VertexRecordV4(0, "mutual_cycle", "v1")
    v2 = VertexRecordV4(0, "mutual_cycle", "v2")
    graph.add_vertex(v1)
    graph.add_vertex(v2)

    graph.add_edge(CodeEdgeV4("e1", "v1", "v2"))
    graph.add_edge(CodeEdgeV4("e2", "v2", "v1"))

    with pytest.raises(GraphTopologyError) as exc_info:
        graph.validate()

    msg = str(exc_info.value)
    assert "Cycle detected in forward edges" in msg
    assert ("v1 -> v2 -> v1" in msg) or ("v2 -> v1 -> v2" in msg)


def test_state_coloring_deep_cycle_path_pinpointed():
    """Verify cycle sub-path is isolated when cycle starts downstream (v1 -> v2 -> v3 -> v4 -> v2)."""
    graph = GraphV4("deep_cycle")
    for name in ["v1", "v2", "v3", "v4"]:
        graph.add_vertex(VertexRecordV4(0, "deep_cycle", name))

    graph.add_edge(CodeEdgeV4("e_entry", "v1", "v2"))
    graph.add_edge(CodeEdgeV4("e_cycle1", "v2", "v3"))
    graph.add_edge(CodeEdgeV4("e_cycle2", "v3", "v4"))
    graph.add_edge(CodeEdgeV4("e_back", "v4", "v2"))

    with pytest.raises(GraphTopologyError) as exc_info:
        graph.detect_cycles_and_order()

    msg = str(exc_info.value)
    assert "Cycle detected in forward edges" in msg
    assert "v2 -> v3 -> v4 -> v2" in msg


def test_state_coloring_reflexive_self_loops_exempt():
    """Verify reflexive recovery self-loops (v1 -> v1) are excluded from DAG cycle detection."""
    graph = GraphV4("reflexive_graph")
    v1 = VertexRecordV4(0, "reflexive_graph", "v1")
    v2 = VertexRecordV4(0, "reflexive_graph", "v2")
    graph.add_vertex(v1)
    graph.add_vertex(v2)

    # Forward edge
    graph.add_edge(CodeEdgeV4("e_forward", "v1", "v2"))
    # Reflexive self-loop
    graph.add_edge(ReflexiveEdgeV4("e_recovery", "v1"))

    # Must validate without error
    graph.validate()
    assert graph.edge_tiers["e_forward"] == 0
    assert graph.edge_tiers["e_recovery"] == -1


def test_missing_vertex_reference_raises():
    """Verify edges referencing non-existent vertices raise GraphTopologyError."""
    graph = GraphV4("broken_links")
    graph.add_vertex(VertexRecordV4(0, "broken_links", "v1"))

    # Missing output vertex
    graph.add_edge(CodeEdgeV4("e_broken_out", "v1", "non_existent"))
    with pytest.raises(GraphTopologyError, match="references non-existent output vertex"):
        graph.validate()

    graph.edges.clear()

    # Missing input vertex
    graph.add_edge(CodeEdgeV4("e_broken_in", "non_existent", "v1"))
    with pytest.raises(GraphTopologyError, match="references non-existent input vertex"):
        graph.validate()


# =====================================================================
# 2. Orphan Vertices & Activation Management
# =====================================================================


def test_orphan_vertex_detection():
    """Verify orphan vertices with degree 0 are recognized accurately."""
    graph = GraphV4("orphan_test")
    v_conn1 = VertexRecordV4(0, "orphan_test", "conn1")
    v_conn2 = VertexRecordV4(0, "orphan_test", "conn2")
    v_orphan1 = VertexRecordV4(0, "orphan_test", "orphan1")
    v_orphan2 = VertexRecordV4(0, "orphan_test", "orphan2")

    graph.add_vertex(v_conn1)
    graph.add_vertex(v_conn2)
    graph.add_vertex(v_orphan1)
    graph.add_vertex(v_orphan2)

    graph.add_edge(CodeEdgeV4("e1", "conn1", "conn2"))

    assert not graph.is_orphan("conn1")
    assert not graph.is_orphan("conn2")
    assert graph.is_orphan("orphan1")
    assert graph.is_orphan("orphan2")
    assert set(graph.get_orphans()) == {"orphan1", "orphan2"}


def test_orphan_default_inactive_and_activation():
    """Verify orphan vertices are inactive by default, and can be activated/deactivated on demand."""
    graph = GraphV4("activation_test")
    v_active = VertexRecordV4(0, "activation_test", "connected_node")
    v_target = VertexRecordV4(0, "activation_test", "target_node")
    v_orphan = VertexRecordV4(0, "activation_test", "orphan_node")

    graph.add_vertex(v_active)
    graph.add_vertex(v_target)
    graph.add_vertex(v_orphan)
    graph.add_edge(CodeEdgeV4("e1", "connected_node", "target_node"))

    # Connected nodes are active by default
    assert graph.is_vertex_active("connected_node")
    assert graph.is_vertex_active("target_node")

    # Orphan node is inactive by default
    assert not graph.is_vertex_active("orphan_node")
    assert "orphan_node" in graph.get_inactive_vertices()

    # Explicitly activate orphan
    graph.activate_vertex("orphan_node")
    assert graph.is_vertex_active("orphan_node")
    assert "orphan_node" in graph.get_active_vertices()
    assert "orphan_node" not in graph.get_inactive_vertices()

    # Deactivate orphan again
    graph.deactivate_vertex("orphan_node")
    assert not graph.is_vertex_active("orphan_node")

    # Activating non-existent vertex raises KeyError
    with pytest.raises(KeyError, match="not found in graph"):
        graph.activate_vertex("unknown_vertex")


def test_declarative_activation_via_attributes():
    """Verify declarative activation/deactivation via VertexAttributeV4.ACTIVE and INACTIVE."""
    graph = GraphV4("decl_test")
    # Orphan with ACTIVE attribute
    v_active_orphan = VertexRecordV4(
        0, "decl_test", "active_orphan", attributes=[VertexAttributeV4.ACTIVE]
    )
    # Connected node with INACTIVE attribute
    v_inactive_conn = VertexRecordV4(
        0, "decl_test", "inactive_conn", attributes=[VertexAttributeV4.INACTIVE]
    )
    v_conn2 = VertexRecordV4(0, "decl_test", "conn2")

    graph.add_vertex(v_active_orphan)
    graph.add_vertex(v_inactive_conn)
    graph.add_vertex(v_conn2)
    graph.add_edge(CodeEdgeV4("e1", "inactive_conn", "conn2"))

    # active_orphan is an orphan, but active due to attribute
    assert graph.is_orphan("active_orphan")
    assert graph.is_vertex_active("active_orphan")

    # inactive_conn is connected, but inactive due to attribute
    assert not graph.is_orphan("inactive_conn")
    assert not graph.is_vertex_active("inactive_conn")

    # conn2 is connected with default state
    assert graph.is_vertex_active("conn2")


@pytest.mark.asyncio
async def test_executor_ignores_default_inactive_orphans(mem_store: VertexStoreV4):
    """Verify ExecutorV4 runs to success without being blocked by inactive orphan vertices."""
    session_id = "orphan_executor_test"
    graph = GraphV4(session_id)

    v_start = VertexRecordV4(
        0, session_id, "start", content="initial",
        attributes=[VertexAttributeV4.START], state=VertexStateV4.DATA_READY.value
    )
    v_end = VertexRecordV4(
        0, session_id, "end", content="",
        attributes=[VertexAttributeV4.END], state=VertexStateV4.TODO.value
    )
    # Inactive orphan in IDLE or TODO state
    v_orphan = VertexRecordV4(
        0, session_id, "orphan", content="standalone",
        state=VertexStateV4.TODO.value
    )

    graph.add_vertex(v_start)
    graph.add_vertex(v_end)
    graph.add_vertex(v_orphan)

    # Edge from start to end
    edge = CodeEdgeV4("e_work", "start", "end", script=None)
    graph.add_edge(edge)
    graph.validate()

    assert graph.is_orphan("orphan")
    assert not graph.is_vertex_active("orphan")

    DiscreteGraphLoaderV4.populate_store(graph, mem_store)

    executor = ExecutorV4(graph=graph, store=mem_store)
    res = await executor.run()

    # Graph must finish successfully without deadlocking on the orphan
    assert res.success
    end_rec = mem_store.get_vertex(session_id, "end")
    assert end_rec.state == VertexStateV4.DATA_READY.value


# =====================================================================
# 3. Topological Tier Calculation (compute_dag_tiers)
# =====================================================================


def test_dag_tiers_diamond_topology():
    """Verify edge tiers on a classic diamond DAG (A -> B, A -> C, B -> D, C -> D)."""
    graph = GraphV4("diamond")
    for name in ["A", "B", "C", "D"]:
        attrs = [VertexAttributeV4.START] if name == "A" else []
        graph.add_vertex(VertexRecordV4(0, "diamond", name, attributes=attrs))

    graph.add_edge(CodeEdgeV4("e_AB", "A", "B"))
    graph.add_edge(CodeEdgeV4("e_AC", "A", "C"))
    graph.add_edge(CodeEdgeV4("e_BD", "B", "D"))
    graph.add_edge(CodeEdgeV4("e_CD", "C", "D"))

    tiers = graph.compute_dag_tiers()
    assert tiers["e_AB"] == 0
    assert tiers["e_AC"] == 0
    assert tiers["e_BD"] == 1
    assert tiers["e_CD"] == 1

    assert graph.node_tiers["A"] == 0
    assert graph.node_tiers["B"] == 1
    assert graph.node_tiers["C"] == 1
    assert graph.node_tiers["D"] == 2


def test_dag_tiers_with_inactive_and_active_orphans():
    """Verify tier assignment distinguishes active vs inactive orphan nodes."""
    graph = GraphV4("orphan_tiers")
    v_root = VertexRecordV4(0, "orphan_tiers", "root", attributes=[VertexAttributeV4.START])
    v_leaf = VertexRecordV4(0, "orphan_tiers", "leaf")
    v_inactive_orphan = VertexRecordV4(0, "orphan_tiers", "inactive_orphan")
    v_active_orphan = VertexRecordV4(0, "orphan_tiers", "active_orphan")

    graph.add_vertex(v_root)
    graph.add_vertex(v_leaf)
    graph.add_vertex(v_inactive_orphan)
    graph.add_vertex(v_active_orphan)

    graph.add_edge(CodeEdgeV4("e1", "root", "leaf"))
    graph.activate_vertex("active_orphan")

    graph.compute_dag_tiers()

    # Active root is tier 0
    assert graph.node_tiers["root"] == 0
    assert graph.node_tiers["leaf"] == 1

    # Inactive orphan receives -1
    assert graph.node_tiers["inactive_orphan"] == -1
    # Active orphan receives tier 0 as standalone root
    assert graph.node_tiers["active_orphan"] == 0


# =====================================================================
# 4. Dynamic Graph CRUD & Cascade Mutation
# =====================================================================


def test_graph_crud_and_cascade_deletion():
    """Verify dynamic deletion of vertices cascades to connected edges and invalidates tiers."""
    graph = GraphV4("crud_test")
    v1 = VertexRecordV4(0, "crud_test", "v1")
    v2 = VertexRecordV4(0, "crud_test", "v2")
    v3 = VertexRecordV4(0, "crud_test", "v3")

    graph.add_vertex(v1)
    graph.add_vertex(v2)
    graph.add_vertex(v3)

    graph.add_edge(CodeEdgeV4("e1", "v1", "v2"))
    graph.add_edge(CodeEdgeV4("e2", "v2", "v3"))
    graph.validate()

    assert len(graph.edges) == 2
    assert "e1" in graph.edge_tiers
    assert "e2" in graph.edge_tiers

    # Delete v2: should remove v2 and cascade to e1 and e2
    assert graph.delete_vertex("v2")
    assert graph.get_vertex("v2") is None
    assert graph.get_edge("e1") is None
    assert graph.get_edge("e2") is None
    assert "e1" not in graph.edge_tiers
    assert "e2" not in graph.edge_tiers

    # Deleting non-existent returns False
    assert not graph.delete_vertex("v2")
    assert not graph.delete_edge("e1")


# =====================================================================
# 5. DiscreteGraphLoaderV4 & SQLite Store Roundtrip
# =====================================================================


def test_discrete_loader_with_orphans_and_store_roundtrip(tmp_path: Path, mem_store: VertexStoreV4):
    """Verify DiscreteGraphLoaderV4 parses discrete components and hydrates store isomorphic to memory."""
    graph_dir = tmp_path / "discrete_graph"
    graph_dir.mkdir()

    # Write discrete vertices
    v1_data = {
        "name": "in_node",
        "content": "payload_1",
        "attributes": ["start"],
        "state": "data ready",
    }
    v2_data = {
        "name": "out_node",
        "content": "",
        "attributes": ["end"],
        "state": "todo",
    }
    v3_orphan_data = {
        "name": "dormant_orphan",
        "content": "extra_data",
        "attributes": ["json"],
        "state": "idle",
    }

    (graph_dir / "v1.json").write_text(json.dumps(v1_data), encoding="utf-8")
    (graph_dir / "v2.json").write_text(json.dumps(v2_data), encoding="utf-8")
    (graph_dir / "v3.json").write_text(json.dumps(v3_orphan_data), encoding="utf-8")

    # Write discrete edge
    e1_data: Dict[str, Any] = {
        "id": "e_flow",
        "type": "code",
        "input_vertex": "in_node",
        "output_vertex": "out_node",
        "priority": 5,
    }
    (graph_dir / "e1.json").write_text(json.dumps(e1_data), encoding="utf-8")

    # Write master manifest
    manifest_data = {
        "session_id": "sess_discrete",
        "metadata": {"name": "test_modular_workflow"},
        "vertices": ["v1.json", "v2.json", "v3.json"],
        "edges": ["e1.json"],
    }
    manifest_path = graph_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Load via DiscreteGraphLoaderV4
    graph = DiscreteGraphLoaderV4.load_from_manifest(manifest_path)
    assert len(graph.vertices) == 3
    assert len(graph.edges) == 1
    assert graph.is_orphan("dormant_orphan")
    assert not graph.is_vertex_active("dormant_orphan")
    assert graph.is_vertex_active("in_node")

    # Populate SQLite store
    DiscreteGraphLoaderV4.populate_store(graph, mem_store)

    # Re-hydrate from SQLite store
    hydrated = GraphV4.load_from_store(mem_store, "sess_discrete", name="test_modular_workflow")
    assert len(hydrated.vertices) == 3
    assert len(hydrated.edges) == 1
    assert hydrated.is_orphan("dormant_orphan")
    assert not hydrated.is_vertex_active("dormant_orphan")
    assert hydrated.edge_tiers["e_flow"] == 0


# =====================================================================
# 6. Dynamic Graph Mutation & Reconnection
# =====================================================================


def test_dynamic_reconnect_edge():
    """Verify dynamic reconnection of an edge to new endpoints."""
    graph = GraphV4("reconnect_test")
    v1 = VertexRecordV4(0, "reconnect_test", "v1")
    v2 = VertexRecordV4(0, "reconnect_test", "v2")
    v3 = VertexRecordV4(0, "reconnect_test", "v3")
    graph.add_vertex(v1)
    graph.add_vertex(v2)
    graph.add_vertex(v3)

    graph.add_edge(CodeEdgeV4("e1", "v1", "v2"))
    graph.validate()
    assert graph.get_edge("e1").output_vertex == "v2"

    # Reconnect e1 to point to v3 instead
    assert graph.reconnect_edge("e1", new_output_vertex="v3")
    assert graph.get_edge("e1").output_vertex == "v3"
    graph.validate()
    assert graph.edge_tiers["e1"] == 0


def test_dynamic_replace_vertex_with_edge_transfer():
    """Verify dynamic replacement of a vertex with automatic edge rewiring."""
    graph = GraphV4("replace_test")
    v1 = VertexRecordV4(0, "replace_test", "v1")
    v_old = VertexRecordV4(0, "replace_test", "v_old")
    v3 = VertexRecordV4(0, "replace_test", "v3")
    graph.add_vertex(v1)
    graph.add_vertex(v_old)
    graph.add_vertex(v3)

    graph.add_edge(CodeEdgeV4("e_in", "v1", "v_old"))
    graph.add_edge(CodeEdgeV4("e_out", "v_old", "v3"))
    graph.validate()

    v_new = VertexRecordV4(0, "replace_test", "v_new", content="upgraded")
    assert graph.replace_vertex("v_old", v_new, transfer_edges=True)

    assert graph.get_vertex("v_old") is None
    assert graph.get_vertex("v_new") is not None
    assert graph.get_edge("e_in").output_vertex == "v_new"
    assert graph.get_edge("e_out").input_vertex == "v_new"
    graph.validate()


# =====================================================================
# 7. Transitive Reachability & Downstream Reset for Re-entry
# =====================================================================


def test_get_downstream_and_upstream_transitive_closure():
    """Verify downstream and upstream reachability analysis in forward DAG."""
    graph = GraphV4("reachability")
    for name in ["A", "B", "C", "D", "E"]:
        graph.add_vertex(VertexRecordV4(0, "reachability", name))

    # A -> B, A -> C, B -> D, C -> D, D -> E
    graph.add_edge(CodeEdgeV4("e1", "A", "B"))
    graph.add_edge(CodeEdgeV4("e2", "A", "C"))
    graph.add_edge(CodeEdgeV4("e3", "B", "D"))
    graph.add_edge(CodeEdgeV4("e4", "C", "D"))
    graph.add_edge(CodeEdgeV4("e5", "D", "E"))

    # Downstream from B: should be D, E
    down_b = graph.get_downstream_vertices("B", include_self=False)
    assert set(down_b) == {"D", "E"}

    # Downstream from A: should be B, C, D, E
    down_a = graph.get_downstream_vertices("A", include_self=False)
    assert set(down_a) == {"B", "C", "D", "E"}

    # Upstream to D: should be B, C, A
    up_d = graph.get_upstream_vertices("D", include_self=False)
    assert set(up_d) == {"A", "B", "C"}


def test_vertex_reentry_resets_downstream_affected_vertices():
    """Verify reenter_vertex sets target to DATA_READY and resets downstream to TODO."""
    graph = GraphV4("reenter_test")
    v1 = VertexRecordV4(0, "reenter_test", "v1", content="step1", state=VertexStateV4.DATA_READY.value)
    v2 = VertexRecordV4(0, "reenter_test", "v2", content="step2", state=VertexStateV4.DATA_READY.value)
    v3 = VertexRecordV4(0, "reenter_test", "v3", content="step3", state=VertexStateV4.DATA_READY.value)
    graph.add_vertex(v1)
    graph.add_vertex(v2)
    graph.add_vertex(v3)

    graph.add_edge(CodeEdgeV4("e1", "v1", "v2"))
    graph.add_edge(CodeEdgeV4("e2", "v2", "v3"))

    # Re-enter v2 with new content
    affected = graph.reenter_vertex("v2", new_content="reentered_step2", reset_state=VertexStateV4.TODO.value)

    # v2 is now DATA_READY with new content
    assert graph.get_vertex("v2").state == VertexStateV4.DATA_READY.value
    assert graph.get_vertex("v2").content == "reentered_step2"

    # Downstream v3 is reset to TODO
    assert affected == ["v3"]
    assert graph.get_vertex("v3").state == VertexStateV4.TODO.value

    # Upstream v1 is unaffected
    assert graph.get_vertex("v1").state == VertexStateV4.DATA_READY.value


@pytest.mark.asyncio
async def test_executor_reentry_run_lifecycle(mem_store: VertexStoreV4):
    """Verify full execution lifecycle: initial run completes, node is re-entered, and downstream re-executes."""
    session_id = "lifecycle_reentry"
    graph = GraphV4(session_id)

    v_a = VertexRecordV4(0, session_id, "A", content="Run1", attributes=[VertexAttributeV4.START], state=VertexStateV4.DATA_READY.value)
    v_b = VertexRecordV4(0, session_id, "B", content="", state=VertexStateV4.TODO.value)
    v_c = VertexRecordV4(0, session_id, "C", content="", attributes=[VertexAttributeV4.END], state=VertexStateV4.TODO.value)

    graph.add_vertex(v_a)
    graph.add_vertex(v_b)
    graph.add_vertex(v_c)

    graph.add_edge(CodeEdgeV4("e1", "A", "B"))
    graph.add_edge(CodeEdgeV4("e2", "B", "C"))
    graph.validate()

    DiscreteGraphLoaderV4.populate_store(graph, mem_store)

    executor = ExecutorV4(graph=graph, store=mem_store)

    # First run completes
    res1 = await executor.run()
    assert res1.success
    assert mem_store.get_vertex(session_id, "C").state == VertexStateV4.DATA_READY.value
    assert mem_store.get_vertex(session_id, "B").state == VertexStateV4.DATA_READY.value

    # Re-enter node A with new content: resets B and C to TODO in memory and store
    affected = graph.reenter_vertex("A", new_content="Run2", store=mem_store, session_id=session_id)
    assert set(affected) == {"B", "C"}
    assert mem_store.get_vertex(session_id, "B").state == VertexStateV4.TODO.value
    assert mem_store.get_vertex(session_id, "C").state == VertexStateV4.TODO.value

    # Second run re-triggers handshakes and flows to completion
    res2 = await executor.run()
    assert res2.success
    assert mem_store.get_vertex(session_id, "A").content == "Run2"
    assert mem_store.get_vertex(session_id, "C").state == VertexStateV4.DATA_READY.value


# =====================================================================
# 8. Subgraph Splicing and Insertion
# =====================================================================


def test_splice_subgraph_inlines_nested_topology():
    """Verify splicing a subgraph in place of an existing vertex with edge rewiring."""
    parent = GraphV4("parent_graph")
    parent.add_vertex(VertexRecordV4(0, "parent_graph", "input_node", attributes=[VertexAttributeV4.START]))
    parent.add_vertex(VertexRecordV4(0, "parent_graph", "sub_box"))
    parent.add_vertex(VertexRecordV4(0, "parent_graph", "output_node", attributes=[VertexAttributeV4.END]))

    parent.add_edge(CodeEdgeV4("e_in", "input_node", "sub_box"))
    parent.add_edge(CodeEdgeV4("e_out", "sub_box", "output_node"))
    parent.validate()

    # Subgraph: s1 -> s2 -> s3
    sub = GraphV4("inner_sub")
    sub.add_vertex(VertexRecordV4(0, "inner_sub", "s1", attributes=[VertexAttributeV4.START]))
    sub.add_vertex(VertexRecordV4(0, "inner_sub", "s2"))
    sub.add_vertex(VertexRecordV4(0, "inner_sub", "s3", attributes=[VertexAttributeV4.END]))
    sub.add_edge(CodeEdgeV4("sub_e1", "s1", "s2"))
    sub.add_edge(CodeEdgeV4("sub_e2", "s2", "s3"))
    sub.validate()

    # Splice sub into sub_box with prefix
    res = parent.splice_subgraph("sub_box", sub, name_prefix="mod")

    assert parent.get_vertex("sub_box") is None
    assert parent.get_vertex("mod_s1") is not None
    assert parent.get_vertex("mod_s2") is not None
    assert parent.get_vertex("mod_s3") is not None

    # Verify edge rewiring
    assert parent.get_edge("e_in").output_vertex == "mod_s1"
    assert parent.get_edge("e_out").input_vertex == "mod_s3"

    # Verify DAG validity and tiers
    parent.validate()
    assert parent.edge_tiers["e_in"] == 0
    assert parent.edge_tiers["mod_sub_e1"] == 1
    assert parent.edge_tiers["mod_sub_e2"] == 2
    assert parent.edge_tiers["e_out"] == 3


@pytest.mark.asyncio
async def test_splice_subgraph_execution_dataflow(mem_store: VertexStoreV4):
    """Verify end-to-end dataflow through a dynamically spliced subgraph."""
    session_id = "splice_exec"
    parent = GraphV4(session_id)
    parent.add_vertex(VertexRecordV4(0, session_id, "root", content="INIT", attributes=[VertexAttributeV4.START], state=VertexStateV4.DATA_READY.value))
    parent.add_vertex(VertexRecordV4(0, session_id, "box", state=VertexStateV4.TODO.value))
    parent.add_vertex(VertexRecordV4(0, session_id, "sink", attributes=[VertexAttributeV4.END], state=VertexStateV4.TODO.value))
    parent.add_edge(CodeEdgeV4("e_to_box", "root", "box"))
    parent.add_edge(CodeEdgeV4("e_from_box", "box", "sink"))

    # Child subgraph: step1 -> step2
    sub = GraphV4("sub")
    sub.add_vertex(VertexRecordV4(0, "sub", "step1", attributes=[VertexAttributeV4.START], state=VertexStateV4.TODO.value))
    sub.add_vertex(VertexRecordV4(0, "sub", "step2", attributes=[VertexAttributeV4.END], state=VertexStateV4.TODO.value))
    sub.add_edge(CodeEdgeV4("e_sub", "step1", "step2"))

    # Splice box
    parent.splice_subgraph("box", sub, name_prefix="sub")

    DiscreteGraphLoaderV4.populate_store(parent, mem_store)

    executor = ExecutorV4(graph=parent, store=mem_store)
    res = await executor.run()

    assert res.success
    sink_rec = mem_store.get_vertex(session_id, "sink")
    assert sink_rec.state == VertexStateV4.DATA_READY.value


def test_insert_subgraph_with_boundary_bindings():
    """Verify insert_subgraph with explicit boundary edge bindings."""
    graph = GraphV4("insert_test")
    graph.add_vertex(VertexRecordV4(0, "insert_test", "entry_node", attributes=[VertexAttributeV4.START]))
    graph.add_vertex(VertexRecordV4(0, "insert_test", "exit_node", attributes=[VertexAttributeV4.END]))

    sub = GraphV4("sub_worker")
    sub.add_vertex(VertexRecordV4(0, "sub_worker", "w1"))
    sub.add_vertex(VertexRecordV4(0, "sub_worker", "w2"))
    sub.add_edge(CodeEdgeV4("w_edge", "w1", "w2"))

    res = graph.insert_subgraph(
        subgraph=sub,
        incoming_bindings={"entry_node": "w1"},
        outgoing_bindings={"w2": "exit_node"},
        name_prefix="p1",
    )

    assert len(res["inserted_vertices"]) == 2
    assert "p1_w1" in graph.vertices
    assert "p1_w2" in graph.vertices
    graph.validate()

    assert graph.edge_tiers["bridge_in_entry_node_p1_w1"] == 0
    assert graph.edge_tiers["p1_w_edge"] == 1
    assert graph.edge_tiers["bridge_out_p1_w2_exit_node"] == 2


def test_session_graph_manager_splice_and_reentry_sqlite_persistence(mem_store: VertexStoreV4):
    """Verify SessionGraphManagerV4 coordinates dynamic splicing, re-entry, and SQLite persistence."""
    from framework.server_v4 import SessionGraphManagerV4

    session_id = "mgr_splice_reentry"
    manager = SessionGraphManagerV4(store=mem_store)

    # Initial graph
    manager.add_or_update_vertex(session_id, "source", content="DATA_V1", attributes=["start"], state="data ready")
    manager.add_or_update_vertex(session_id, "compute_box", state="todo")
    manager.add_or_update_vertex(session_id, "dest", attributes=["end"], state="todo")
    manager.add_or_update_edge(session_id, "e1", "code", "source", "compute_box")
    manager.add_or_update_edge(session_id, "e2", "code", "compute_box", "dest")

    # Subgraph to splice
    sub = GraphV4("inner")
    sub.add_vertex(VertexRecordV4(0, "inner", "in_core", attributes=["start"]))
    sub.add_vertex(VertexRecordV4(0, "inner", "out_core", attributes=["end"]))
    sub.add_edge(CodeEdgeV4("e_core", "in_core", "out_core"))

    # Splice compute_box
    res = manager.splice_subgraph(session_id, "compute_box", sub, name_prefix="sub")
    assert res["spliced_vertex"] == "compute_box"

    # Re-hydrate from SQLite to verify complete DB isomorphism
    hydrated = GraphV4.load_from_store(mem_store, session_id)
    assert "compute_box" not in hydrated.vertices
    assert "sub_in_core" in hydrated.vertices
    assert "sub_out_core" in hydrated.vertices
    assert hydrated.get_edge("e1").output_vertex == "sub_in_core"
    assert hydrated.get_edge("e2").input_vertex == "sub_out_core"

    # Test re-entry through manager
    affected = manager.reenter_vertex(session_id, "source", new_content="DATA_V2", reset_state="todo")
    assert "sub_in_core" in affected
    assert "sub_out_core" in affected
    assert "dest" in affected

    db_dest = mem_store.get_vertex(session_id, "dest")
    assert db_dest.state == VertexStateV4.TODO.value
    db_source = mem_store.get_vertex(session_id, "source")
    assert db_source.content == "DATA_V2"
    assert db_source.state == VertexStateV4.DATA_READY.value


@pytest.mark.asyncio
async def test_permissive_definition_with_runtime_result_resolution(mem_store: VertexStoreV4):
    """Verify graphs with feedback loops can be defined freely, and DAG conclusions emerge at runtime."""
    session_id = "sess_feedback_runtime"

    # Define vertices in store
    v_start = mem_store.save_vertex(session_id, "v_start", "input_data", attributes=["start"], state=VertexStateV4.DATA_READY)
    v_process = mem_store.save_vertex(session_id, "v_process", "", state=VertexStateV4.TODO)
    v_end = mem_store.save_vertex(session_id, "v_end", "", attributes=["end"], state=VertexStateV4.TODO)

    # Graph with forward path and a conditional feedback edge
    graph = GraphV4(session_id)
    graph.add_vertex(v_start)
    graph.add_vertex(v_process)
    graph.add_vertex(v_end)

    graph.add_edge(CodeEdgeV4("e_forward1", "v_start", "v_process"))
    graph.add_edge(CodeEdgeV4("e_forward2", "v_process", "v_end"))
    # Feedback edge (cycle: v_end -> v_process)
    graph.add_edge(CodeEdgeV4("e_feedback", "v_end", "v_process"))

    # Definition time: validate(strict_dag=False) succeeds without forcing static DAG conclusion
    graph.validate(strict_dag=False)
    assert graph.edge_tiers["e_forward1"] == 0

    # Runtime: execute workflow dynamically and observe conclusion from ExecutionResultV4
    executor = ExecutorV4(graph=graph, store=mem_store, timeout=5.0)
    result = await executor.run()

    # Verify execution output inspected after run
    assert result.success is True
    assert "e_forward1" in result.completed_edges
    assert "e_forward2" in result.completed_edges
    assert result.vertex_states["v_end"] == VertexStateV4.DATA_READY.value


def test_loaded_nodes_provenance_and_management():
    """Verify tracking of loaded nodes across memory, loader, and subgraph operations."""
    graph = GraphV4("provenance_sess")
    v1 = VertexRecordV4(0, "provenance_sess", "v1", state=VertexStateV4.DATA_READY)
    graph.add_vertex(v1, source="custom_source", metadata={"tag": "initial"})

    meta = graph.get_loaded_node("v1")
    assert meta is not None
    assert meta["source"] == "custom_source"
    assert meta["metadata"]["tag"] == "initial"

    loaded = graph.list_loaded_nodes()
    assert len(loaded) == 1
    assert loaded[0]["name"] == "v1"


def test_node_and_graph_relationships():
    """Verify inspection of node predecessors, successors, and full graph relationship matrix."""
    graph = GraphV4("rel_test")
    for name in ["r1", "r2", "m1", "s1", "iso"]:
        graph.add_vertex(VertexRecordV4(0, "rel_test", name))

    # r1 -> m1, r2 -> m1, m1 -> s1
    graph.add_edge(CodeEdgeV4("e1", "r1", "m1"))
    graph.add_edge(CodeEdgeV4("e2", "r2", "m1"))
    graph.add_edge(CodeEdgeV4("e3", "m1", "s1"))
    graph.add_edge(ReflexiveEdgeV4("e_ref", "m1"))

    # Direct relations on m1
    assert graph.get_node_predecessors("m1") == ["r1", "r2"]
    assert graph.get_node_successors("m1") == ["s1"]

    m1_rel = graph.get_node_relationships("m1")
    assert m1_rel["in_degree"] == 2
    assert m1_rel["out_degree"] == 1
    assert m1_rel["predecessors"] == ["r1", "r2"]
    assert m1_rel["successors"] == ["s1"]
    assert len(m1_rel["incoming_edges"]) == 2
    assert len(m1_rel["outgoing_edges"]) == 1
    assert len(m1_rel["reflexive_edges"]) == 1
    assert m1_rel["is_orphan"] is False

    # Graph-level relationships
    full_rel = graph.get_graph_relationships()
    assert set(full_rel["roots"]) == {"r1", "r2"}
    assert full_rel["sinks"] == ["s1"]
    assert full_rel["orphans"] == ["iso"]
    assert full_rel["adjacency_list"]["m1"] == ["s1"]
    assert full_rel["reverse_adjacency_list"]["m1"] == ["r1", "r2"]


def test_add_subgraph_with_connections_and_bindings():
    """Verify adding an arbitrary subgraph with connections, bindings, and loaded nodes provenance."""
    parent = GraphV4(session_id="p_sess", name="parent_graph")
    parent.add_vertex(VertexRecordV4(0, "p_sess", "start_node", "", [], VertexStateV4.IDLE.value))
    parent.add_vertex(VertexRecordV4(0, "p_sess", "end_node", "", [], VertexStateV4.IDLE.value))

    sub = GraphV4(session_id="sub_sess", name="sub_pipeline")
    sub.add_vertex(VertexRecordV4(0, "sub_sess", "step_a", "", [], VertexStateV4.IDLE.value))
    sub.add_vertex(VertexRecordV4(0, "sub_sess", "step_b", "", [], VertexStateV4.IDLE.value))
    sub.add_edge(CodeEdgeV4("edge_ab", "step_a", "step_b"))

    res = parent.add_subgraph(
        subgraph=sub,
        name_prefix="sub",
        incoming_bindings={"start_node": "step_a"},
        connections=[{"from": "sub_step_b", "to": "end_node", "type": "code"}],
        source="catalog:sub_pipeline_v1",
    )

    assert set(res["added_vertices"]) == {"sub_step_a", "sub_step_b"}
    assert "sub_step_a" in parent.vertices
    assert "sub_step_b" in parent.vertices
    assert parent.get_loaded_node("sub_step_a")["source"] == "catalog:sub_pipeline_v1"
    assert parent.get_loaded_node("sub_step_b")["source"] == "catalog:sub_pipeline_v1"

    # Verify edge connectivity
    assert parent.get_node_predecessors("sub_step_a") == ["start_node"]
    assert parent.get_node_successors("sub_step_a") == ["sub_step_b"]
    assert parent.get_node_successors("sub_step_b") == ["end_node"]
    assert parent.get_node_predecessors("end_node") == ["sub_step_b"]

    # Verify duplicate naming collision raises ValueError
    with pytest.raises(ValueError, match="collides with existing vertex"):
        parent.add_subgraph(subgraph=sub, name_prefix="sub")


def test_flexible_add_vertex_and_add_edge():
    """Verify add_vertex and add_edge with string, dict, and keyword parameters."""
    graph = GraphV4(session_id="test_flex", name="flex_graph")

    # 1. Add vertex with string name and kwargs
    v1 = graph.add_vertex("input_a", content="hello", state=VertexStateV4.DATA_READY)
    assert v1.name == "input_a"
    assert v1.content == "hello"
    assert v1.state == VertexStateV4.DATA_READY.value
    assert "input_a" in graph.vertices

    # 2. Add vertex with dict
    v2 = graph.add_vertex({"name": "process_b", "content": "world", "state": "todo"})
    assert v2.name == "process_b"
    assert v2.content == "world"
    assert v2.state == VertexStateV4.TODO.value

    # 3. Add edge with pure kwargs
    e1 = graph.add_edge(edge_id="e_ab", input_vertex="input_a", output_vertex="process_b", edge_type="code")
    assert e1.id == "e_ab"
    assert "e_ab" in graph.edges

    # 4. Add edge with dict
    e2 = graph.add_edge({"id": "e_dict", "type": "code", "input_vertex": "input_a", "output_vertex": "process_b"})
    assert e2.id == "e_dict"
    assert "e_dict" in graph.edges


def test_graph_dump_and_to_dict(tmp_path: Path):
    """Verify serialization via dump() to dictionary and JSON file."""
    graph = GraphV4(session_id="dump_sess", name="dump_graph")
    graph.add_vertex("root", content="start", state=VertexStateV4.DATA_READY)
    graph.add_vertex("leaf", content="end", state=VertexStateV4.TODO)
    graph.add_edge(edge_id="e_rl", input_vertex="root", output_vertex="leaf")

    # Dump to dict
    dumped = graph.dump()
    assert dumped["version"] == "4.0"
    assert dumped["session_id"] == "dump_sess"
    assert dumped["name"] == "dump_graph"
    assert len(dumped["vertices"]) == 2
    assert len(dumped["edges"]) == 1
    assert "relationships" in dumped
    assert dumped["relationships"]["roots"] == ["root"]
    assert dumped["relationships"]["sinks"] == ["leaf"]

    # Dump to file
    out_file = tmp_path / "graph_dump.json"
    dumped_file_data = graph.dump(path=out_file)
    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        file_json = json.load(f)
    assert file_json == dumped_file_data
    assert graph.to_dict() == dumped


def test_add_subgraph_from_dict_and_manifest(tmp_path: Path):
    """Verify add_subgraph directly accepts in-memory dict and manifest JSON path."""
    parent = GraphV4(session_id="p_sess", name="parent")
    parent.add_vertex("start_v", content="123", state=VertexStateV4.DATA_READY)

    # Subgraph dict
    sub_dict = {
        "metadata": {"name": "sub_worker"},
        "vertices": [
            {"name": "sub_w1", "state": "idle", "content": "w1"}
        ],
        "edges": []
    }
    res_dict = parent.add_subgraph(
        subgraph=sub_dict,
        name_prefix="d",
        incoming_bindings={"start_v": "sub_w1"},
    )
    assert "d_sub_w1" in res_dict["added_vertices"]
    assert "d_sub_w1" in parent.vertices

    # Subgraph manifest file
    manifest_path = tmp_path / "sub_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {"name": "file_worker"},
            "vertices": [{"name": "file_w1", "state": "idle", "content": "fw"}],
            "edges": []
        }, f)

    res_file = parent.add_subgraph(
        subgraph=manifest_path,
        name_prefix="f",
    )
    assert "f_file_w1" in res_file["added_vertices"]
    assert "f_file_w1" in parent.vertices


