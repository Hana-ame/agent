"""Dynamic subgraph splicing, insertion, re-entry, and cascade reset operations."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from framework.edges.base import EdgeV4
from framework.edges.code import CodeEdgeV4
from framework.edges.reflexive import ReflexiveEdgeV4
from framework.vertex_v4 import VertexAttributeV4, VertexRecordV4, VertexStateV4, VertexStoreV4

if TYPE_CHECKING:
    from framework.graphs.core import GraphV4

logger = logging.getLogger("vertex_edge_agent.graphs.subgraph_ops")


def reset_affected_vertices_op(
    graph: "GraphV4",
    reenter_vertex: str,
    reset_state: str = VertexStateV4.TODO.value,
    clear_content: bool = False,
    store: Optional[VertexStoreV4] = None,
    session_id: Optional[str] = None,
) -> List[str]:
    """Reset all downstream vertices affected by re-entry into reenter_vertex."""
    downstream = graph.get_downstream_vertices(reenter_vertex, include_self=False)
    target_session = session_id or graph.session_id

    for v_name in downstream:
        v = graph.vertices.get(v_name)
        if v:
            v.state = reset_state
            if clear_content:
                v.content = ""
        if store:
            if clear_content:
                store.update_vertex_content(target_session, v_name, "", reset_state)
            else:
                store.update_vertex_state(target_session, v_name, reset_state)

    return downstream


def reenter_vertex_op(
    graph: "GraphV4",
    vertex_name: str,
    new_content: Optional[str] = None,
    reset_state: str = VertexStateV4.TODO.value,
    clear_content: bool = False,
    store: Optional[VertexStoreV4] = None,
    session_id: Optional[str] = None,
) -> List[str]:
    """Re-enter a vertex for re-execution and reset all affected downstream vertices."""
    if vertex_name not in graph.vertices:
        raise KeyError(f"Vertex '{vertex_name}' not found in graph")

    target_session = session_id or graph.session_id
    v = graph.vertices[vertex_name]
    v.state = VertexStateV4.DATA_READY.value
    if new_content is not None:
        v.content = new_content

    if store:
        if new_content is not None:
            store.update_vertex_content(
                target_session, vertex_name, new_content, VertexStateV4.DATA_READY.value
            )
        else:
            store.update_vertex_state(
                target_session, vertex_name, VertexStateV4.DATA_READY.value
            )

    affected = reset_affected_vertices_op(
        graph=graph,
        reenter_vertex=vertex_name,
        reset_state=reset_state,
        clear_content=clear_content,
        store=store,
        session_id=target_session,
    )
    return affected


def splice_subgraph_op(
    graph: "GraphV4",
    target_vertex_name: str,
    subgraph: "GraphV4",
    name_prefix: Optional[str] = None,
    entry_vertex_name: Optional[str] = None,
    exit_vertex_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Splice (inline) a subgraph in place of an existing vertex."""
    if target_vertex_name not in graph.vertices:
        raise KeyError(f"Target vertex '{target_vertex_name}' not found in graph")

    prefix = f"{name_prefix}_" if name_prefix else ""
    name_map: Dict[str, str] = {
        old_v: f"{prefix}{old_v}" if prefix else old_v
        for old_v in subgraph.vertices
    }

    for old_v, new_v in name_map.items():
        if new_v in graph.vertices and new_v != target_vertex_name:
            raise ValueError(
                f"Subgraph vertex '{new_v}' collides with existing graph vertex. Provide a distinct name_prefix."
            )

    sub_in_edges: Dict[str, List[str]] = defaultdict(list)
    sub_out_edges: Dict[str, List[str]] = defaultdict(list)
    for e in subgraph.edges.values():
        if not e.is_reflexive:
            sub_in_edges[e.output_vertex].append(e.input_vertex)
            sub_out_edges[e.input_vertex].append(e.output_vertex)

    entry_nodes = [
        name_map[v.name] for v in subgraph.vertices.values()
        if (entry_vertex_name and v.name == entry_vertex_name)
        or (not entry_vertex_name and (not sub_in_edges[v.name] or v.has_attribute(VertexAttributeV4.START)))
    ]
    exit_nodes = [
        name_map[v.name] for v in subgraph.vertices.values()
        if (exit_vertex_name and v.name == exit_vertex_name)
        or (not exit_vertex_name and (not sub_out_edges[v.name] or v.has_attribute(VertexAttributeV4.END)))
    ]

    if not entry_nodes and subgraph.vertices:
        entry_nodes = [name_map[next(iter(subgraph.vertices))]]
    if not exit_nodes and subgraph.vertices:
        exit_nodes = [name_map[next(reversed(list(subgraph.vertices)))]]

    # 1. Insert vertices
    inserted_vertices: List[str] = []
    for old_name, v in subgraph.vertices.items():
        new_name = name_map[old_name]
        cloned_v = VertexRecordV4(
            id=0,
            session_id=graph.session_id,
            name=new_name,
            content=v.content,
            attributes=list(v.attributes),
            state=v.state,
            processed_count=v.processed_count,
        )
        graph.add_vertex(cloned_v, source=f"splice:{target_vertex_name}")
        inserted_vertices.append(new_name)

    # 2. Insert internal edges
    inserted_edges: List[str] = []
    for e in subgraph.edges.values():
        e_cfg = e.to_dict()
        new_eid = f"{prefix}{e.id}" if prefix else e.id
        if new_eid in graph.edges:
            new_eid = f"{new_eid}_sub_{target_vertex_name}"
        e_cfg["id"] = new_eid
        e_cfg["input_vertex"] = name_map.get(e.input_vertex, e.input_vertex)
        e_cfg["output_vertex"] = name_map.get(e.output_vertex, e.output_vertex)
        cloned_edge = EdgeV4.from_config(e_cfg)
        graph.add_edge(cloned_edge)
        inserted_edges.append(cloned_edge.id)

    # 3. Rewire incident edges
    rewired_edges: List[str] = []
    primary_entry = entry_nodes[0] if entry_nodes else None
    primary_exit = exit_nodes[0] if exit_nodes else None

    for e in list(graph.edges.values()):
        if e.output_vertex == target_vertex_name:
            if primary_entry:
                e.output_vertex = primary_entry
                rewired_edges.append(e.id)
                for extra_entry in entry_nodes[1:]:
                    extra_cfg = e.to_dict()
                    extra_cfg["id"] = f"{e.id}_fan_{extra_entry}"
                    extra_cfg["output_vertex"] = extra_entry
                    extra_edge = EdgeV4.from_config(extra_cfg)
                    graph.add_edge(extra_edge)
                    inserted_edges.append(extra_edge.id)

    for e in list(graph.edges.values()):
        if e.input_vertex == target_vertex_name:
            if primary_exit:
                e.input_vertex = primary_exit
                rewired_edges.append(e.id)
                for extra_exit in exit_nodes[1:]:
                    extra_cfg = e.to_dict()
                    extra_cfg["id"] = f"{e.id}_fan_{extra_exit}"
                    extra_cfg["input_vertex"] = extra_exit
                    extra_edge = EdgeV4.from_config(extra_cfg)
                    graph.add_edge(extra_edge)
                    inserted_edges.append(extra_edge.id)

    # 4. Remove original target vertex
    graph.delete_vertex(target_vertex_name)

    # 5. Revalidate endpoints and compute tiers softly
    graph.validate(strict_dag=False)

    return {
        "spliced_vertex": target_vertex_name,
        "entry_vertices": entry_nodes,
        "exit_vertices": exit_nodes,
        "inserted_vertices": inserted_vertices,
        "inserted_edges": inserted_edges,
        "rewired_edges": rewired_edges,
    }


def insert_subgraph_op(
    graph: "GraphV4",
    subgraph: "GraphV4",
    incoming_bindings: Optional[Dict[str, str]] = None,
    outgoing_bindings: Optional[Dict[str, str]] = None,
    name_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert an independent subgraph with explicit boundary edge bindings."""
    prefix = f"{name_prefix}_" if name_prefix else ""
    name_map: Dict[str, str] = {
        old_v: f"{prefix}{old_v}" if prefix else old_v
        for old_v in subgraph.vertices
    }

    for new_v in name_map.values():
        if new_v in graph.vertices:
            raise ValueError(f"Subgraph vertex '{new_v}' collides with existing graph vertex.")

    inserted_vertices: List[str] = []
    for old_name, v in subgraph.vertices.items():
        new_name = name_map[old_name]
        cloned_v = VertexRecordV4(
            id=0,
            session_id=graph.session_id,
            name=new_name,
            content=v.content,
            attributes=list(v.attributes),
            state=v.state,
            processed_count=v.processed_count,
        )
        graph.add_vertex(cloned_v, source="subgraph_insert")
        inserted_vertices.append(new_name)

    inserted_edges: List[str] = []
    for e in subgraph.edges.values():
        e_cfg = e.to_dict()
        new_eid = f"{prefix}{e.id}" if prefix else e.id
        if new_eid in graph.edges:
            new_eid = f"{new_eid}_sub"
        e_cfg["id"] = new_eid
        e_cfg["input_vertex"] = name_map.get(e.input_vertex, e.input_vertex)
        e_cfg["output_vertex"] = name_map.get(e.output_vertex, e.output_vertex)
        cloned_edge = EdgeV4.from_config(e_cfg)
        graph.add_edge(cloned_edge)
        inserted_edges.append(cloned_edge.id)

    for parent_v, sub_v in (incoming_bindings or {}).items():
        mapped_sub = name_map.get(sub_v, sub_v)
        bridge_id = f"bridge_in_{parent_v}_{mapped_sub}"
        bridge_edge = CodeEdgeV4(edge_id=bridge_id, input_vertex=parent_v, output_vertex=mapped_sub)
        graph.add_edge(bridge_edge)
        inserted_edges.append(bridge_id)

    for sub_v, parent_v in (outgoing_bindings or {}).items():
        mapped_sub = name_map.get(sub_v, sub_v)
        bridge_id = f"bridge_out_{mapped_sub}_{parent_v}"
        bridge_edge = CodeEdgeV4(edge_id=bridge_id, input_vertex=mapped_sub, output_vertex=parent_v)
        graph.add_edge(bridge_edge)
        inserted_edges.append(bridge_id)

    graph.validate(strict_dag=False)

    return {
        "inserted_vertices": inserted_vertices,
        "inserted_edges": inserted_edges,
        "incoming_bindings": incoming_bindings or {},
        "outgoing_bindings": outgoing_bindings or {},
    }


def add_subgraph_op(
    graph: "GraphV4",
    subgraph: Union["GraphV4", Dict[str, Any], str, Path],
    name_prefix: Optional[str] = None,
    connections: Optional[List[Dict[str, Any]]] = None,
    incoming_bindings: Optional[Dict[str, str]] = None,
    outgoing_bindings: Optional[Dict[str, str]] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Add and join a sub-graph into this graph."""
    from framework.graphs.loader import DiscreteGraphLoaderV4

    if isinstance(subgraph, (str, Path)):
        subgraph_obj = DiscreteGraphLoaderV4.load_from_manifest(subgraph)
    elif isinstance(subgraph, dict):
        subgraph_obj = DiscreteGraphLoaderV4.load_from_dict(subgraph)
    elif hasattr(subgraph, "vertices") and hasattr(subgraph, "edges"):
        subgraph_obj = subgraph
    else:
        raise TypeError(f"Unsupported subgraph type: {type(subgraph).__name__}")

    prefix = f"{name_prefix}_" if name_prefix else ""
    name_map: Dict[str, str] = {
        old_v: f"{prefix}{old_v}" if prefix else old_v
        for old_v in subgraph_obj.vertices
    }

    for old_v, new_v in name_map.items():
        if new_v in graph.vertices:
            raise ValueError(
                f"Subgraph vertex '{new_v}' collides with existing vertex in graph. Provide a distinct name_prefix."
            )

    added_vertices: List[str] = []
    node_source = source or f"subgraph:{subgraph_obj.name}"
    for old_name, v in subgraph_obj.vertices.items():
        new_name = name_map[old_name]
        cloned_v = VertexRecordV4(
            id=0,
            session_id=graph.session_id,
            name=new_name,
            content=v.content,
            attributes=list(v.attributes),
            state=v.state,
            processed_count=v.processed_count,
        )
        graph.add_vertex(cloned_v, source=node_source)
        added_vertices.append(new_name)

    added_edges: List[str] = []
    for e in subgraph_obj.edges.values():
        e_cfg = e.to_dict()
        new_eid = f"{prefix}{e.id}" if prefix else e.id
        if new_eid in graph.edges:
            new_eid = f"{new_eid}_sub_{subgraph_obj.session_id}"
        e_cfg["id"] = new_eid
        e_cfg["input_vertex"] = name_map.get(e.input_vertex, e.input_vertex)
        e_cfg["output_vertex"] = name_map.get(e.output_vertex, e.output_vertex)
        cloned_edge = EdgeV4.from_config(e_cfg)
        graph.add_edge(cloned_edge)
        added_edges.append(cloned_edge.id)

    for parent_v, sub_v in (incoming_bindings or {}).items():
        mapped_sub = name_map.get(sub_v, sub_v)
        bridge_id = f"bridge_in_{parent_v}_{mapped_sub}"
        bridge_edge = CodeEdgeV4(edge_id=bridge_id, input_vertex=parent_v, output_vertex=mapped_sub)
        graph.add_edge(bridge_edge)
        added_edges.append(bridge_id)

    for sub_v, parent_v in (outgoing_bindings or {}).items():
        mapped_sub = name_map.get(sub_v, sub_v)
        bridge_id = f"bridge_out_{mapped_sub}_{parent_v}"
        bridge_edge = CodeEdgeV4(edge_id=bridge_id, input_vertex=mapped_sub, output_vertex=parent_v)
        graph.add_edge(bridge_edge)
        added_edges.append(bridge_id)

    for conn in (connections or []):
        c_from = conn.get("from") or conn.get("input_vertex") or conn.get("source")
        c_to = conn.get("to") or conn.get("output_vertex") or conn.get("target")
        if not c_from or not c_to:
            continue
        mapped_from = name_map.get(c_from, c_from)
        mapped_to = name_map.get(c_to, c_to)
        c_type = conn.get("type", "code")
        c_id = conn.get("edge_id") or conn.get("id") or f"conn_{mapped_from}_{mapped_to}"
        if c_type == "reflexive":
            conn_edge: EdgeV4 = ReflexiveEdgeV4(
                edge_id=c_id,
                vertex_name=mapped_from,
                trigger_state=conn.get("trigger_state", VertexStateV4.REJECT.value),
                target_state=conn.get("target_state", VertexStateV4.TODO_URGENT.value),
                script=conn.get("script"),
                settings=conn.get("settings", {}),
            )
        else:
            conn_edge = CodeEdgeV4(
                edge_id=c_id,
                input_vertex=mapped_from,
                output_vertex=mapped_to,
                script=conn.get("script"),
                settings=conn.get("settings", {}),
            )
        graph.add_edge(conn_edge)
        added_edges.append(conn_edge.id)

    graph.validate(strict_dag=False)

    return {
        "subgraph_name": subgraph_obj.name,
        "name_prefix": name_prefix,
        "added_vertices": added_vertices,
        "added_edges": added_edges,
        "name_mapping": name_map,
    }
