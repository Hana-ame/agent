"""Topological validation, cycle detection, and DAG tier calculation for GraphV4."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple

from framework.graphs.exceptions import GraphTopologyError
from framework.vertex_v4 import TraversalColor, VertexAttributeV4, VertexStateV4

if TYPE_CHECKING:
    from framework.graphs.core import GraphV4

logger = logging.getLogger("vertex_edge_agent.graphs.validation")


def detect_cycles_and_order(graph: "GraphV4", strict_dag: bool = True) -> List[str]:
    """Detect cycles using state coloring DFS and return topological ordering.

    Uses :class:`TraversalColor` (WHITE, GRAY, BLACK) held **only** in
    ``graph.node_states``. Vertex lifecycle states are never touched: doing so
    used to overwrite persisted ``idle``/``todo`` states with DFS colours.

    Args:
        graph: Target GraphV4 instance.
        strict_dag: If True, raises GraphTopologyError when cycles are detected.
                    If False, breaks cycles gracefully to permit cyclic execution and dynamic graphs.

    Returns:
        List[str]: Vertices ordered in forward topological sequence.

    Raises:
        GraphTopologyError: If circular dependencies are detected in forward edges and strict_dag is True.
    """
    adj: Dict[str, List[str]] = {v: [] for v in graph.vertices}
    for edge in graph.edges.values():
        if not edge.is_reflexive:
            if edge.input_vertex in adj:
                adj[edge.input_vertex].append(edge.output_vertex)

    states: Dict[str, TraversalColor] = {v: TraversalColor.WHITE for v in graph.vertices}
    current_path: List[str] = []
    post_order: List[str] = []

    def dfs(node: str) -> None:
        states[node] = TraversalColor.GRAY
        current_path.append(node)

        for neighbor in adj.get(node, []):
            state = states.get(neighbor)
            if state == TraversalColor.GRAY:
                cycle_start_idx = current_path.index(neighbor)
                cycle_path = current_path[cycle_start_idx:] + [neighbor]
                cycle_str = " -> ".join(cycle_path)
                graph.node_states = dict(states)
                if strict_dag:
                    raise GraphTopologyError(
                        f"Cycle detected in forward edges: {cycle_str}. "
                        "Forward edges must form a directed acyclic graph (DAG)."
                    )
                else:
                    logger.info("Cycle detected at definition time: %s (will be resolved at runtime)", cycle_str)
                    continue
            elif state == TraversalColor.WHITE:
                dfs(neighbor)
            elif state is None:
                graph.node_states = dict(states)
                raise GraphTopologyError(
                    f"Forward edge targeting non-existent vertex '{neighbor}'"
                )

        current_path.pop()
        states[node] = TraversalColor.BLACK
        post_order.append(node)

    for v in sorted(graph.vertices.keys()):
        if states[v] == TraversalColor.WHITE:
            dfs(v)

    graph.node_states = dict(states)
    return list(reversed(post_order))


def validate_topology(graph: "GraphV4", strict_dag: bool = True) -> None:
    """Validate vertex bindings and DAG constraints using state coloring.

    Args:
        graph: Target GraphV4 instance.
        strict_dag: If True, enforces strictly acyclic forward edges.
                    If False, validates endpoint references and soft-computes tiers.
    """
    for edge in graph.edges.values():
        if edge.input_vertex not in graph.vertices:
            raise GraphTopologyError(
                f"Edge '{edge.id}' references non-existent input vertex '{edge.input_vertex}'"
            )
        if edge.output_vertex not in graph.vertices:
            raise GraphTopologyError(
                f"Edge '{edge.id}' references non-existent output vertex '{edge.output_vertex}'"
            )

    detect_cycles_and_order(graph, strict_dag=strict_dag)
    compute_dag_tiers(graph, strict_dag=strict_dag)


def compute_dag_tiers(graph: "GraphV4", strict_dag: bool = False) -> Dict[str, int]:
    """Compute topological tier depth for each edge to guide DAG execution order.

    Tier 0 represents edges originating directly from start or entry vertices.
    Higher tiers depend on outputs from earlier tiers.
    Reflexive edges are assigned Tier -1 (highest priority during recovery).
    """
    try:
        topo_order = detect_cycles_and_order(graph, strict_dag=strict_dag)
    except GraphTopologyError:
        if strict_dag:
            raise
        topo_order = list(graph.vertices.keys())

    node_tier: Dict[str, int] = {}
    for v in graph.vertices.values():
        if v.has_attribute(VertexAttributeV4.START):
            node_tier[v.name] = 0

    in_edges: Dict[str, List[str]] = defaultdict(list)
    for e in graph.edges.values():
        if not e.is_reflexive:
            in_edges[e.output_vertex].append(e.input_vertex)

    for v_name in graph.vertices:
        if not in_edges[v_name]:
            if graph.is_vertex_active(v_name):
                node_tier[v_name] = 0
            else:
                node_tier[v_name] = -1

    adj_edges: Dict[str, List[Any]] = defaultdict(list)
    for e in graph.edges.values():
        if not e.is_reflexive:
            adj_edges[e.input_vertex].append(e)

    for u in topo_order:
        if not graph.is_vertex_active(u):
            continue
        curr_tier = node_tier.get(u, 0)
        for e in adj_edges.get(u, []):
            out_name = e.output_vertex
            expected = curr_tier + 1
            if node_tier.get(out_name, -1) < expected:
                node_tier[out_name] = expected

    tiers: Dict[str, int] = {}
    for e in graph.edges.values():
        if e.is_reflexive:
            tiers[e.id] = -1
        else:
            tiers[e.id] = node_tier.get(e.input_vertex, 0)

    graph.node_tiers = dict(node_tier)
    graph.edge_tiers = tiers
    return tiers
