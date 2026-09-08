"""V4 Discrete Graph Manifest and Topology Specification.

Loads modular discrete JSON manifests (one JSON per vertex and edge),
validates topological DAG ordering, and exposes edge dependency levels.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from framework.edge_v4 import CodeEdgeV4, EdgeV4, LLMEdgeV4, ReflexiveEdgeV4
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.graph_v4")


class GraphTopologyError(Exception):
    """Raised when graph definition contains circular dependencies or broken links."""


class GraphV4:
    """Represents a V4 workflow graph consisting of discrete vertices and executable edges."""

    def __init__(
        self,
        session_id: str,
        name: str = "v4_graph",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.name = name
        self.metadata = dict(metadata or {})
        self.vertices: Dict[str, VertexRecordV4] = {}
        self.edges: Dict[str, EdgeV4] = {}
        self.edge_tiers: Dict[str, int] = {}

    def add_vertex(self, vertex: VertexRecordV4) -> None:
        """Register a vertex record in the graph."""
        self.vertices[vertex.name] = vertex

    def add_edge(self, edge: EdgeV4) -> None:
        """Register an edge in the graph."""
        self.edges[edge.id] = edge

    def get_vertex(self, name: str) -> Optional[VertexRecordV4]:
        """Retrieve vertex definition by name."""
        return self.vertices.get(name)

    def delete_vertex(self, name: str) -> bool:
        """Remove a vertex and all its connected edges from the graph."""
        if name not in self.vertices:
            return False
        del self.vertices[name]
        # Remove all edges connected to this vertex
        dangling = [eid for eid, e in self.edges.items()
                    if e.input_vertex == name or e.output_vertex == name]
        for eid in dangling:
            del self.edges[eid]
        # Invalidate tiers
        self.edge_tiers = {k: v for k, v in self.edge_tiers.items() if k not in dangling}
        return True

    def delete_edge(self, edge_id: str) -> bool:
        """Remove an edge from the graph."""
        if edge_id not in self.edges:
            return False
        del self.edges[edge_id]
        self.edge_tiers.pop(edge_id, None)
        return True

    def get_edge(self, edge_id: str) -> Optional[EdgeV4]:
        """Retrieve edge definition by ID."""
        return self.edges.get(edge_id)

    def get_outgoing_edges(self, vertex_name: str) -> List[EdgeV4]:
        """Get all forward and reflexive edges originating from vertex_name."""
        return [e for e in self.edges.values() if e.input_vertex == vertex_name]

    def get_incoming_edges(self, vertex_name: str) -> List[EdgeV4]:
        """Get all forward edges targeting vertex_name."""
        return [
            e for e in self.edges.values()
            if e.output_vertex == vertex_name and not e.is_reflexive
        ]

    def get_reflexive_edges(self, vertex_name: Optional[str] = None) -> List[ReflexiveEdgeV4]:
        """Get reflexive recovery edges, optionally filtered by vertex name."""
        res: List[ReflexiveEdgeV4] = []
        for e in self.edges.values():
            if isinstance(e, ReflexiveEdgeV4) or e.is_reflexive:
                if vertex_name is None or e.input_vertex == vertex_name:
                    res.append(e if isinstance(e, ReflexiveEdgeV4) else ReflexiveEdgeV4(
                        edge_id=e.id,
                        vertex_name=e.input_vertex,
                        script=getattr(e, 'script', None),
                        trigger_state=e.settings.get('trigger_state', VertexStateV4.REJECT.value),
                        target_state=e.settings.get('target_state', VertexStateV4.TODO_URGENT.value),
                        settings=e.settings,
                    ))
        return res

    def validate(self) -> None:
        """Validate vertex bindings and DAG constraints."""
        # Check all edge endpoints exist
        for edge in self.edges.values():
            if edge.input_vertex not in self.vertices:
                raise GraphTopologyError(
                    f"Edge '{edge.id}' references non-existent input vertex '{edge.input_vertex}'"
                )
            if edge.output_vertex not in self.vertices:
                raise GraphTopologyError(
                    f"Edge '{edge.id}' references non-existent output vertex '{edge.output_vertex}'"
                )

        # Validate DAG on forward edges (excluding reflexive self-loops)
        adj: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = {v: 0 for v in self.vertices}

        for edge in self.edges.values():
            if not edge.is_reflexive:
                adj[edge.input_vertex].append(edge.output_vertex)
                in_degree[edge.output_vertex] += 1

        queue = deque([v for v, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            node = queue.popleft()
            visited_count += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(self.vertices):
            raise GraphTopologyError(
                "Cycle detected in forward edges. Forward edges must form a directed acyclic graph (DAG)."
            )

        self.compute_dag_tiers()

    def compute_dag_tiers(self) -> Dict[str, int]:
        """Compute topological tier depth for each edge to guide DAG execution order.

        Tier 0 represents edges originating directly from start or entry vertices.
        Higher tiers depend on outputs from earlier tiers.
        Reflexive edges are assigned Tier -1 (highest priority during recovery).
        """
        # Node tier calculation
        node_tier: Dict[str, int] = {}
        for v in self.vertices.values():
            if v.has_attribute(VertexAttributeV4.START):
                node_tier[v.name] = 0

        # In-degree of forward edges
        in_edges: Dict[str, List[str]] = defaultdict(list)
        for e in self.edges.values():
            if not e.is_reflexive:
                in_edges[e.output_vertex].append(e.input_vertex)

        # Roots are vertices with 0 in-degree or start attribute
        for v_name in self.vertices:
            if not in_edges[v_name]:
                node_tier[v_name] = 0

        max_iterations = len(self.vertices) + 1
        for _ in range(max_iterations):
            changed = False
            for e in self.edges.values():
                if not e.is_reflexive:
                    in_name = e.input_vertex
                    out_name = e.output_vertex
                    if in_name in node_tier:
                        expected_out_tier = node_tier[in_name] + 1
                        if node_tier.get(out_name, -1) < expected_out_tier:
                            node_tier[out_name] = expected_out_tier
                            changed = True
            if not changed:
                break

        # Assign tier to each edge
        tiers: Dict[str, int] = {}
        for e in self.edges.values():
            if e.is_reflexive:
                tiers[e.id] = -1
            else:
                tiers[e.id] = node_tier.get(e.input_vertex, 0)

        self.edge_tiers = tiers
        return tiers

    @classmethod
    def load_from_store(cls, store: VertexStoreV4, session_id: str, name: str = "v4_graph") -> GraphV4:
        """Hydrate a GraphV4 instance from SQLite persisted vertices and edges."""
        graph = cls(session_id=session_id, name=name)
        for v in store.list_vertices(session_id):
            graph.add_vertex(v)

        for er in store.list_edges(session_id):
            edge_type = er.edge_type
            if edge_type == "reflexive" or er.input_vertex == er.output_vertex:
                edge: EdgeV4 = ReflexiveEdgeV4(
                    edge_id=er.edge_id,
                    vertex_name=er.input_vertex,
                    trigger_state=er.trigger_state or VertexStateV4.REJECT.value,
                    target_state=er.target_state or VertexStateV4.TODO_URGENT.value,
                    max_retries=er.max_retries,
                    script=er.script,
                    settings=er.settings,
                )
            elif edge_type == "llm":
                edge = LLMEdgeV4(
                    edge_id=er.edge_id,
                    input_vertex=er.input_vertex,
                    output_vertex=er.output_vertex,
                    model=er.settings.get("model", "sensenova-6.8-flash-lite"),
                    prompt_template=er.settings.get("prompt"),
                    settings=er.settings,
                )
            else:
                edge = CodeEdgeV4(
                    edge_id=er.edge_id,
                    input_vertex=er.input_vertex,
                    output_vertex=er.output_vertex,
                    script=er.script,
                    settings=er.settings,
                )
            graph.add_edge(edge)

        try:
            graph.compute_dag_tiers()
        except Exception:
            pass
        return graph


class DiscreteGraphLoaderV4:
    """Loads discrete vertex and edge JSON components specified by a master manifest."""

    @classmethod
    def load_from_manifest(
        cls,
        manifest_path: Union[str, Path],
        override_session_id: Optional[str] = None,
    ) -> GraphV4:
        """Parse master manifest JSON and all discrete vertex/edge JSON files."""
        path = Path(manifest_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Master manifest not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        base_dir = path.parent
        session_id = override_session_id or data.get("session_id", "default_session")
        name = (data.get("metadata") or {}).get("name", path.stem)
        graph = GraphV4(session_id=session_id, name=name, metadata=data.get("metadata") or {})

        # Load discrete vertices
        vertex_files = data.get("vertices", [])
        for rel_v in vertex_files:
            v_path = (base_dir / rel_v).resolve()
            if not v_path.exists():
                raise FileNotFoundError(f"Discrete vertex file not found: {v_path}")
            with open(v_path, "r", encoding="utf-8") as vf:
                v_data = json.load(vf)

            v_record = VertexRecordV4(
                id=0,
                session_id=session_id,
                name=v_data["name"],
                content=v_data.get("content", ""),
                attributes=v_data.get("attributes", []),
                state=v_data.get("state", VertexStateV4.IDLE.value),
                processed_count=v_data.get("processed_count", 0),
            )
            graph.add_vertex(v_record)

        # Load discrete edges
        edge_files = data.get("edges", [])
        for rel_e in edge_files:
            e_path = (base_dir / rel_e).resolve()
            if not e_path.exists():
                raise FileNotFoundError(f"Discrete edge file not found: {e_path}")
            with open(e_path, "r", encoding="utf-8") as ef:
                e_data = json.load(ef)

            e_id = e_data["id"]
            e_type = e_data.get("type", "code")
            in_v = e_data["input_vertex"]
            out_v = e_data["output_vertex"]
            settings = e_data.get("settings") or {}
            script = e_data.get("script")

            edge_instance: EdgeV4
            if e_type == "reflexive" or in_v == out_v:
                trigger_state = e_data.get("trigger_state", VertexStateV4.REJECT.value)
                target_state = e_data.get("target_state", VertexStateV4.TODO_URGENT.value)
                raw_retries = settings.get("max_retries") or e_data.get("max_retries") or 3
                max_retries = int(raw_retries)
                edge_instance = ReflexiveEdgeV4(
                    edge_id=e_id,
                    vertex_name=in_v,
                    trigger_state=trigger_state,
                    target_state=target_state,
                    max_retries=max_retries,
                    script=script,
                    settings=settings,
                )
            elif e_type == "code":
                edge_instance = CodeEdgeV4(
                    edge_id=e_id,
                    input_vertex=in_v,
                    output_vertex=out_v,
                    script=script,
                    settings=settings,
                )
            elif e_type == "llm":
                edge_instance = LLMEdgeV4(
                    edge_id=e_id,
                    input_vertex=in_v,
                    output_vertex=out_v,
                    model=settings.get("model", "sensenova-6.8-flash-lite"),
                    prompt_template=settings.get("prompt"),
                    settings=settings,
                )
            else:
                raise ValueError(f"Unsupported edge type '{e_type}' in '{e_path}'")

            graph.add_edge(edge_instance)

        graph.validate()
        return graph

    @classmethod
    def populate_store(cls, graph: GraphV4, store: VertexStoreV4) -> None:
        """Seed all graph vertices and edges into SQLite store."""
        for v in graph.vertices.values():
            db_record = store.save_vertex(
                session_id=graph.session_id,
                name=v.name,
                content=v.content,
                attributes=v.attributes,
                state=v.state,
                processed_count=v.processed_count,
            )
            v.id = db_record.id

        for e in graph.edges.values():
            script_val = getattr(e, "script", None)
            script_str = script_val if isinstance(script_val, str) else None
            trigger_state = getattr(e, "trigger_state", None)
            target_state = getattr(e, "target_state", None)
            max_retries = getattr(e, "max_retries", 3)
            store.save_edge(
                session_id=graph.session_id,
                edge_id=e.id,
                edge_type=e.type,
                input_vertex=e.input_vertex,
                output_vertex=e.output_vertex,
                script=script_str,
                trigger_state=trigger_state,
                target_state=target_state,
                max_retries=max_retries,
                settings=e.settings,
            )
