"""V4 Discrete Graph Manifest and Topology Specification.

Loads modular discrete JSON manifests (one JSON per vertex and edge),
validates topological DAG ordering, and exposes edge dependency levels.
"""

from __future__ import annotations

import enum
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from framework.edge_v4 import (
    CallableLLMEdgeV4,
    ChatLLMEdgeV4,
    CodeEdgeV4,
    EdgeV4,
    GenerateLLMEdgeV4,
    LLMEdgeV4,
    ProcessLLMEdgeV4,
    ReflexiveEdgeV4,
)
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.graph_v4")

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)

# NodeColor is unified into VertexStateV4 for state coloring
NodeColor = VertexStateV4


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
        self.node_tiers: Dict[str, int] = {}
        self.node_states: Dict[str, VertexStateV4] = {}
        self.loaded_nodes: Dict[str, Dict[str, Any]] = {}
        self._active_overrides: Dict[str, bool] = {}

    @property
    def node_colors(self) -> Dict[str, VertexStateV4]:
        """Backward-compatible alias for node_states."""
        return self.node_states

    @node_colors.setter
    def node_colors(self, val: Dict[str, VertexStateV4]) -> None:
        self.node_states = val

    def add_vertex(
        self,
        vertex: Union[VertexRecordV4, Dict[str, Any], str],
        source: str = "memory",
        metadata: Optional[Dict[str, Any]] = None,
        content: str = "",
        attributes: Optional[List[str]] = None,
        state: Optional[Union[VertexStateV4, str]] = None,
        processed_count: int = 0,
        **kwargs: Any,
    ) -> VertexRecordV4:
        """Register a vertex record in the graph with provenance metadata.

        Supports VertexRecordV4 instances, configuration dicts, or vertex names.
        """
        if isinstance(vertex, (str, Path)):
            v_p = Path(str(vertex))
            if not v_p.is_absolute() and not v_p.exists():
                cand = Path(_REPO_ROOT) / v_p
                if cand.exists():
                    v_p = cand
            if v_p.is_dir():
                loaded_list = []
                for json_file in sorted(v_p.glob("*.json")):
                    with open(json_file, "r", encoding="utf-8") as vf:
                        v_data = json.load(vf)
                    if isinstance(v_data, list):
                        for item in v_data:
                            loaded_list.append(self.add_vertex(item, source=f"file:{json_file}", metadata=metadata, **kwargs))
                    else:
                        loaded_list.append(self.add_vertex(v_data, source=f"file:{json_file}", metadata=metadata, **kwargs))
                return loaded_list[0] if len(loaded_list) == 1 else loaded_list
            elif str(vertex).endswith(".json") or v_p.is_file():
                if not v_p.exists():
                    raise FileNotFoundError(f"Vertex config file not found: {vertex}")
                with open(v_p, "r", encoding="utf-8") as vf:
                    v_data = json.load(vf)
                if isinstance(v_data, list):
                    res = [self.add_vertex(item, source=f"file:{v_p}", metadata=metadata, **kwargs) for item in v_data]
                    return res[0] if len(res) == 1 else res
                return self.add_vertex(v_data, source=f"file:{v_p}", metadata=metadata, **kwargs)

        if isinstance(vertex, VertexRecordV4):
            v_record = vertex
        elif isinstance(vertex, dict):
            raw_state = vertex.get("state", state or VertexStateV4.IDLE.value)
            if isinstance(raw_state, VertexStateV4):
                st_val = raw_state.value
            else:
                raw_str = str(raw_state)
                norm = raw_str.replace("_", " ").lower()
                valid_states = {s.value for s in VertexStateV4}
                st_val = norm if norm in valid_states else raw_str

            raw_attrs = list(vertex.get("attributes", attributes or []))
            valid_attrs = {a.value for a in VertexAttributeV4}
            norm_attrs = []
            for a in raw_attrs:
                a_str = a.value if isinstance(a, VertexAttributeV4) else str(a)
                norm_a = a_str.replace("_", " ").lower()
                norm_attrs.append(norm_a if norm_a in valid_attrs else a_str)

            raw_content = vertex.get("content", content)
            content_str = json.dumps(raw_content) if isinstance(raw_content, (dict, list)) else str(raw_content)
            v_record = VertexRecordV4(
                id=int(vertex.get("id", 0)),
                session_id=str(vertex.get("session_id", self.session_id)),
                name=str(vertex["name"]),
                content=content_str,
                attributes=norm_attrs,
                state=st_val,
                processed_count=int(vertex.get("processed_count", processed_count)),
            )
            metadata = metadata or vertex.get("metadata")
            source = vertex.get("source", source)
        elif isinstance(vertex, str):
            if isinstance(state, VertexStateV4):
                st_val = state.value
            elif state:
                raw_str = str(state)
                norm = raw_str.replace("_", " ").lower()
                valid_states = {s.value for s in VertexStateV4}
                st_val = norm if norm in valid_states else raw_str
            else:
                st_val = VertexStateV4.IDLE.value

            raw_attrs = list(attributes or [])
            valid_attrs = {a.value for a in VertexAttributeV4}
            norm_attrs = []
            for a in raw_attrs:
                a_str = a.value if isinstance(a, VertexAttributeV4) else str(a)
                norm_a = a_str.replace("_", " ").lower()
                norm_attrs.append(norm_a if norm_a in valid_attrs else a_str)

            v_record = VertexRecordV4(
                id=0,
                session_id=self.session_id,
                name=vertex,
                content=content,
                attributes=norm_attrs,
                state=st_val,
                processed_count=processed_count,
            )
        else:
            raise TypeError(f"Unsupported vertex type: {type(vertex).__name__}")

        self.vertices[v_record.name] = v_record
        now = datetime.now(timezone.utc).isoformat()
        self.loaded_nodes[v_record.name] = {
            "name": v_record.name,
            "session_id": v_record.session_id,
            "source": source,
            "loaded_at": now,
            "state": v_record.state,
            "attributes": list(v_record.attributes),
            "metadata": dict(metadata or {}),
        }
        return v_record

    def get_loaded_node(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve tracking metadata for a loaded vertex."""
        info = self.loaded_nodes.get(name)
        if info is not None:
            v = self.vertices.get(name)
            if v:
                info["state"] = v.state
                info["attributes"] = list(v.attributes)
            return dict(info)
        return None

    def list_loaded_nodes(self) -> List[Dict[str, Any]]:
        """List all loaded vertices with provenance and lifecycle metadata."""
        res: List[Dict[str, Any]] = []
        for name in self.vertices:
            info = self.get_loaded_node(name)
            if info:
                res.append(info)
            else:
                v = self.vertices[name]
                res.append({
                    "name": v.name,
                    "session_id": v.session_id,
                    "source": "memory",
                    "loaded_at": "",
                    "state": v.state,
                    "attributes": list(v.attributes),
                    "metadata": {},
                })
        return res


    def add_edge(
        self,
        edge: Optional[Union[EdgeV4, Dict[str, Any], str]] = None,
        *,
        edge_id: Optional[str] = None,
        input_vertex: Optional[str] = None,
        output_vertex: Optional[str] = None,
        edge_type: str = "code",
        script: Optional[str] = None,
        trigger_state: Optional[str] = None,
        target_state: Optional[str] = None,
        max_retries: int = 3,
        settings: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> EdgeV4:
        """Register an edge in the graph.

        Supports EdgeV4 instances, configuration dicts, edge IDs with keyword args,
        or pure keyword arguments.
        """
        if isinstance(edge, (str, Path)):
            e_p = Path(str(edge))
            if not e_p.is_absolute() and not e_p.exists():
                cand = Path(_REPO_ROOT) / e_p
                if cand.exists():
                    e_p = cand
            if e_p.is_dir():
                loaded_list = []
                for json_file in sorted(e_p.glob("*.json")):
                    with open(json_file, "r", encoding="utf-8") as ef:
                        e_data = json.load(ef)
                    if isinstance(e_data, list):
                        for item in e_data:
                            edge_obj = EdgeV4.from_config(item, base_dir=str(json_file.parent))
                            self.edges[edge_obj.id] = edge_obj
                            loaded_list.append(edge_obj)
                    else:
                        edge_obj = EdgeV4.from_config_file(json_file, base_dir=str(json_file.parent))
                        self.edges[edge_obj.id] = edge_obj
                        loaded_list.append(edge_obj)
                return loaded_list[0] if len(loaded_list) == 1 else loaded_list
            elif str(edge).endswith(".json") or e_p.is_file():
                if not e_p.exists():
                    raise FileNotFoundError(f"Edge config file not found: {edge}")
                with open(e_p, "r", encoding="utf-8") as ef:
                    e_data = json.load(ef)
                if isinstance(e_data, list):
                    loaded_list = []
                    for item in e_data:
                        edge_obj = EdgeV4.from_config(item, base_dir=str(e_p.parent))
                        self.edges[edge_obj.id] = edge_obj
                        loaded_list.append(edge_obj)
                    return loaded_list[0] if len(loaded_list) == 1 else loaded_list
                edge_obj = EdgeV4.from_config_file(e_p, base_dir=str(e_p.parent))
                self.edges[edge_obj.id] = edge_obj
                return edge_obj

        if isinstance(edge, EdgeV4):
            edge_obj = edge
        elif isinstance(edge, dict):
            edge_obj = EdgeV4.from_config(edge)
        else:
            eid = edge if isinstance(edge, str) else (edge_id or f"edge_{input_vertex}_{output_vertex}")
            cfg: Dict[str, Any] = {
                "id": eid,
                "type": edge_type,
                "input_vertex": input_vertex,
                "output_vertex": output_vertex,
                "max_retries": max_retries,
                "settings": dict(settings or {}),
            }
            if script:
                cfg["script"] = script
            if trigger_state:
                cfg["trigger_state"] = trigger_state
            if target_state:
                cfg["target_state"] = target_state
            cfg.update(kwargs)
            edge_obj = EdgeV4.from_config(cfg)

        self.edges[edge_obj.id] = edge_obj
        return edge_obj

    def load_directory(self, directory_path: Union[str, Path]) -> "GraphV4":
        """Load vertex and edge configurations from a directory into this graph."""
        loaded = DiscreteGraphLoaderV4.load_from_directory(
            directory_path, override_session_id=self.session_id
        )
        for v in loaded.vertices.values():
            self.vertices[v.name] = v
            if v.name in loaded.loaded_nodes:
                self.loaded_nodes[v.name] = loaded.loaded_nodes[v.name]
        for e in loaded.edges.values():
            self.edges[e.id] = e
        return self

    @classmethod
    def from_directory(
        cls,
        directory_path: Union[str, Path],
        session_id: Optional[str] = None,
    ) -> "GraphV4":
        """Construct a new GraphV4 from a directory containing configurations."""
        return DiscreteGraphLoaderV4.load_from_directory(
            directory_path, override_session_id=session_id
        )

    def get_vertex(self, name: str) -> Optional[VertexRecordV4]:
        """Retrieve vertex definition by name."""
        return self.vertices.get(name)

    def delete_vertex(self, name: str) -> bool:
        """Remove a vertex and all its connected edges from the graph."""
        if name not in self.vertices:
            return False
        del self.vertices[name]
        self.node_colors.pop(name, None)
        self.node_tiers.pop(name, None)
        self.loaded_nodes.pop(name, None)
        self._active_overrides.pop(name, None)
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

    def is_orphan(self, vertex_name: str) -> bool:
        """Return True if vertex has no incoming, outgoing, or reflexive edges."""
        if vertex_name not in self.vertices:
            return False
        for e in self.edges.values():
            if e.input_vertex == vertex_name or e.output_vertex == vertex_name:
                return False
        return True

    def get_orphans(self) -> List[str]:
        """Return names of all orphan (isolated) vertices in the graph."""
        connected = {e.input_vertex for e in self.edges.values()} | {e.output_vertex for e in self.edges.values()}
        return [v for v in self.vertices if v not in connected]

    def is_vertex_active(self, vertex_name: str) -> bool:
        """Return True if vertex is active.

        - Orphan vertices are inactive by default, unless explicitly activated
          via activate_vertex() or tagged with VertexAttributeV4.ACTIVE.
        - Connected vertices are active by default, unless explicitly deactivated
          via deactivate_vertex() or tagged with VertexAttributeV4.INACTIVE.
        """
        if vertex_name not in self.vertices:
            return False
        if vertex_name in self._active_overrides:
            return self._active_overrides[vertex_name]
        v = self.vertices[vertex_name]
        if v.has_attribute(VertexAttributeV4.ACTIVE):
            return True
        if v.has_attribute(VertexAttributeV4.INACTIVE):
            return False
        # Orphan vertices are inactive by default; connected nodes are active
        return not self.is_orphan(vertex_name)

    def activate_vertex(self, vertex_name: str) -> None:
        """Explicitly activate a vertex (e.g. an orphan node)."""
        if vertex_name not in self.vertices:
            raise KeyError(f"Vertex '{vertex_name}' not found in graph")
        self._active_overrides[vertex_name] = True

    def deactivate_vertex(self, vertex_name: str) -> None:
        """Explicitly deactivate a vertex."""
        if vertex_name not in self.vertices:
            raise KeyError(f"Vertex '{vertex_name}' not found in graph")
        self._active_overrides[vertex_name] = False

    def get_active_vertices(self) -> List[str]:
        """Return names of all active vertices in the graph."""
        return [v for v in self.vertices if self.is_vertex_active(v)]

    def get_inactive_vertices(self) -> List[str]:
        """Return names of all inactive vertices in the graph."""
        return [v for v in self.vertices if not self.is_vertex_active(v)]

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
                    if isinstance(e, ReflexiveEdgeV4):
                        res.append(e)
                    else:
                        logger.warning("Constructing fallback ReflexiveEdgeV4 for non-ReflexiveEdgeV4 edge '%s'", e.id)
                        res.append(ReflexiveEdgeV4(
                            edge_id=e.id,
                            vertex_name=e.input_vertex,
                            script=getattr(e, 'script', None),
                            trigger_state=e.settings.get('trigger_state', VertexStateV4.REJECT.value),
                            target_state=e.settings.get('target_state', VertexStateV4.TODO_URGENT.value),
                            settings=e.settings,
                        ))
        return res

    def reconnect_edge(
        self,
        edge_id: str,
        new_input_vertex: Optional[str] = None,
        new_output_vertex: Optional[str] = None,
    ) -> bool:
        """Dynamically reconnect an existing edge to new input or output vertices."""
        edge = self.edges.get(edge_id)
        if edge is None:
            return False
        if new_input_vertex is not None:
            edge.input_vertex = new_input_vertex
        if new_output_vertex is not None:
            edge.output_vertex = new_output_vertex
        self.edge_tiers.pop(edge_id, None)
        return True

    def replace_vertex(
        self,
        old_name: str,
        new_vertex: VertexRecordV4,
        transfer_edges: bool = True,
    ) -> bool:
        """Replace an existing vertex with a new vertex, optionally transferring incident edges."""
        if old_name not in self.vertices:
            return False
        del self.vertices[old_name]
        self.node_colors.pop(old_name, None)
        self.node_tiers.pop(old_name, None)
        self.loaded_nodes.pop(old_name, None)
        self._active_overrides.pop(old_name, None)

        self.add_vertex(new_vertex, source="replace")

        if transfer_edges and old_name != new_vertex.name:
            for edge in self.edges.values():
                if edge.input_vertex == old_name:
                    edge.input_vertex = new_vertex.name
                if edge.output_vertex == old_name:
                    edge.output_vertex = new_vertex.name

        self.compute_dag_tiers()
        return True

    def get_downstream_vertices(self, start_vertex: str, include_self: bool = False) -> List[str]:
        """Compute downstream transitive closure reachable from start_vertex via forward edges."""
        if start_vertex not in self.vertices:
            return []

        adj: Dict[str, List[str]] = defaultdict(list)
        for e in self.edges.values():
            if not e.is_reflexive:
                adj[e.input_vertex].append(e.output_vertex)

        visited: set[str] = set()
        queue: deque[str] = deque([start_vertex])
        result: List[str] = []

        while queue:
            curr = queue.popleft()
            for nxt in adj.get(curr, []):
                if nxt not in visited and nxt in self.vertices:
                    visited.add(nxt)
                    result.append(nxt)
                    queue.append(nxt)

        if include_self:
            return [start_vertex] + result
        return result

    def get_upstream_vertices(self, target_vertex: str, include_self: bool = False) -> List[str]:
        """Compute upstream transitive closure leading to target_vertex via forward edges."""
        if target_vertex not in self.vertices:
            return []

        reverse_adj: Dict[str, List[str]] = defaultdict(list)
        for e in self.edges.values():
            if not e.is_reflexive:
                reverse_adj[e.output_vertex].append(e.input_vertex)

        visited: set[str] = set()
        queue: deque[str] = deque([target_vertex])
        result: List[str] = []

        while queue:
            curr = queue.popleft()
            for prev in reverse_adj.get(curr, []):
                if prev not in visited and prev in self.vertices:
                    visited.add(prev)
                    result.append(prev)
                    queue.append(prev)

        if include_self:
            return [target_vertex] + result
        return result

    def get_node_predecessors(self, vertex_name: str) -> List[str]:
        """Get names of immediate upstream vertices with forward edges targeting vertex_name."""
        if vertex_name not in self.vertices:
            return []
        preds = {
            e.input_vertex
            for e in self.edges.values()
            if not e.is_reflexive and e.output_vertex == vertex_name and e.input_vertex in self.vertices
        }
        return sorted(preds)

    def get_node_successors(self, vertex_name: str) -> List[str]:
        """Get names of immediate downstream vertices reachable from vertex_name via forward edges."""
        if vertex_name not in self.vertices:
            return []
        succs = {
            e.output_vertex
            for e in self.edges.values()
            if not e.is_reflexive and e.input_vertex == vertex_name and e.output_vertex in self.vertices
        }
        return sorted(succs)

    def get_node_relationships(self, vertex_name: str) -> Dict[str, Any]:
        """Get comprehensive relationship details and topology attributes for a vertex."""
        if vertex_name not in self.vertices:
            raise KeyError(f"Vertex '{vertex_name}' not found in graph")

        v = self.vertices[vertex_name]
        node_meta = self.get_loaded_node(vertex_name) or {}
        preds = self.get_node_predecessors(vertex_name)
        succs = self.get_node_successors(vertex_name)

        in_edges = [
            {
                "edge_id": e.id,
                "type": e.type,
                "source": e.input_vertex,
                "target": e.output_vertex,
                "priority": e.priority,
            }
            for e in self.get_incoming_edges(vertex_name)
        ]
        out_edges = [
            {
                "edge_id": e.id,
                "type": e.type,
                "source": e.input_vertex,
                "target": e.output_vertex,
                "priority": e.priority,
            }
            for e in self.get_outgoing_edges(vertex_name)
            if not e.is_reflexive
        ]
        reflexive_edges = [
            {
                "edge_id": e.id,
                "type": e.type,
                "trigger_state": getattr(e, "trigger_state", None),
                "target_state": getattr(e, "target_state", None),
                "max_retries": getattr(e, "max_retries", 3),
            }
            for e in self.get_reflexive_edges(vertex_name)
        ]

        return {
            "vertex_name": vertex_name,
            "session_id": v.session_id,
            "state": v.state,
            "attributes": list(v.attributes),
            "source": node_meta.get("source", "memory"),
            "loaded_at": node_meta.get("loaded_at", ""),
            "tier": self.node_tiers.get(vertex_name, 0),
            "is_orphan": self.is_orphan(vertex_name),
            "is_active": self.is_vertex_active(vertex_name),
            "in_degree": len(preds),
            "out_degree": len(succs),
            "predecessors": preds,
            "successors": succs,
            "incoming_edges": in_edges,
            "outgoing_edges": out_edges,
            "reflexive_edges": reflexive_edges,
            "downstream_closure": self.get_downstream_vertices(vertex_name, include_self=False),
            "upstream_closure": self.get_upstream_vertices(vertex_name, include_self=False),
        }

    def get_graph_relationships(self) -> Dict[str, Any]:
        """Get full graph relationship matrix and topology summary."""
        node_rels = {name: self.get_node_relationships(name) for name in self.vertices}
        roots = [name for name, rel in node_rels.items() if rel["in_degree"] == 0 and not rel["is_orphan"]]
        sinks = [name for name, rel in node_rels.items() if rel["out_degree"] == 0 and not rel["is_orphan"]]
        orphans = [name for name, rel in node_rels.items() if rel["is_orphan"]]

        edge_list = [
            {
                "id": e.id,
                "type": e.type,
                "input_vertex": e.input_vertex,
                "output_vertex": e.output_vertex,
                "is_reflexive": e.is_reflexive,
                "tier": self.edge_tiers.get(e.id, 0),
            }
            for e in self.edges.values()
        ]

        return {
            "session_id": self.session_id,
            "graph_name": self.name,
            "total_nodes": len(self.vertices),
            "total_edges": len(self.edges),
            "loaded_nodes": self.list_loaded_nodes(),
            "roots": sorted(roots),
            "sinks": sorted(sinks),
            "orphans": sorted(orphans),
            "active_nodes": sorted(self.get_active_vertices()),
            "inactive_nodes": sorted(self.get_inactive_vertices()),
            "adjacency_list": {name: rel["successors"] for name, rel in node_rels.items()},
            "reverse_adjacency_list": {name: rel["predecessors"] for name, rel in node_rels.items()},
            "nodes": node_rels,
            "edges": edge_list,
        }

    @property
    def is_valid(self) -> bool:
        """Check if graph references and bindings are structurally valid."""
        try:
            self.validate(strict_dag=False)
            return True
        except Exception:
            return False

    @property
    def has_cycle(self) -> bool:
        """Check if forward edges in graph contain a cycle."""
        try:
            self.detect_cycles_and_order(strict_dag=True)
            return False
        except Exception:
            return True

    def dump(
        self,
        path: Optional[Union[str, Path]] = None,
        indent: int = 2,
    ) -> Dict[str, Any]:
        """Serialize complete graph structure, components, and topology relationships.

        Args:
            path: Optional filesystem path to dump JSON representation.
            indent: JSON indentation spacing if path is provided.

        Returns:
            Dict containing serialized graph state.
        """
        valid = self.is_valid
        cycle = self.has_cycle
        data: Dict[str, Any] = {
            "version": "4.0",
            "session_id": self.session_id,
            "name": self.name,
            "metadata": dict(self.metadata),
            "vertices": [v.to_dict() for v in self.vertices.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "loaded_nodes": self.list_loaded_nodes(),
            "edge_tiers": dict(self.edge_tiers),
            "is_valid": valid,
            "has_cycle": cycle,
            "relationships": self.get_graph_relationships(),
        }
        if path:
            out_path = Path(path).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
        return data

    def to_dict(self) -> Dict[str, Any]:
        """Return complete dictionary representation of the graph."""
        return self.dump()


    def reset_affected_vertices(
        self,
        reenter_vertex: str,
        reset_state: str = VertexStateV4.TODO.value,
        clear_content: bool = False,
        store: Optional[VertexStoreV4] = None,
        session_id: Optional[str] = None,
    ) -> List[str]:
        """Reset all downstream vertices affected by re-entry into reenter_vertex.

        Resets their execution state (default: 'todo') and optionally clears content,
        enabling upstream-downstream handshakes to re-trigger.
        """
        downstream = self.get_downstream_vertices(reenter_vertex, include_self=False)
        target_session = session_id or self.session_id

        for v_name in downstream:
            v = self.vertices.get(v_name)
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

    def reenter_vertex(
        self,
        vertex_name: str,
        new_content: Optional[str] = None,
        reset_state: str = VertexStateV4.TODO.value,
        clear_content: bool = False,
        store: Optional[VertexStoreV4] = None,
        session_id: Optional[str] = None,
    ) -> List[str]:
        """Re-enter a vertex for re-execution and reset all affected downstream vertices."""
        if vertex_name not in self.vertices:
            raise KeyError(f"Vertex '{vertex_name}' not found in graph")

        target_session = session_id or self.session_id
        v = self.vertices[vertex_name]
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

        affected = self.reset_affected_vertices(
            reenter_vertex=vertex_name,
            reset_state=reset_state,
            clear_content=clear_content,
            store=store,
            session_id=target_session,
        )
        return affected

    def splice_subgraph(
        self,
        target_vertex_name: str,
        subgraph: GraphV4,
        name_prefix: Optional[str] = None,
        entry_vertex_name: Optional[str] = None,
        exit_vertex_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Splice (inline) a subgraph in place of an existing vertex.

        Replaces target_vertex_name with the nodes and edges of subgraph:
        - Incoming edges to target_vertex_name are rewired to subgraph entry vertices.
        - Outgoing edges from target_vertex_name are rewired from subgraph exit vertices.
        - The original target_vertex_name is removed.
        - DAG validation and tier calculation are performed on the updated topology.

        Returns:
            Dict containing inserted_vertices, inserted_edges, and rewired_edges.
        """
        if target_vertex_name not in self.vertices:
            raise KeyError(f"Target vertex '{target_vertex_name}' not found in graph")

        prefix = f"{name_prefix}_" if name_prefix else ""

        name_map: Dict[str, str] = {
            old_v: f"{prefix}{old_v}" if prefix else old_v
            for old_v in subgraph.vertices
        }

        for old_v, new_v in name_map.items():
            if new_v in self.vertices and new_v != target_vertex_name:
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
                session_id=self.session_id,
                name=new_name,
                content=v.content,
                attributes=list(v.attributes),
                state=v.state,
                processed_count=v.processed_count,
            )
            self.add_vertex(cloned_v, source=f"splice:{target_vertex_name}")
            inserted_vertices.append(new_name)

        # 2. Insert internal edges
        inserted_edges: List[str] = []
        for e in subgraph.edges.values():
            e_cfg = e.to_dict()
            new_eid = f"{prefix}{e.id}" if prefix else e.id
            if new_eid in self.edges:
                new_eid = f"{new_eid}_sub_{target_vertex_name}"
            e_cfg["id"] = new_eid
            e_cfg["input_vertex"] = name_map.get(e.input_vertex, e.input_vertex)
            e_cfg["output_vertex"] = name_map.get(e.output_vertex, e.output_vertex)
            cloned_edge = EdgeV4.from_config(e_cfg)
            self.add_edge(cloned_edge)
            inserted_edges.append(cloned_edge.id)

        # 3. Rewire incident edges
        rewired_edges: List[str] = []
        primary_entry = entry_nodes[0] if entry_nodes else None
        primary_exit = exit_nodes[0] if exit_nodes else None

        for e in list(self.edges.values()):
            if e.output_vertex == target_vertex_name:
                if primary_entry:
                    e.output_vertex = primary_entry
                    rewired_edges.append(e.id)
                    for extra_entry in entry_nodes[1:]:
                        extra_cfg = e.to_dict()
                        extra_cfg["id"] = f"{e.id}_fan_{extra_entry}"
                        extra_cfg["output_vertex"] = extra_entry
                        extra_edge = EdgeV4.from_config(extra_cfg)
                        self.add_edge(extra_edge)
                        inserted_edges.append(extra_edge.id)

        for e in list(self.edges.values()):
            if e.input_vertex == target_vertex_name:
                if primary_exit:
                    e.input_vertex = primary_exit
                    rewired_edges.append(e.id)
                    for extra_exit in exit_nodes[1:]:
                        extra_cfg = e.to_dict()
                        extra_cfg["id"] = f"{e.id}_fan_{extra_exit}"
                        extra_cfg["input_vertex"] = extra_exit
                        extra_edge = EdgeV4.from_config(extra_cfg)
                        self.add_edge(extra_edge)
                        inserted_edges.append(extra_edge.id)

        # 4. Remove original target vertex
        self.delete_vertex(target_vertex_name)

        # 5. Revalidate endpoints and compute tiers softly
        self.validate(strict_dag=False)

        return {
            "spliced_vertex": target_vertex_name,
            "entry_vertices": entry_nodes,
            "exit_vertices": exit_nodes,
            "inserted_vertices": inserted_vertices,
            "inserted_edges": inserted_edges,
            "rewired_edges": rewired_edges,
        }

    def insert_subgraph(
        self,
        subgraph: GraphV4,
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
            if new_v in self.vertices:
                raise ValueError(f"Subgraph vertex '{new_v}' collides with existing graph vertex.")

        inserted_vertices: List[str] = []
        for old_name, v in subgraph.vertices.items():
            new_name = name_map[old_name]
            cloned_v = VertexRecordV4(
                id=0,
                session_id=self.session_id,
                name=new_name,
                content=v.content,
                attributes=list(v.attributes),
                state=v.state,
                processed_count=v.processed_count,
            )
            self.add_vertex(cloned_v, source="subgraph_insert")
            inserted_vertices.append(new_name)

        inserted_edges: List[str] = []
        for e in subgraph.edges.values():
            e_cfg = e.to_dict()
            new_eid = f"{prefix}{e.id}" if prefix else e.id
            if new_eid in self.edges:
                new_eid = f"{new_eid}_sub"
            e_cfg["id"] = new_eid
            e_cfg["input_vertex"] = name_map.get(e.input_vertex, e.input_vertex)
            e_cfg["output_vertex"] = name_map.get(e.output_vertex, e.output_vertex)
            cloned_edge = EdgeV4.from_config(e_cfg)
            self.add_edge(cloned_edge)
            inserted_edges.append(cloned_edge.id)

        for parent_v, sub_v in (incoming_bindings or {}).items():
            mapped_sub = name_map.get(sub_v, sub_v)
            bridge_id = f"bridge_in_{parent_v}_{mapped_sub}"
            bridge_edge = CodeEdgeV4(edge_id=bridge_id, input_vertex=parent_v, output_vertex=mapped_sub)
            self.add_edge(bridge_edge)
            inserted_edges.append(bridge_id)

        for sub_v, parent_v in (outgoing_bindings or {}).items():
            mapped_sub = name_map.get(sub_v, sub_v)
            bridge_id = f"bridge_out_{mapped_sub}_{parent_v}"
            bridge_edge = CodeEdgeV4(edge_id=bridge_id, input_vertex=mapped_sub, output_vertex=parent_v)
            self.add_edge(bridge_edge)
            inserted_edges.append(bridge_id)

        # Validate structural bindings without enforcing strict DAG conclusion at definition time
        self.validate(strict_dag=False)

        return {
            "inserted_vertices": inserted_vertices,
            "inserted_edges": inserted_edges,
            "incoming_bindings": incoming_bindings or {},
            "outgoing_bindings": outgoing_bindings or {},
        }

    def add_subgraph(
        self,
        subgraph: Union[GraphV4, Dict[str, Any], str, Path],
        name_prefix: Optional[str] = None,
        connections: Optional[List[Dict[str, Any]]] = None,
        incoming_bindings: Optional[Dict[str, str]] = None,
        outgoing_bindings: Optional[Dict[str, str]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add and join a sub-graph into this graph.

        Merges all vertices and edges from subgraph into self:
        - Prefix is applied to subgraph vertex names if specified to avoid collision.
        - Provenance is recorded in loaded_nodes.
        - Optional boundary connections or bindings bridge parent and sub-graph nodes.
        - Revalidates endpoints and computes tiers softly.

        Args:
            subgraph: GraphV4 instance, dict definition, or manifest path.
            name_prefix: Optional prefix for vertex and edge names.
            connections: Explicit edge connections between parent and sub-graph.
            incoming_bindings: Map of parent_vertex -> sub_vertex (creates bridge edge).
            outgoing_bindings: Map of sub_vertex -> parent_vertex (creates bridge edge).
            source: Source identifier for loaded_nodes tracking.

        Returns:
            Dict with added_vertices, added_edges, and name_mapping.
        """
        if isinstance(subgraph, (str, Path)):
            subgraph_obj = DiscreteGraphLoaderV4.load_from_manifest(subgraph)
        elif isinstance(subgraph, dict):
            subgraph_obj = DiscreteGraphLoaderV4.load_from_dict(subgraph)
        elif isinstance(subgraph, GraphV4):
            subgraph_obj = subgraph
        else:
            raise TypeError(f"Unsupported subgraph type: {type(subgraph).__name__}")

        prefix = f"{name_prefix}_" if name_prefix else ""
        name_map: Dict[str, str] = {
            old_v: f"{prefix}{old_v}" if prefix else old_v
            for old_v in subgraph_obj.vertices
        }


        # Validate no unintentional vertex collisions
        for old_v, new_v in name_map.items():
            if new_v in self.vertices:
                raise ValueError(
                    f"Subgraph vertex '{new_v}' collides with existing vertex in graph. Provide a distinct name_prefix."
                )

        # 1. Add vertices with provenance
        added_vertices: List[str] = []
        node_source = source or f"subgraph:{subgraph_obj.name}"
        for old_name, v in subgraph_obj.vertices.items():
            new_name = name_map[old_name]
            cloned_v = VertexRecordV4(
                id=0,
                session_id=self.session_id,
                name=new_name,
                content=v.content,
                attributes=list(v.attributes),
                state=v.state,
                processed_count=v.processed_count,
            )
            self.add_vertex(cloned_v, source=node_source)
            added_vertices.append(new_name)

        # 2. Add internal edges
        added_edges: List[str] = []
        for e in subgraph_obj.edges.values():
            e_cfg = e.to_dict()
            new_eid = f"{prefix}{e.id}" if prefix else e.id
            if new_eid in self.edges:
                new_eid = f"{new_eid}_sub_{subgraph_obj.session_id}"
            e_cfg["id"] = new_eid
            e_cfg["input_vertex"] = name_map.get(e.input_vertex, e.input_vertex)
            e_cfg["output_vertex"] = name_map.get(e.output_vertex, e.output_vertex)
            cloned_edge = EdgeV4.from_config(e_cfg)
            self.add_edge(cloned_edge)
            added_edges.append(cloned_edge.id)

        # 3. Add incoming bindings
        for parent_v, sub_v in (incoming_bindings or {}).items():
            mapped_sub = name_map.get(sub_v, sub_v)
            bridge_id = f"bridge_in_{parent_v}_{mapped_sub}"
            bridge_edge = CodeEdgeV4(edge_id=bridge_id, input_vertex=parent_v, output_vertex=mapped_sub)
            self.add_edge(bridge_edge)
            added_edges.append(bridge_id)

        # 4. Add outgoing bindings
        for sub_v, parent_v in (outgoing_bindings or {}).items():
            mapped_sub = name_map.get(sub_v, sub_v)
            bridge_id = f"bridge_out_{mapped_sub}_{parent_v}"
            bridge_edge = CodeEdgeV4(edge_id=bridge_id, input_vertex=mapped_sub, output_vertex=parent_v)
            self.add_edge(bridge_edge)
            added_edges.append(bridge_id)

        # 5. Add explicit connections
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
            self.add_edge(conn_edge)
            added_edges.append(conn_edge.id)

        # 6. Revalidate softly
        self.validate(strict_dag=False)

        return {
            "subgraph_name": subgraph_obj.name,
            "name_prefix": name_prefix,
            "added_vertices": added_vertices,
            "added_edges": added_edges,
            "name_mapping": name_map,
        }

    def detect_cycles_and_order(self, strict_dag: bool = True) -> List[str]:
        """Detect cycles using state coloring DFS and return topological ordering.

        Uses VertexStateV4 (WHITE, GRAY, BLACK) directly on vertex states:
            - WHITE: Unvisited node.
            - GRAY: Currently visiting (ancestor in active DFS recursion path).
            - BLACK: Visited and fully explored (verified acyclic).

        Args:
            strict_dag: If True, raises GraphTopologyError when cycles are detected.
                        If False, breaks cycles gracefully to permit cyclic execution and dynamic graphs.

        Returns:
            List[str]: Vertices ordered in forward topological sequence.

        Raises:
            GraphTopologyError: If circular dependencies are detected in forward edges and strict_dag is True.
        """
        adj: Dict[str, List[str]] = {v: [] for v in self.vertices}
        for edge in self.edges.values():
            if not edge.is_reflexive:
                if edge.input_vertex in adj:
                    adj[edge.input_vertex].append(edge.output_vertex)

        states: Dict[str, VertexStateV4] = {v: VertexStateV4.WHITE for v in self.vertices}
        current_path: List[str] = []
        post_order: List[str] = []

        def dfs(node: str) -> None:
            states[node] = VertexStateV4.GRAY
            if node in self.vertices and self.vertices[node].state in (
                VertexStateV4.IDLE.value,
                VertexStateV4.WHITE.value,
                VertexStateV4.GRAY.value,
            ):
                self.vertices[node].state = VertexStateV4.GRAY.value
            current_path.append(node)

            for neighbor in adj.get(node, []):
                state = states.get(neighbor)
                if state == VertexStateV4.GRAY:
                    cycle_start_idx = current_path.index(neighbor)
                    cycle_path = current_path[cycle_start_idx:] + [neighbor]
                    cycle_str = " -> ".join(cycle_path)
                    self.node_states = dict(states)
                    if strict_dag:
                        raise GraphTopologyError(
                            f"Cycle detected in forward edges: {cycle_str}. "
                            "Forward edges must form a directed acyclic graph (DAG)."
                        )
                    else:
                        logger.info("Cycle detected at definition time: %s (will be resolved at runtime)", cycle_str)
                        continue
                elif state == VertexStateV4.WHITE:
                    dfs(neighbor)
                elif state is None:
                    self.node_states = dict(states)
                    raise GraphTopologyError(
                        f"Forward edge targeting non-existent vertex '{neighbor}'"
                    )

            current_path.pop()
            states[node] = VertexStateV4.BLACK
            if node in self.vertices and self.vertices[node].state in (
                VertexStateV4.IDLE.value,
                VertexStateV4.WHITE.value,
                VertexStateV4.GRAY.value,
            ):
                self.vertices[node].state = VertexStateV4.BLACK.value
            post_order.append(node)

        for v in sorted(self.vertices.keys()):
            if states[v] == VertexStateV4.WHITE:
                dfs(v)

        self.node_states = dict(states)
        return list(reversed(post_order))

    def validate(self, strict_dag: bool = True) -> None:
        """Validate vertex bindings and DAG constraints using state coloring.

        Args:
            strict_dag: If True, enforces strictly acyclic forward edges.
                        If False, validates endpoint references and soft-computes tiers.
        """
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

        # Validate DAG on forward edges using state coloring DFS
        self.detect_cycles_and_order(strict_dag=strict_dag)
        self.compute_dag_tiers(strict_dag=strict_dag)

    def compute_dag_tiers(self, strict_dag: bool = False) -> Dict[str, int]:
        """Compute topological tier depth for each edge to guide DAG execution order.

        Tier 0 represents edges originating directly from start or entry vertices.
        Higher tiers depend on outputs from earlier tiers.
        Reflexive edges are assigned Tier -1 (highest priority during recovery).
        """
        try:
            topo_order = self.detect_cycles_and_order(strict_dag=strict_dag)
        except GraphTopologyError:
            if strict_dag:
                raise
            topo_order = list(self.vertices.keys())


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
                if self.is_vertex_active(v_name):
                    node_tier[v_name] = 0
                else:
                    node_tier[v_name] = -1

        # Propagate tiers along topological order in a single pass
        adj_edges: Dict[str, List[EdgeV4]] = defaultdict(list)
        for e in self.edges.values():
            if not e.is_reflexive:
                adj_edges[e.input_vertex].append(e)

        for u in topo_order:
            if not self.is_vertex_active(u):
                continue
            curr_tier = node_tier.get(u, 0)
            for e in adj_edges.get(u, []):
                out_name = e.output_vertex
                expected = curr_tier + 1
                if node_tier.get(out_name, -1) < expected:
                    node_tier[out_name] = expected

        # Assign tier to each edge
        tiers: Dict[str, int] = {}
        for e in self.edges.values():
            if e.is_reflexive:
                tiers[e.id] = -1
            else:
                tiers[e.id] = node_tier.get(e.input_vertex, 0)

        self.node_tiers = dict(node_tier)
        self.edge_tiers = tiers
        return tiers

    @classmethod
    def load_from_store(cls, store: VertexStoreV4, session_id: str, name: str = "v4_graph") -> GraphV4:
        """Hydrate a GraphV4 instance from SQLite persisted vertices and edges."""
        graph = cls(session_id=session_id, name=name)
        for v in store.list_vertices(session_id):
            graph.add_vertex(v, source="sqlite_store")

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
            elif edge_type in ("llm", "llm_chat", "llm_generate", "llm_process", "llm_callable"):
                cls_map = {
                    "llm": LLMEdgeV4,
                    "llm_chat": ChatLLMEdgeV4,
                    "llm_generate": GenerateLLMEdgeV4,
                    "llm_process": ProcessLLMEdgeV4,
                    "llm_callable": CallableLLMEdgeV4,
                }
                target_cls = cls_map.get(edge_type, LLMEdgeV4)
                edge = target_cls(
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
        except Exception as e:
            logger.warning("Failed to compute DAG tiers during load_from_store: %s", e)
        return graph


class DiscreteGraphLoaderV4:
    """Loads discrete vertex and edge JSON components specified by a master manifest."""

    @classmethod
    def load_from_dict(
        cls,
        data: Dict[str, Any],
        base_dir: Optional[Union[str, Path]] = None,
        override_session_id: Optional[str] = None,
    ) -> GraphV4:
        """Construct GraphV4 from in-memory dictionary definition."""
        session_id = override_session_id or data.get("session_id", "default_session")
        name = (data.get("metadata") or {}).get("name", "subgraph")
        meta = dict(data.get("metadata") or {})
        b_dir = Path(base_dir).resolve() if base_dir else None
        if b_dir:
            meta.setdefault("base_dir", str(b_dir))
        graph = GraphV4(session_id=session_id, name=name, metadata=meta)

        def _resolve_item_path(item: Union[str, Path]) -> Path:
            p = Path(item)
            if b_dir and (b_dir / p).exists():
                return (b_dir / p).resolve()
            if p.is_absolute() and p.exists():
                return p.resolve()
            if p.exists():
                return p.resolve()
            if (Path(_REPO_ROOT) / p).exists():
                return (Path(_REPO_ROOT) / p).resolve()
            target = (b_dir / p) if b_dir else p
            raise FileNotFoundError(f"Configuration path not found: {target}")

        # Load discrete vertices
        for v_item in data.get("vertices", []):
            if isinstance(v_item, (str, Path)):
                v_path = _resolve_item_path(v_item)
                if v_path.is_dir():
                    for vf in sorted(v_path.glob("*.json")):
                        graph.add_vertex(vf)
                else:
                    graph.add_vertex(v_path)
            elif isinstance(v_item, dict):
                graph.add_vertex(v_item, source="manifest_dict")

        # Load discrete edges
        for e_item in data.get("edges", []):
            if isinstance(e_item, (str, Path)):
                e_path = _resolve_item_path(e_item)
                if e_path.is_dir():
                    for ef in sorted(e_path.glob("*.json")):
                        graph.add_edge(ef)
                else:
                    graph.add_edge(e_path)
            elif isinstance(e_item, dict):
                edge_instance = EdgeV4.from_config(e_item, base_dir=str(b_dir) if b_dir else None)
                graph.add_edge(edge_instance)

        # Validate structural bindings without enforcing strict DAG conclusion at definition time
        graph.validate(strict_dag=False)
        return graph

    @classmethod
    def load_from_directory(
        cls,
        directory_path: Union[str, Path],
        override_session_id: Optional[str] = None,
    ) -> GraphV4:
        """Construct GraphV4 by loading all vertex and edge configurations in a directory.

        Supports:
        1. Manifest files inside the directory (graph.json or manifest.json).
        2. Subdirectories: vertices/ (or vertex/, nodes/) and edges/ (or edge/).
        3. Flat directory with individual vertex and edge JSON files.
        """
        d_p = Path(directory_path)
        if not d_p.is_absolute() and not d_p.exists():
            cand = Path(_REPO_ROOT) / d_p
            if cand.exists():
                d_p = cand
        if not d_p.is_dir():
            raise FileNotFoundError(f"Graph directory not found: {directory_path}")
        d_p = d_p.resolve()

        # Check for standard manifest file in directory
        for m_name in ("graph.json", "manifest.json"):
            cand_m = d_p / m_name
            if cand_m.is_file():
                return cls.load_from_manifest(cand_m, override_session_id=override_session_id)

        session_id = override_session_id or d_p.name
        graph = GraphV4(session_id=session_id, name=d_p.name, metadata={"base_dir": str(d_p)})

        # Check for vertices subdir
        has_subdirs = False
        for v_sub in ("vertices", "vertex", "nodes"):
            sub_dir = d_p / v_sub
            if sub_dir.is_dir():
                has_subdirs = True
                for vf in sorted(sub_dir.glob("*.json")):
                    graph.add_vertex(vf)
                break

        # Check for edges subdir
        for e_sub in ("edges", "edge"):
            sub_dir = d_p / e_sub
            if sub_dir.is_dir():
                has_subdirs = True
                for ef in sorted(sub_dir.glob("*.json")):
                    graph.add_edge(ef)
                break

        # If not already populated via subdirs, or to pick up top-level loose configs
        for jf in sorted(d_p.glob("*.json")):
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Determine if item is an edge config
                if (
                    any(k in item for k in ("input_vertex", "output_vertex", "edge_id", "source", "destination"))
                    or item.get("type") in ("code", "reflexive", "llm", "llm_chat", "llm_generate", "llm_process", "llm_callable")
                ):
                    edge_instance = EdgeV4.from_config(item, base_dir=str(d_p))
                    graph.add_edge(edge_instance)
                elif "name" in item:
                    graph.add_vertex(item, source=f"file:{jf}")

        graph.validate(strict_dag=False)
        return graph

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

        return cls.load_from_dict(
            data=data,
            base_dir=path.parent,
            override_session_id=override_session_id or data.get("session_id", path.stem),
        )

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
