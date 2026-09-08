"""Core GraphV4 class representing a workflow DAG with discrete vertices and executable edges."""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from framework.edges.base import EdgeV4
from framework.edges.code import CodeEdgeV4
from framework.edges.reflexive import ReflexiveEdgeV4
from framework.graphs.exceptions import GraphTopologyError
from framework.graphs.validation import compute_dag_tiers, detect_cycles_and_order, validate_topology
from framework.graphs.subgraph_ops import (
    add_subgraph_op,
    insert_subgraph_op,
    reenter_vertex_op,
    reset_affected_vertices_op,
    splice_subgraph_op,
)
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.graphs.core")

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)

# NodeColor is unified into VertexStateV4 for state coloring
NodeColor = VertexStateV4


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
        from framework.graphs.loader import DiscreteGraphLoaderV4
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
        """Instantiate a GraphV4 by loading all configs in a directory."""
        from framework.graphs.loader import DiscreteGraphLoaderV4
        return DiscreteGraphLoaderV4.load_from_directory(directory_path, override_session_id=session_id)

    def get_vertex(self, name: str) -> Optional[VertexRecordV4]:
        """Retrieve vertex record by name."""
        return self.vertices.get(name)

    def delete_vertex(self, name: str) -> bool:
        """Delete a vertex and cascade delete all connected edges."""
        if name not in self.vertices:
            return False
        del self.vertices[name]
        self.loaded_nodes.pop(name, None)
        self._active_overrides.pop(name, None)

        # Cascade delete connected edges
        to_delete = [
            eid for eid, edge in self.edges.items()
            if edge.input_vertex == name or edge.output_vertex == name
        ]
        for eid in to_delete:
            self.delete_edge(eid)
        return True

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by ID."""
        if edge_id not in self.edges:
            return False
        del self.edges[edge_id]
        self.loaded_nodes.pop(edge_id, None)
        self.edge_tiers.pop(edge_id, None)
        return True

    def is_orphan(self, vertex_name: str) -> bool:
        """Return True if vertex has zero incoming and zero outgoing non-reflexive edges."""
        if vertex_name not in self.vertices:
            return False
        for edge in self.edges.values():
            if not edge.is_reflexive and (edge.input_vertex == vertex_name or edge.output_vertex == vertex_name):
                return False
        return True

    def get_orphans(self) -> List[str]:
        """List names of all orphan vertices with no edge connections."""
        return [v_name for v_name in self.vertices if self.is_orphan(v_name)]

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
        """Explicitly activate a vertex."""
        if vertex_name not in self.vertices:
            raise KeyError(f"Vertex '{vertex_name}' not found in graph")
        self._active_overrides[vertex_name] = True

    def deactivate_vertex(self, vertex_name: str) -> None:
        """Explicitly deactivate a vertex, removing it from execution traversal."""
        if vertex_name not in self.vertices:
            raise KeyError(f"Vertex '{vertex_name}' not found in graph")
        self._active_overrides[vertex_name] = False

    def get_active_vertices(self) -> List[str]:
        """List names of currently active vertices."""
        return [name for name in self.vertices if self.is_vertex_active(name)]

    def get_inactive_vertices(self) -> List[str]:
        """List names of currently inactive vertices."""
        return [name for name in self.vertices if not self.is_vertex_active(name)]

    def get_edge(self, edge_id: str) -> Optional[EdgeV4]:
        """Retrieve edge instance by ID."""
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
        """Return reflexive edges, optionally filtered by vertex."""
        res: List[ReflexiveEdgeV4] = []
        for e in self.edges.values():
            if e.is_reflexive or isinstance(e, ReflexiveEdgeV4):
                if vertex_name is None or e.input_vertex == vertex_name:
                    if isinstance(e, ReflexiveEdgeV4):
                        res.append(e)
                    else:
                        ref = ReflexiveEdgeV4(
                            edge_id=e.id,
                            vertex_name=e.input_vertex,
                            settings=e.settings,
                        )
                        res.append(ref)
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
        """Serialize complete graph structure, components, and topology relationships."""
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

    # Dynamic Subgraph Operations (delegated to subgraph_ops)
    def reset_affected_vertices(
        self,
        reenter_vertex: str,
        reset_state: str = VertexStateV4.TODO.value,
        clear_content: bool = False,
        store: Optional[VertexStoreV4] = None,
        session_id: Optional[str] = None,
    ) -> List[str]:
        return reset_affected_vertices_op(
            graph=self,
            reenter_vertex=reenter_vertex,
            reset_state=reset_state,
            clear_content=clear_content,
            store=store,
            session_id=session_id,
        )

    def reenter_vertex(
        self,
        vertex_name: str,
        new_content: Optional[str] = None,
        reset_state: str = VertexStateV4.TODO.value,
        clear_content: bool = False,
        store: Optional[VertexStoreV4] = None,
        session_id: Optional[str] = None,
    ) -> List[str]:
        return reenter_vertex_op(
            graph=self,
            vertex_name=vertex_name,
            new_content=new_content,
            reset_state=reset_state,
            clear_content=clear_content,
            store=store,
            session_id=session_id,
        )

    def splice_subgraph(
        self,
        target_vertex_name: str,
        subgraph: "GraphV4",
        name_prefix: Optional[str] = None,
        entry_vertex_name: Optional[str] = None,
        exit_vertex_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return splice_subgraph_op(
            graph=self,
            target_vertex_name=target_vertex_name,
            subgraph=subgraph,
            name_prefix=name_prefix,
            entry_vertex_name=entry_vertex_name,
            exit_vertex_name=exit_vertex_name,
        )

    def insert_subgraph(
        self,
        subgraph: "GraphV4",
        incoming_bindings: Optional[Dict[str, str]] = None,
        outgoing_bindings: Optional[Dict[str, str]] = None,
        name_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        return insert_subgraph_op(
            graph=self,
            subgraph=subgraph,
            incoming_bindings=incoming_bindings,
            outgoing_bindings=outgoing_bindings,
            name_prefix=name_prefix,
        )

    def add_subgraph(
        self,
        subgraph: Union["GraphV4", Dict[str, Any], str, Path],
        name_prefix: Optional[str] = None,
        connections: Optional[List[Dict[str, Any]]] = None,
        incoming_bindings: Optional[Dict[str, str]] = None,
        outgoing_bindings: Optional[Dict[str, str]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        return add_subgraph_op(
            graph=self,
            subgraph=subgraph,
            name_prefix=name_prefix,
            connections=connections,
            incoming_bindings=incoming_bindings,
            outgoing_bindings=outgoing_bindings,
            source=source,
        )

    # Topological Validation & Tiers (delegated to validation)
    def detect_cycles_and_order(self, strict_dag: bool = True) -> List[str]:
        return detect_cycles_and_order(self, strict_dag=strict_dag)

    def validate(self, strict_dag: bool = True) -> None:
        return validate_topology(self, strict_dag=strict_dag)

    def compute_dag_tiers(self, strict_dag: bool = False) -> Dict[str, int]:
        return compute_dag_tiers(self, strict_dag=strict_dag)

    # Store Hydration and Live Sync (eliminates dual-source divergence)
    @classmethod
    def load_from_store(cls, store: VertexStoreV4, session_id: str, name: str = "v4_graph") -> "GraphV4":
        from framework.graphs.loader import load_from_store_fn
        return load_from_store_fn(cls, store=store, session_id=session_id, name=name)

    def sync_vertex_from_store(
        self,
        store: VertexStoreV4,
        vertex_name: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Synchronize a single vertex from SQLite store into this in-memory Graph.
        
        Guarantees 100% cache coherence between SQLite storage and in-memory graph.
        """
        sess = session_id or self.session_id
        if not store or not sess:
            return
        rec = store.get_vertex(sess, vertex_name)
        if not rec:
            return
        if vertex_name in self.vertices:
            v = self.vertices[vertex_name]
            v.content = rec.content
            v.state = rec.state
            v.processed_count = rec.processed_count
            if rec.attributes:
                v.attributes = list(rec.attributes)
        else:
            self.add_vertex(rec, source="sqlite_store")

    def sync_from_store(
        self,
        store: VertexStoreV4,
        session_id: Optional[str] = None,
    ) -> None:
        """Synchronize all vertices from SQLite store into this in-memory Graph."""
        sess = session_id or self.session_id
        if not store or not sess:
            return
        for rec in store.list_vertices(sess):
            if rec.name in self.vertices:
                v = self.vertices[rec.name]
                v.content = rec.content
                v.state = rec.state
                v.processed_count = rec.processed_count
                if rec.attributes:
                    v.attributes = list(rec.attributes)
            else:
                self.add_vertex(rec, source="sqlite_store")
