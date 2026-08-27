"""Graph module - Load and manage the computation graph from JSON.

A Graph is a network of Vertex nodes connected by Edge arrows.  It is loaded
from a JSON configuration and validated for referential integrity.  Pure DAGs
are enforced by default; cycles are permitted only when every back-edge carries
``max_iterations > 0``, enabling stateful self-correction loops.
"""

import json
import logging
import os
import inspect
from typing import Any, Dict, List, Optional, Set

from .vertex import Vertex
from .edge import Edge
from .script_loader import load_script

logger = logging.getLogger("vertex_edge_agent.graph")


class Graph:
    """Network of vertices and edges loaded from JSON configuration.

    JSON schema::

        {
          "metadata": { ... },
          "vertices": [
            {
              "id": "v1",
              "settings": {},
              "script": "path/to/script.py",   // optional
              "initial_data": [                 // optional
                {"data_id": "text", "tags": ["en"], "value": "hello"}
              ]
            }
          ],
          "edges": [
            {
              "id": "e1",
              "source": "v1",
              "destination": "v2",
              "channel": "text",
              "prompt": "Summarise this:",
              "model": "gemini-pro",
              "settings": {},
              "max_iterations": 3,             // optional — enables loop back
              "script": "path/to/edge_script.py"  // optional
            }
          ]
        }
    """

    def __init__(self):
        self.vertices: Dict[str, Vertex] = {}
        self.edges: Dict[str, Edge] = {}
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, json_path: str) -> "Graph":
        """Load graph from a JSON file."""
        logger.info("[Graph] Loading from %s", json_path)
        with open(json_path, "r") as fh:
            config = json.load(fh)
        # resolve script paths relative to the JSON file
        base_dir = os.path.dirname(os.path.abspath(json_path))
        return cls.from_dict(config, base_dir=base_dir)

    @classmethod
    def from_dict(cls, config: Dict, base_dir: Optional[str] = None) -> "Graph":
        """Build a graph from a configuration dict."""
        graph = cls()
        graph.metadata = config.get("metadata", {})
        base_dir = base_dir or os.getcwd()

        # --- vertices ---
        for vc in config.get("vertices", []):
            script = vc.get("script")
            if script and not os.path.isabs(script):
                script = os.path.join(base_dir, script)

            v_type = vc.get("type", "vertex")
            if v_type == "subgraph":
                from .subgraph import SubgraphVertex
                vertex_cls = SubgraphVertex
            else:
                vertex_cls = Vertex

            script_module = None
            if script:
                try:
                    script_module = load_script(script)
                    for name, obj in inspect.getmembers(script_module, inspect.isclass):
                        if issubclass(obj, Vertex) and obj not in (Vertex, SubgraphVertex):
                            vertex_cls = obj
                            break
                except Exception as exc:
                    logger.error(
                        "[Graph] Script load failed for vertex '%s': %s", vc["id"], exc
                    )

            vertex = vertex_cls(
                vertex_id=vc["id"],
                settings=vc.get("settings", {}),
                script_path=script,
                initial_data=vc.get("initial_data"),
            )

            if script_module:
                vertex.set_script_module(script_module)

            graph.vertices[vertex.id] = vertex

        # --- edges ---
        for ec in config.get("edges", []):
            script = ec.get("script")
            if script and not os.path.isabs(script):
                script = os.path.join(base_dir, script)

            edge_cls = Edge
            script_module = None
            if script:
                try:
                    script_module = load_script(script)
                    for name, obj in inspect.getmembers(script_module, inspect.isclass):
                        if issubclass(obj, Edge) and obj is not Edge:
                            edge_cls = obj
                            break
                except Exception as exc:
                    logger.error(
                        "[Graph] Script load failed for edge '%s': %s", ec["id"], exc
                    )

            edge = edge_cls(
                edge_id=ec["id"],
                source_id=ec["source"],
                destination_id=ec["destination"],
                channel=ec.get("channel", ec.get("data_id", "default")),
                prompt=ec.get("prompt", ""),
                model=ec.get("model", "default"),
                settings=ec.get("settings", {}),
                script_path=script,
                max_iterations=int(ec.get("max_iterations", 0)),
            )

            if script_module:
                edge.set_script_module(script_module)

            graph.edges[edge.id] = edge

            # register on vertices
            if edge.source_id in graph.vertices:
                graph.vertices[edge.source_id].outgoing_edges.append(edge.id)
            else:
                logger.error(
                    "[Graph] Edge '%s' references unknown source '%s'",
                    edge.id, edge.source_id,
                )

            if edge.destination_id in graph.vertices:
                dest = graph.vertices[edge.destination_id]
                dest.incoming_edges.append(edge.id)
                dest.required_input_count = len(dest.incoming_edges)
            else:
                logger.error(
                    "[Graph] Edge '%s' references unknown destination '%s'",
                    edge.id, edge.destination_id,
                )

        graph.validate()
        logger.info(
            "[Graph] Loaded %d vertices, %d edges",
            len(graph.vertices), len(graph.edges),
        )
        return graph

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self):
        """Validate referential integrity and cycle policy.

        Pure DAGs are always valid.  A cycle is permitted only when *every*
        back-edge in the cycle carries ``max_iterations > 0``; the back-edge
        limit is then propagated to the destination vertex's
        ``loop_incoming_edges`` mapping so the executor can enforce it.

        Raises:
            ValueError: If any referential error or unguarded cycle is found.
        """
        errors: List[str] = []

        for edge in self.edges.values():
            if edge.source_id not in self.vertices:
                errors.append(
                    f"Edge '{edge.id}': source '{edge.source_id}' not found"
                )
            if edge.destination_id not in self.vertices:
                errors.append(
                    f"Edge '{edge.id}': destination '{edge.destination_id}' not found"
                )

        # ── Cycle detection (DFS) — collect all back-edges ────────────
        visited: Set[str] = set()
        stack: Set[str] = set()
        back_edges: Set[str] = set()

        def _dfs(vid: str) -> None:
            visited.add(vid)
            stack.add(vid)
            for eid in self.vertices[vid].outgoing_edges:
                if eid not in self.edges:
                    continue
                nxt = self.edges[eid].destination_id
                if nxt not in self.vertices:
                    continue  # referential error caught above
                if nxt not in visited:
                    _dfs(nxt)
                elif nxt in stack:
                    back_edges.add(eid)  # this edge closes a cycle
            stack.discard(vid)

        for vid in self.vertices:
            if vid not in visited:
                _dfs(vid)

        # ── Policy: back-edges must be guarded by max_iterations ───────
        for eid in back_edges:
            edge = self.edges[eid]
            if edge.max_iterations <= 0:
                errors.append(
                    f"Graph contains an unguarded cycle via edge '{eid}' "
                    f"({edge.source_id} -> {edge.destination_id}). "
                    f"Add 'max_iterations' > 0 to this edge to enable stateful loops."
                )

        if errors:
            for e in errors:
                logger.error("[Graph] Validation: %s", e)
            raise ValueError(f"Graph validation failed: {'; '.join(errors)}")

        # ── Propagate loop metadata to destination vertices ────────────
        for eid in back_edges:
            edge = self.edges[eid]
            dest = self.vertices[edge.destination_id]
            dest.loop_incoming_edges[eid] = edge.max_iterations
            logger.info(
                "[Graph] Loop edge '%s' registered on vertex '%s' (max_iterations=%d)",
                eid, dest.id, edge.max_iterations,
            )

        logger.info("[Graph] Validation passed ✓")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_source_vertices(self) -> List[Vertex]:
        """Vertices with no incoming edges (entry points)."""
        return [v for v in self.vertices.values() if v.is_source()]

    def get_sink_vertices(self) -> List[Vertex]:
        """Vertices with no outgoing edges (exit points)."""
        return [v for v in self.vertices.values() if v.is_sink()]

    def get_outgoing_edges(self, vertex_id: str) -> List[Edge]:
        return [self.edges[eid] for eid in self.vertices[vertex_id].outgoing_edges]

    def get_incoming_edges(self, vertex_id: str) -> List[Edge]:
        return [self.edges[eid] for eid in self.vertices[vertex_id].incoming_edges]

    def __repr__(self):
        return f"Graph(V={len(self.vertices)}, E={len(self.edges)})"
