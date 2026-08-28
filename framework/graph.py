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
from .utils.script_loader import load_script
from .utils.schema import SchemaMismatchError

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
        """Load graph from a JSON file (supports // and /* */ comments)."""
        logger.debug("[Graph] Loading from %s", json_path)
        import re
        
        def _strip_comments(text: str) -> str:
            # Match JSON strings OR comments
            pattern = r'("(?:\\.|[^"\\])*")|(/\*.*?\*/|//[^\r\n]*)'
            regex = re.compile(pattern, re.DOTALL)
            def replacer(match):
                if match.group(2) is not None:
                    return "" # It's a comment, strip it
                return match.group(1) # It's a string, keep it
            return regex.sub(replacer, text)

        with open(json_path, "r", encoding="utf-8") as fh:
            raw_content = fh.read()
            
        clean_content = _strip_comments(raw_content)
        config = json.loads(clean_content)
        
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
            script = vc.get("pipeline")
            entrypoint = None
            if script and ":" in script:
                script, entrypoint = script.split(":", 1)
            if script and not os.path.isabs(script):
                script = os.path.join(base_dir, script)

            v_type = vc.get("type", "vertex")
            if v_type == "subgraph":
                from .subgraph import SubgraphVertex
                vertex_base_cls = SubgraphVertex
            else:
                vertex_base_cls = Vertex

            pipeline_module = None
            if script:
                from .utils.script_loader import load_class_from_script
                try:
                    vertex_cls = load_class_from_script(script, Vertex, vertex_base_cls)
                    pipeline_module = load_script(script)  # Need the module for hooks
                    if entrypoint:
                        pipeline_module = getattr(pipeline_module, entrypoint)
                except Exception as exc:
                    logger.error("[Graph] Pipeline script load failed for vertex '%s': %s", vc["id"], exc)
                    raise RuntimeError(f"Pipeline script load failed for vertex '{vc['id']}': {exc}") from exc
            else:
                vertex_cls = vertex_base_cls

            vertex = vertex_cls(
                vertex_id=vc["id"],
                settings=vc.get("settings", {}),
                initial_data=vc.get("initial_data"),
            )

            if pipeline_module:
                vertex.set_pipeline_module(pipeline_module)

            graph.vertices[vertex.id] = vertex

        # --- edges ---
        for ec in config.get("edges", []):
            script = ec.get("pipeline")
            entrypoint = None
            if script and ":" in script:
                script, entrypoint = script.split(":", 1)
            if script and not os.path.isabs(script):
                script = os.path.join(base_dir, script)

            agent_spec = ec.get("agent")
            if isinstance(agent_spec, str) and agent_spec.endswith(".py") and not os.path.isabs(agent_spec):
                agent_spec = os.path.join(base_dir, agent_spec)
            elif isinstance(agent_spec, dict) and isinstance(agent_spec.get("type"), str) and agent_spec["type"].endswith(".py") and not os.path.isabs(agent_spec["type"]):
                agent_spec["type"] = os.path.join(base_dir, agent_spec["type"])

            pipeline_module = None
            if script:
                from .utils.script_loader import load_class_from_script
                try:
                    edge_cls = load_class_from_script(script, Edge, Edge)
                    pipeline_module = load_script(script)  # Need the module for hooks
                    if entrypoint:
                        pipeline_module = getattr(pipeline_module, entrypoint)

                except Exception as exc:
                    logger.error("[Graph] Pipeline script load failed for edge '%s': %s", ec["id"], exc)
                    raise RuntimeError(f"Pipeline script load failed for edge '{ec['id']}': {exc}") from exc
            else:
                edge_cls = Edge

            edge = edge_cls(
                edge_id=ec["id"],
                source_id=ec["source"],
                destination_id=ec["destination"],
                channel=ec.get("channel", ec.get("data_id", "default")),
                prompt=ec.get("prompt", ""),
                model=ec.get("model", "default"),
                settings=ec.get("settings", {}),
                max_iterations=int(ec.get("max_iterations", 0)),
                agent=agent_spec,
                concurrency_type=ec.get("concurrency_type", "default"),
            )

            if pipeline_module:
                edge.set_pipeline_module(pipeline_module)

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
        logger.debug(
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
        schema_errors: List[str] = []

        for edge in self.edges.values():
            if edge.source_id not in self.vertices:
                errors.append(
                    f"Edge '{edge.id}': source '{edge.source_id}' not found"
                )
            if edge.destination_id not in self.vertices:
                errors.append(
                    f"Edge '{edge.id}': destination '{edge.destination_id}' not found"
                )
                
            # ── Static Schema Validation (Compile-time) ────────────
            if edge.destination_id in self.vertices:
                dest = self.vertices[edge.destination_id]
                out_schema_name = edge.settings.get("output_schema")
                in_schema_name = dest.settings.get("input_schema")
                
                if out_schema_name and in_schema_name and out_schema_name != in_schema_name:
                    schema_errors.append(
                        f"Schema Mismatch on edge '{edge.id}': Edge outputs '{out_schema_name}' "
                        f"but destination vertex '{dest.id}' expects '{in_schema_name}'"
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

        if errors or schema_errors:
            all_errors = errors + schema_errors
            for e in all_errors:
                logger.error("[Graph] Validation: %s", e)
            if schema_errors and not errors:
                raise SchemaMismatchError(f"Graph validation failed: {'; '.join(schema_errors)}")
            raise ValueError(f"Graph validation failed: {'; '.join(all_errors)}")

        # ── Propagate loop metadata to destination vertices ────────────
        for eid in back_edges:
            edge = self.edges[eid]
            dest = self.vertices[edge.destination_id]
            dest.loop_incoming_edges[eid] = edge.max_iterations
            logger.debug(
                "[Graph] Loop edge '%s' registered on vertex '%s' (max_iterations=%d)",
                eid, dest.id, edge.max_iterations,
            )

        logger.debug("[Graph] Validation passed ✓")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph back into a configuration dictionary."""
        config: Dict[str, Any] = {
            "metadata": dict(self.metadata),
            "vertices": [],
            "edges": [],
        }
        for v in self.vertices.values():
            vc: Dict[str, Any] = {"id": v.id}
            if v.settings:
                vc["settings"] = dict(v.settings)
            if hasattr(v, "initial_data") and v.initial_data:
                vc["initial_data"] = list(v.initial_data)
            from .subgraph import SubgraphVertex
            if isinstance(v, SubgraphVertex):
                vc["type"] = "subgraph"
            config["vertices"].append(vc)

        for e in self.edges.values():
            ec: Dict[str, Any] = {
                "id": e.id,
                "source": e.source_id,
                "destination": e.destination_id,
            }
            if e.channel != "default":
                ec["channel"] = e.channel
            if e.prompt:
                ec["prompt"] = e.prompt
            if e.model and e.model != "default":
                ec["model"] = e.model
            if e.settings:
                ec["settings"] = dict(e.settings)
            if e.max_iterations > 0:
                ec["max_iterations"] = e.max_iterations
            config["edges"].append(ec)

        return config

    def to_json(self, json_path: Optional[str] = None, indent: int = 2) -> str:
        """Serialize the graph to a JSON string or write to a JSON file."""
        data = json.dumps(self.to_dict(), indent=indent)
        if json_path:
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(data)
        return data

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
