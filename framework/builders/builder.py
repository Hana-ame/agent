"""GraphBuilder module — Fluent programmatic builder for Graphs.

Provides ``GraphBuilder`` to construct complex graph topologies without writing raw JSON/dict.
"""

from typing import Any, Dict, List, Optional
from ..graph import Graph


class GraphBuilder:
    """Fluent API for building computation graphs programmatically.

    Example::

        g = (
            GraphBuilder("my_pipeline", "A sample pipeline")
            .vertex("input", initial_data=[{"channel": "text", "value": "hello"}])
            .vertex("summarizer")
            .vertex("output")
            .edge("input", "summarizer", prompt="Summarize this:", model="gemini-flash")
            .edge("summarizer", "output", prompt="Translate to French:")
            .build()
        )
    """

    def __init__(self, name: str = "custom_graph", description: str = ""):
        self.metadata: Dict[str, Any] = {
            "name": name,
            "description": description,
        }
        self._vertices: List[Dict[str, Any]] = []
        self._edges: List[Dict[str, Any]] = []

    def vertex(
        self,
        vertex_id: str,
        settings: Optional[Dict[str, Any]] = None,
        initial_data: Optional[List[Dict[str, Any]]] = None,
        script: Optional[str] = None,
        type: str = "vertex",
        **kwargs,
    ) -> "GraphBuilder":
        """Add a vertex node to the graph."""
        vc: Dict[str, Any] = {
            "id": vertex_id,
            "settings": settings or {},
            **kwargs,
        }
        if initial_data:
            vc["initial_data"] = initial_data
        if script:
            vc["script"] = script
        if type != "vertex":
            vc["type"] = type
        self._vertices.append(vc)
        return self

    def edge(
        self,
        source: str,
        destination: str,
        edge_id: Optional[str] = None,
        channel: str = "default",
        prompt: str = "",
        model: str = "default",
        settings: Optional[Dict[str, Any]] = None,
        max_iterations: int = 0,
        script: Optional[str] = None,
        **kwargs,
    ) -> "GraphBuilder":
        """Add a directed edge connection between two vertices."""
        eid = edge_id or f"e_{source}_{destination}"
        s = settings or {}
        if prompt:
            s["prompt"] = prompt
        if model != "default":
            s["model"] = model

        ec: Dict[str, Any] = {
            "id": eid,
            "source": source,
            "destination": destination,
            "channel": channel,
            "settings": s,
            "max_iterations": max_iterations,
            **kwargs,
        }
        if script:
            ec["script"] = script
        self._edges.append(ec)
        return self

    def build(self) -> Graph:
        """Construct, link, and validate the final Graph instance."""
        config: Dict[str, Any] = {
            "metadata": self.metadata,
            "vertices": self._vertices,
            "edges": self._edges,
        }
        return Graph.from_dict(config)
