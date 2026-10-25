"""VEA Context Manager — Vertex modularization + context management.

Each vertex is an independent JSON file; graph.json defines only topology (edges) and agent configuration.
The VEA class implements __enter__/__exit__, encapsulating graph loading, execution, and cleanup.

Usage:
    with VEA("vea_graphs/my_graph") as agent:
        result = agent.run("hello")
        # or
        async for event in agent.stream("hello"):
            print(event)

Directory structure:
    my_graph/
    ├── graph.json          # { "edges": [...], "agent": {...} }
    ├── vertices/
    │   ├── src.json        # { "id": "src", "type": "source", ... }
    │   ├── mid.json
    │   └── sink.json
    └── edges/              # optional, custom edge scripts
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from ..agents.http_llm_agent import HttpLLMAgent
from ..agents.mock_agent import MockAgent
from ..executor import Executor, GraphEvent
from ..graph import Graph

logger = logging.getLogger("vertex_edge_agent.serve.context")

# ---------------------------------------------------------------------------
# Vertex JSON format
# ---------------------------------------------------------------------------

_VERTEX_TYPES = {"source", "sink", "intermediate", "transform", "router", "aggregator"}


def _load_vertex(path: Path) -> Dict[str, Any]:
    """Load a single vertex JSON file and validate basic structure."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "id" not in data:
        data["id"] = path.stem  # fallback: filename without extension

    vtype = data.get("type", "intermediate")
    if vtype not in _VERTEX_TYPES:
        logger.warning("[Context] Unknown vertex type %r for %r, treating as intermediate",
                       vtype, data["id"])
        data["type"] = "intermediate"

    # Normalize settings
    settings = data.get("settings", {})
    settings["type"] = data["type"]
    data["settings"] = settings

    # Ensure initial_data exists for source vertices
    if vtype == "source" and "initial_data" not in data:
        data["initial_data"] = [{"channel": "text", "value": ""}]

    return data


def _load_all_vertices(vertices_dir: Path) -> List[Dict[str, Any]]:
    """Load all vertex JSON files from a directory."""
    if not vertices_dir.is_dir():
        return []
    vertices = []
    for path in sorted(vertices_dir.glob("*.json")):
        try:
            vertices.append(_load_vertex(path))
        except Exception as exc:
            logger.error("[Context] Failed to load vertex %s: %s", path, exc)
    return vertices


# ---------------------------------------------------------------------------
# Graph JSON format
# ---------------------------------------------------------------------------

def _load_graph_config(config_path: Path) -> Dict[str, Any]:
    """Load graph.json: topology (edges) + agent config + metadata."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# VEA Context Manager
# ---------------------------------------------------------------------------

class VEA:
    """VEA context manager — modular vertex JSON + graph execution.

    Parameters
    ----------
    config_dir : str or Path
        Directory containing ``graph.json`` and ``vertices/`` subdirectory.
    agent_cls : class, optional
        Agent class (default: :class:`HttpLLMAgent`). Pass ``MockAgent`` for testing.
    verbose : bool
        Enable debug logging.

    Examples
    --------
    Synchronous usage:

        with VEA("my_graph") as agent:
            result = agent.run("hello")
            print(result)

    Streaming usage:

        with VEA("my_graph") as agent:
            async for event in agent.stream("hello"):
                print(event)

    Multiple graphs:

        with VEA("graph_a") as a, VEA("graph_b") as b:
            r1 = a.run("hello")
            r2 = b.run("world")
    """

    def __init__(
        self,
        config_dir: Union[str, Path],
        *,
        agent_cls: Any = None,
        verbose: bool = False,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.agent_cls = agent_cls or HttpLLMAgent
        self.verbose = verbose

        self._graph: Optional[Graph] = None
        self._agent: Optional[HttpLLMAgent] = None
        self._executor: Optional[Executor] = None
        self._graph_config: Dict[str, Any] = {}
        self._agent_config: Dict[str, Any] = {}

    # -- Context manager protocol ----------------------------------------

    def __enter__(self) -> "VEA":
        """Load graph, build agent, return self."""
        self._load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup agent and executor."""
        self._cleanup()

    async def __aenter__(self) -> "VEA":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cleanup()

    # -- Loading ---------------------------------------------------------

    def _load(self) -> None:
        """Load graph config, vertices, and build the Graph object."""
        # 1. Load graph.json
        graph_path = self.config_dir / "graph.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"graph.json not found in {self.config_dir}")
        self._graph_config = _load_graph_config(graph_path)

        # 2. Load vertices from vertices/ directory
        vertices_dir = self.config_dir / "vertices"
        vertices = _load_all_vertices(vertices_dir)
        if not vertices:
            raise ValueError(f"No vertex JSON files found in {vertices_dir}")

        # 3. Assemble graph config
        config = {
            "metadata": self._graph_config.get("metadata", {"name": self.config_dir.name}),
            "vertices": vertices,
            "edges": self._graph_config.get("edges", []),
        }

        # 4. Extract agent config
        self._agent_config = {**self._graph_config.get("agent", {})}

        # 5. Build Graph
        base_dir = str(self.config_dir)
        self._graph = Graph.from_dict(config, base_dir=base_dir)

        # 6. Build Agent
        self._agent = self._build_agent()

        logger.info(
            "[Context] Loaded graph %r: %d vertices, %d edges",
            config["metadata"].get("name", "?"),
            len(vertices),
            len(config["edges"]),
        )

    def _build_agent(self) -> HttpLLMAgent:
        """Build agent from graph config's 'agent' section."""
        if self.agent_cls is MockAgent:
            return MockAgent()

        cfg = self._agent_config
        return self.agent_cls(
            api_key=cfg.get("api_key", os.environ.get("OPENAI_API_KEY", "public")),
            base_url=cfg.get("base_url", os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")),
            trust_env=cfg.get("trust_env", True),
            proxy=cfg.get("proxy", None),
            max_retries=cfg.get("max_retries", 3),
            timeout=float(cfg.get("timeout", 300.0)),
        )

    def _cleanup(self) -> None:
        """Close agent and release resources."""
        if self._agent:
            try:
                asyncio.get_event_loop().run_until_complete(self._agent.close())
            except Exception:
                pass
            self._agent = None
        self._executor = None
        self._graph = None

    # -- Execution -------------------------------------------------------

    def run(self, input_text: str, *, channel: str = "text") -> str:
        """Execute the graph synchronously and return the answer.

        Parameters
        ----------
        input_text : str
            User input to inject into the source vertex.
        channel : str
            Channel name for the input data (default: "text").

        Returns
        -------
        str
            The answer extracted from the sink vertex.
        """
        if self._graph is None:
            raise RuntimeError("VEA not loaded. Use 'with' block or call _load() first.")

        # Inject input into source vertex
        source = self._find_source()
        if source:
            asyncio.get_event_loop().run_until_complete(
                source.set_data(channel, input_text)
            )

        # Create fresh executor (graph state may be dirty from previous run)
        self._executor = Executor(
            graph=self._graph,
            agents=self._agent,
            timeout=float(self._agent_config.get("timeout", 300.0)),
        )
        result = asyncio.get_event_loop().run_until_complete(self._executor.run())

        if not result.success:
            raise RuntimeError(f"Graph execution failed: {result.errors}")

        return self._extract_answer(result)

    def stream(self, input_text: str, *, channel: str = "text") -> List[GraphEvent]:
        """Execute the graph and return all events (synchronous version)."""
        if self._graph is None:
            raise RuntimeError("VEA not loaded.")

        source = self._find_source()
        if source:
            asyncio.get_event_loop().run_until_complete(
                source.set_data(channel, input_text)
            )

        self._executor = Executor(
            graph=self._graph,
            agents=self._agent,
            timeout=float(self._agent_config.get("timeout", 300.0)),
        )

        events = []
        async def _collect():
            async for event in self._executor.stream():
                events.append(event)

        asyncio.get_event_loop().run_until_complete(_collect())
        return events

    async def stream_async(self, input_text: str, *, channel: str = "text") -> AsyncGenerator[GraphEvent, None]:
        """Execute the graph and yield events (async version)."""
        if self._graph is None:
            raise RuntimeError("VEA not loaded.")

        source = self._find_source()
        if source:
            await source.set_data(channel, input_text)

        self._executor = Executor(
            graph=self._graph,
            agents=self._agent,
            timeout=float(self._agent_config.get("timeout", 300.0)),
        )
        async for event in self._executor.stream():
            yield event

    # -- Graph introspection ---------------------------------------------

    @property
    def graph(self) -> Optional[Graph]:
        """The loaded Graph object."""
        return self._graph

    @property
    def vertices(self) -> Dict[str, Any]:
        """All vertices in the graph."""
        if self._graph is None:
            return {}
        return dict(self._graph.vertices)

    @property
    def edges(self) -> Dict[str, Any]:
        """All edges in the graph."""
        if self._graph is None:
            return {}
        return dict(self._graph.edges)

    @property
    def vertex_ids(self) -> List[str]:
        """List of vertex IDs."""
        return list(self.vertices.keys()) if self._graph else []

    @property
    def edge_ids(self) -> List[str]:
        """List of edge IDs."""
        return list(self.edges.keys()) if self._graph else []

    # -- Helpers ---------------------------------------------------------

    def _find_source(self) -> Any:
        """Find the first source vertex (no incoming edges)."""
        for v in self._graph.vertices.values():
            non_loop = [
                eid for eid in v.incoming_edges
                if eid not in v.loop_incoming_edges
            ]
            if not non_loop:
                return v
        return None

    def _extract_answer(self, result: Any) -> str:
        """Extract answer from sink vertex."""
        for vid, info in result.vertex_results.items():
            v = self._graph.vertices.get(vid)
            if v and v.settings.get("type") == "sink":
                data = info.get("data", {})
                for key, val in data.items():
                    if isinstance(val, str) and val:
                        return val
                    if isinstance(val, dict):
                        for k2, v2 in val.items():
                            if isinstance(v2, str) and v2:
                                return v2
        # Fallback: edge results
        for eid, val in result.edge_results.items():
            if isinstance(val, str) and val:
                return val
        return ""

    def __repr__(self) -> str:
        name = self._graph_config.get("metadata", {}).get("name", "?")
        return f"VEA({name!r}, dir={str(self.config_dir)!r})"
