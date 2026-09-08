"""V4 Standalone Server and Database Viewer.

Provides:
1. Online Graph Mutation API: Modify vertices, edges, and topology per session in real-time.
2. Isolated Session Graphs: Each session maintains an independent GraphV4 instance with live event streaming.
3. Standalone Database Viewer & Dashboard: Inspect SQLite vertices and session_staging scratchpad tables.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from framework.edge_v4 import CodeEdgeV4, EdgeResultV4, EdgeV4, LLMEdgeV4, ReflexiveEdgeV4
from framework.executor_v4 import ExecutionResultV4, ExecutorV4, GraphEventV4
from framework.graph_v4 import DiscreteGraphLoaderV4, GraphTopologyError, GraphV4
from framework.vertex_v4 import (
    StagingRecordV4,
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.server_v4")



def _validate_path_security(path_str: str, base_dir: Path) -> Path:
    """Validate that a path doesn't escape the allowed base directory.
    
    Raises HTTPException(400) with structured error if path is invalid.
    """
    resolved = Path(path_str).resolve()
    base_resolved = base_dir.resolve()
    if not str(resolved).startswith(str(base_resolved) + os.sep) and resolved != base_resolved:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "Invalid path: target is outside allowed directory",
                "error_code": "PATH_TRAVERSAL_REJECTED",
            },
        )
    return resolved

# ---------------------------------------------------------------------------
# Pydantic Schemas for Online Graph API
# ---------------------------------------------------------------------------

class VertexCreateOrUpdateRequest(BaseModel):
    name: str = Field(..., description="Unique vertex name in session")
    content: str = Field(default="", description="Payload content")
    attributes: List[str] = Field(default_factory=list, description="Vertex attributes")
    state: str = Field(default=VertexStateV4.IDLE.value, description="Lifecycle state")
    processed_count: int = Field(default=0, description="Processing counter")

    @field_validator('state')
    @classmethod
    def validate_state(cls, v: str) -> str:
        valid_states = {s.value for s in VertexStateV4}
        if v not in valid_states:
            raise ValueError(f"Invalid state: {v}")
        return v

    @field_validator('attributes')
    @classmethod
    def validate_attributes(cls, v: List[str]) -> List[str]:
        valid_attrs = {a.value for a in VertexAttributeV4}
        for attr in v:
            if attr not in valid_attrs:
                raise ValueError(f"Invalid attribute: {attr}")
        return v


class EdgeCreateOrUpdateRequest(BaseModel):
    id: str = Field(..., description="Unique edge identifier")
    type: str = Field(default="code", description="Edge type: code, llm, or reflexive")
    input_vertex: str = Field(..., description="Input vertex name")
    output_vertex: str = Field(..., description="Output vertex name")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Edge settings")
    script: Optional[str] = Field(default=None, description="Script path or callable specification")
    trigger_state: Optional[str] = Field(default=None, description="Trigger state for reflexive edge")
    target_state: Optional[str] = Field(default=None, description="Target reset state for reflexive edge")
    max_retries: Optional[int] = Field(default=None, description="Max retries for reflexive edge")


class WorkflowRunRequest(BaseModel):
    max_concurrency: int = Field(default=4, ge=1, le=64, description="Max concurrent edges")
    timeout: float = Field(default=120.0, gt=0.0, description="Execution timeout in seconds")


class VertexReentryRequest(BaseModel):
    new_content: Optional[str] = Field(default=None, description="Optional new input content")
    reset_state: str = Field(default=VertexStateV4.TODO.value, description="Target state for affected downstream nodes")
    clear_content: bool = Field(default=False, description="Whether to clear content of reset downstream nodes")


class EdgeReconnectRequest(BaseModel):
    new_input_vertex: Optional[str] = Field(default=None, description="New input vertex name")
    new_output_vertex: Optional[str] = Field(default=None, description="New output vertex name")


class SubgraphSpliceRequest(BaseModel):
    target_vertex: str = Field(..., description="Target vertex name to replace with subgraph")
    subgraph_manifest: Optional[str] = Field(default=None, description="Path to subgraph manifest JSON file")
    subgraph_data: Optional[Dict[str, Any]] = Field(default=None, description="In-memory subgraph definition")
    name_prefix: Optional[str] = Field(default=None, description="Prefix for inserted subgraph entities")
    entry_vertex: Optional[str] = Field(default=None, description="Explicit entry vertex of subgraph")
    exit_vertex: Optional[str] = Field(default=None, description="Explicit exit vertex of subgraph")


class SubgraphInsertRequest(BaseModel):
    subgraph_manifest: Optional[str] = Field(default=None, description="Path to subgraph manifest JSON file")
    subgraph_data: Optional[Dict[str, Any]] = Field(default=None, description="In-memory subgraph definition")
    incoming_bindings: Optional[Dict[str, str]] = Field(default=None, description="{parent_vertex: sub_entry}")
    outgoing_bindings: Optional[Dict[str, str]] = Field(default=None, description="{sub_exit: parent_vertex}")
    name_prefix: Optional[str] = Field(default=None, description="Prefix for inserted subgraph entities")


class SubgraphAddRequest(BaseModel):
    subgraph_manifest: Optional[str] = Field(default=None, description="Path to subgraph manifest JSON file")
    subgraph_data: Optional[Dict[str, Any]] = Field(default=None, description="In-memory subgraph definition")
    name_prefix: Optional[str] = Field(default=None, description="Prefix for added subgraph entities")
    connections: Optional[List[Dict[str, Any]]] = Field(default=None, description="Explicit connections between parent and subgraph")
    incoming_bindings: Optional[Dict[str, str]] = Field(default=None, description="{parent_vertex: sub_entry}")
    outgoing_bindings: Optional[Dict[str, str]] = Field(default=None, description="{sub_exit: parent_vertex}")
    source: Optional[str] = Field(default=None, description="Provenance source identifier for loaded nodes")



# ---------------------------------------------------------------------------
# Session Graph Manager
# ---------------------------------------------------------------------------

class SessionGraphManagerV4:
    """Maintains isolated GraphV4 instances per session synchronized with VertexStoreV4."""

    def __init__(self, store: VertexStoreV4):
        self.store = store
        self._graphs: Dict[str, GraphV4] = {}
        self._event_broadcasters: Dict[str, List[asyncio.Queue]] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        # P2: Track running executors per session for task cancellation on reentry
        self._running_executors: Dict[str, "ExecutorV4"] = {}

    def get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Fetch or create an asyncio.Lock for the session."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    def register_executor(self, session_id: str, executor: "ExecutorV4") -> None:
        """Register a running executor so reentry can cancel its in-flight tasks."""
        self._running_executors[session_id] = executor

    def unregister_executor(self, session_id: str) -> None:
        """Remove a finished executor from tracking."""
        self._running_executors.pop(session_id, None)

    def get_running_executor(self, session_id: str) -> Optional["ExecutorV4"]:
        """Get the currently running executor for a session, if any."""
        return self._running_executors.get(session_id)

    def load_graph_from_store(self, session_id: str) -> GraphV4:
        """Read and update a fresh GraphV4 instance directly from SQLite store for the session."""
        hydrated = GraphV4.load_from_store(self.store, session_id, name=f"graph_{session_id}")
        old_graph = self._graphs.get(session_id)
        if old_graph:
            hydrated.loaded_nodes.update(old_graph.loaded_nodes)
            hydrated.metadata.update(old_graph.metadata)
            for eid, edge in hydrated.edges.items():
                old_edge = old_graph.edges.get(eid)
                if old_edge and callable(getattr(old_edge, "script", None)):
                    edge.script = old_edge.script
        self._graphs[session_id] = hydrated
        return hydrated

    def get_or_create_graph(self, session_id: str) -> GraphV4:
        """Fetch or instantiate an isolated GraphV4 for a session, reading from store if available."""
        hydrated = GraphV4.load_from_store(self.store, session_id, name=f"graph_{session_id}")
        if hydrated.vertices or hydrated.edges:
            old_graph = self._graphs.get(session_id)
            if old_graph:
                hydrated.loaded_nodes.update(old_graph.loaded_nodes)
                hydrated.metadata.update(old_graph.metadata)
                for eid, edge in hydrated.edges.items():
                    old_edge = old_graph.edges.get(eid)
                    if old_edge and callable(getattr(old_edge, "script", None)):
                        edge.script = old_edge.script
            self._graphs[session_id] = hydrated
            return hydrated
        if session_id not in self._graphs:
            self._graphs[session_id] = GraphV4(session_id=session_id, name=f"graph_{session_id}")
        return self._graphs[session_id]

    def list_active_sessions(self) -> List[str]:
        """List distinct session IDs from in-memory graphs and database."""
        session_set: Set[str] = set(self._graphs.keys())
        session_set.update(self.store.list_sessions())
        return sorted(list(session_set))

    def add_or_update_vertex(
        self,
        session_id: str,
        name: str,
        content: str = "",
        attributes: Optional[List[str]] = None,
        state: str = VertexStateV4.IDLE.value,
        processed_count: int = 0,
    ) -> VertexRecordV4:
        """Online vertex addition or update. Persists to SQLite without in-memory graph mutation."""
        db_record = self.store.save_vertex(
            session_id=session_id,
            name=name,
            content=content,
            attributes=attributes,
            state=state,
            processed_count=processed_count,
        )
        return db_record

    def delete_vertex(self, session_id: str, name: str) -> bool:
        """Delete vertex from database, cleaning up connected edges in SQLite."""
        for er in self.store.list_edges(session_id):
            if er.input_vertex == name or er.output_vertex == name:
                self.store.delete_edge(session_id, er.edge_id)
        return self.store.delete_vertex(session_id, name)

    def add_or_update_edge(
        self,
        session_id: str,
        edge_id: str,
        edge_type: str,
        input_vertex: str,
        output_vertex: str,
        settings: Optional[Dict[str, Any]] = None,
        script: Optional[str] = None,
        trigger_state: Optional[str] = None,
        target_state: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> EdgeV4:
        """Online edge addition or update. Persists to SQLite store."""
        if edge_type not in ("code", "llm", "reflexive") and input_vertex != output_vertex:
            raise ValueError(f"Unsupported edge type: {edge_type}")
        edge_settings = dict(settings or {})
        script_str = script if isinstance(script, str) else None

        # Delegate instantiation and validation to EdgeV4.from_config
        cfg = dict(edge_settings)
        cfg.update({
            "id": edge_id,
            "type": edge_type,
            "input_vertex": input_vertex,
            "output_vertex": output_vertex,
            "script": script_str,
            "trigger_state": trigger_state or (VertexStateV4.REJECT.value if edge_type == "reflexive" or input_vertex == output_vertex else None),
            "target_state": target_state or (VertexStateV4.TODO_URGENT.value if edge_type == "reflexive" or input_vertex == output_vertex else None),
            "max_retries": max_retries if max_retries is not None else int(edge_settings.get("max_retries", 3)),
        })
        edge = EdgeV4.from_config(cfg)
        if callable(script):
            edge.script = script
            if hasattr(edge, "_callable"):
                edge._callable = script

        self.store.save_edge(
            session_id=session_id,
            edge_id=edge_id,
            edge_type=edge.type,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            script=script_str,
            trigger_state=getattr(edge, "trigger_state", None),
            target_state=getattr(edge, "target_state", None),
            max_retries=getattr(edge, "max_retries", 3),
            settings=edge_settings,
        )
        return edge

    def delete_edge(self, session_id: str, edge_id: str) -> bool:
        """Delete edge from SQLite database."""
        return self.store.delete_edge(session_id, edge_id)

    def reconnect_edge(
        self,
        session_id: str,
        edge_id: str,
        new_input_vertex: Optional[str] = None,
        new_output_vertex: Optional[str] = None,
    ) -> bool:
        """Dynamically reconnect an existing edge to new input or output vertices, syncing to database."""
        edges = {e.edge_id: e for e in self.store.list_edges(session_id)}
        if edge_id not in edges:
            return False
        er = edges[edge_id]
        new_in = new_input_vertex or er.input_vertex
        new_out = new_output_vertex or er.output_vertex
        self.store.save_edge(
            session_id=session_id,
            edge_id=edge_id,
            edge_type=er.edge_type,
            input_vertex=new_in,
            output_vertex=new_out,
            script=er.script,
            trigger_state=er.trigger_state,
            target_state=er.target_state,
            max_retries=er.max_retries,
            settings=er.settings,
        )
        return True

    def reenter_vertex(
        self,
        session_id: str,
        vertex_name: str,
        new_content: Optional[str] = None,
        reset_state: str = VertexStateV4.TODO.value,
        clear_content: bool = False,
    ) -> List[str]:
        """Re-enter a vertex for re-execution, resetting all affected downstream vertices in DB and memory."""
        graph = self.get_or_create_graph(session_id)
        affected = graph.reenter_vertex(
            vertex_name=vertex_name,
            new_content=new_content,
            reset_state=reset_state,
            clear_content=clear_content,
            store=self.store,
            session_id=session_id,
        )
        return affected

    def splice_subgraph(
        self,
        session_id: str,
        target_vertex_name: str,
        subgraph: GraphV4,
        name_prefix: Optional[str] = None,
        entry_vertex_name: Optional[str] = None,
        exit_vertex_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Splice (inline) a subgraph in place of an existing vertex, fully synchronized with SQLite."""
        graph = self.get_or_create_graph(session_id)
        res = graph.splice_subgraph(
            target_vertex_name=target_vertex_name,
            subgraph=subgraph,
            name_prefix=name_prefix,
            entry_vertex_name=entry_vertex_name,
            exit_vertex_name=exit_vertex_name,
        )

        # Sync to SQLite store
        self.store.delete_vertex(session_id, target_vertex_name)

        for v_name in res["inserted_vertices"]:
            v = graph.get_vertex(v_name)
            if v:
                db_v = self.store.save_vertex(
                    session_id=session_id,
                    name=v.name,
                    content=v.content,
                    attributes=v.attributes,
                    state=v.state,
                    processed_count=v.processed_count,
                )
                v.id = db_v.id

        for e_id in res["inserted_edges"]:
            e = graph.get_edge(e_id)
            if e:
                self.store.save_edge(
                    session_id=session_id,
                    edge_id=e.id,
                    edge_type=e.type,
                    input_vertex=e.input_vertex,
                    output_vertex=e.output_vertex,
                    script=getattr(e, "script", None) if isinstance(getattr(e, "script", None), str) else None,
                    trigger_state=getattr(e, "trigger_state", None),
                    target_state=getattr(e, "target_state", None),
                    max_retries=getattr(e, "max_retries", 3),
                    settings=e.settings,
                )

        for e_id in res["rewired_edges"]:
            e = graph.get_edge(e_id)
            if e:
                self.store.save_edge(
                    session_id=session_id,
                    edge_id=e.id,
                    edge_type=e.type,
                    input_vertex=e.input_vertex,
                    output_vertex=e.output_vertex,
                    script=getattr(e, "script", None) if isinstance(getattr(e, "script", None), str) else None,
                    trigger_state=getattr(e, "trigger_state", None),
                    target_state=getattr(e, "target_state", None),
                    max_retries=getattr(e, "max_retries", 3),
                    settings=e.settings,
                )

        return res

    def insert_subgraph(
        self,
        session_id: str,
        subgraph: GraphV4,
        incoming_bindings: Optional[Dict[str, str]] = None,
        outgoing_bindings: Optional[Dict[str, str]] = None,
        name_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert an independent subgraph with explicit boundary bindings, fully synchronized with SQLite."""
        graph = self.get_or_create_graph(session_id)
        res = graph.insert_subgraph(
            subgraph=subgraph,
            incoming_bindings=incoming_bindings,
            outgoing_bindings=outgoing_bindings,
            name_prefix=name_prefix,
        )

        for v_name in res["inserted_vertices"]:
            v = graph.get_vertex(v_name)
            if v:
                db_v = self.store.save_vertex(
                    session_id=session_id,
                    name=v.name,
                    content=v.content,
                    attributes=v.attributes,
                    state=v.state,
                    processed_count=v.processed_count,
                )
                v.id = db_v.id

        for e_id in res["inserted_edges"]:
            e = graph.get_edge(e_id)
            if e:
                self.store.save_edge(
                    session_id=session_id,
                    edge_id=e.id,
                    edge_type=e.type,
                    input_vertex=e.input_vertex,
                    output_vertex=e.output_vertex,
                    script=getattr(e, "script", None) if isinstance(getattr(e, "script", None), str) else None,
                    trigger_state=getattr(e, "trigger_state", None),
                    target_state=getattr(e, "target_state", None),
                    max_retries=getattr(e, "max_retries", 3),
                    settings=e.settings,
                )

        return res

    def add_subgraph(
        self,
        session_id: str,
        subgraph: GraphV4,
        name_prefix: Optional[str] = None,
        connections: Optional[List[Dict[str, Any]]] = None,
        incoming_bindings: Optional[Dict[str, str]] = None,
        outgoing_bindings: Optional[Dict[str, str]] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add and join an arbitrary subgraph into session graph, fully synchronized with SQLite."""
        graph = self.get_or_create_graph(session_id)
        res = graph.add_subgraph(
            subgraph=subgraph,
            name_prefix=name_prefix,
            connections=connections,
            incoming_bindings=incoming_bindings,
            outgoing_bindings=outgoing_bindings,
            source=source,
        )

        for v_name in res["added_vertices"]:
            v = graph.get_vertex(v_name)
            if v:
                db_v = self.store.save_vertex(
                    session_id=session_id,
                    name=v.name,
                    content=v.content,
                    attributes=v.attributes,
                    state=v.state,
                    processed_count=v.processed_count,
                )
                v.id = db_v.id

        for e_id in res["added_edges"]:
            e = graph.get_edge(e_id)
            if e:
                self.store.save_edge(
                    session_id=session_id,
                    edge_id=e.id,
                    edge_type=e.type,
                    input_vertex=e.input_vertex,
                    output_vertex=e.output_vertex,
                    script=getattr(e, "script", None) if isinstance(getattr(e, "script", None), str) else None,
                    trigger_state=getattr(e, "trigger_state", None),
                    target_state=getattr(e, "target_state", None),
                    max_retries=getattr(e, "max_retries", 3),
                    settings=e.settings,
                )

        return res

    def validate_graph(self, session_id: str) -> Dict[str, Any]:
        """Validate DAG constraints and return edge tiers."""
        graph = self.get_or_create_graph(session_id)
        try:
            graph.validate()
            return {
                "valid": True,
                "tiers": graph.edge_tiers,
                "vertex_count": len(graph.vertices),
                "edge_count": len(graph.edges),
            }
        except GraphTopologyError as err:
            return {
                "valid": False,
                "error": str(err),
                "vertex_count": len(graph.vertices),
                "edge_count": len(graph.edges),
            }

    def list_loaded_nodes(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve tracking metadata for all loaded vertices in a session."""
        graph = self.get_or_create_graph(session_id)
        return graph.list_loaded_nodes()

    def get_node_relationships(self, session_id: str, vertex_name: str) -> Dict[str, Any]:
        """Retrieve relationship and topology details for a single vertex."""
        graph = self.get_or_create_graph(session_id)
        return graph.get_node_relationships(vertex_name)

    def get_graph_relationships(self, session_id: str) -> Dict[str, Any]:
        """Retrieve complete relationship matrix and topology summary for a session."""
        graph = self.get_or_create_graph(session_id)
        return graph.get_graph_relationships()

    def dump_graph(self, session_id: str, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Dump complete graph structure, components, and relationships for a session."""
        graph = self.get_or_create_graph(session_id)
        return graph.dump(path=path)

    def register_event_queue(self, session_id: str, q: asyncio.Queue) -> None:
        """Register an SSE broadcast subscriber queue for a session."""
        if session_id not in self._event_broadcasters:
            self._event_broadcasters[session_id] = []
        self._event_broadcasters[session_id].append(q)

    def unregister_event_queue(self, session_id: str, q: asyncio.Queue) -> None:
        """Remove an SSE broadcast subscriber queue and clean up empty session registry."""
        if session_id in self._event_broadcasters:
            try:
                self._event_broadcasters[session_id].remove(q)
            except ValueError:
                pass
            if not self._event_broadcasters[session_id]:
                self._event_broadcasters.pop(session_id, None)

    def broadcast_event(self, session_id: str, event: GraphEventV4) -> None:
        """Broadcast an execution event to all active SSE subscribers with backpressure protection."""
        queues = list(self._event_broadcasters.get(session_id, []))
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception as e:
                    logger.warning('Exception ignored: %s', e)


# ---------------------------------------------------------------------------
# HTML Dashboard Source
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent / "templates"

def _load_dashboard_html() -> str:
    """Load dashboard HTML template from file."""
    template_path = _TEMPLATE_DIR / "dashboard.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "<html><body><h1>Dashboard template not found</h1></body></html>"

DASHBOARD_HTML = _load_dashboard_html()


# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------

def create_v4_server(
    store_or_db: Union[VertexStoreV4, str, Path] = ":memory:",
    manager: Optional[SessionGraphManagerV4] = None,
    agent: Optional[Any] = None,
    allowed_origins: List[str] = ["*"],
    manifest_base_dir: Optional[Union[str, Path]] = None,
) -> FastAPI:
    """Create a FastAPI application powering online graph APIs and database dashboard."""
    if isinstance(store_or_db, VertexStoreV4):
        store = store_or_db
    else:
        store = VertexStoreV4(str(store_or_db))

    if manager is None:
        manager = SessionGraphManagerV4(store=store)
    else:
        store = manager.store

    app = FastAPI(
        title="VEA v4 Online Graph & Database Server",
        version="4.0.0",
        description="Online graph modification API and standalone database inspector.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach shared instances to app state
    app.state.store = store
    app.state.manager = manager
    app.state.agent = agent
    app.state.manifest_base_dir = manifest_base_dir

    # -----------------------------------------------------------------------
    # Dashboard Endpoint
    # -----------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard() -> HTMLResponse:
        """Serve live single-page dashboard."""
        return HTMLResponse(content=DASHBOARD_HTML)

    # -----------------------------------------------------------------------
    # Database Inspection Endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/db/stats")
    async def get_db_stats() -> Dict[str, Any]:
        """Return overall database statistics across all sessions."""
        return store.get_db_stats()

    @app.get("/api/db/sessions")
    async def list_sessions() -> List[str]:
        """List distinct session IDs."""
        return manager.list_active_sessions()

    @app.get("/api/db/sessions/{session_id}/vertices")
    async def get_db_vertices(
        session_id: str,
        state: Optional[str] = Query(None, description="Optional state filter"),
    ) -> List[Dict[str, Any]]:
        """Fetch raw vertices for a session directly from SQLite."""
        records = store.list_vertices(session_id=session_id, state=state)
        return [r.to_dict() for r in records]

    @app.get("/api/db/sessions/{session_id}/staging")
    async def get_db_staging(
        session_id: str,
        key: Optional[str] = Query(None),
        edge_id: Optional[str] = Query(None),
        vertex_name: Optional[str] = Query(None),
    ) -> List[Dict[str, Any]]:
        """Query raw records from session_staging table."""
        records = store.get_staged(
            session_id=session_id,
            key=key,
            edge_id=edge_id,
            vertex_name=vertex_name,
        )
        return [r.to_dict() for r in records]

    @app.post("/api/db/sessions/{session_id}/clear")
    async def clear_session_db(session_id: str) -> Dict[str, str]:
        """Purge all records for a session from SQLite and in-memory state."""
        store.clear_session(session_id)
        manager._graphs.pop(session_id, None)
        manager._event_broadcasters.pop(session_id, None)
        return {"status": "cleared", "session_id": session_id}

    # -----------------------------------------------------------------------
    # Online Graph Mutation API
    # -----------------------------------------------------------------------

    @app.get("/api/sessions/{session_id}/graph")
    async def get_session_graph(session_id: str) -> Dict[str, Any]:
        """Retrieve the current graph structure, endpoints, and DAG validation."""
        graph = manager.get_or_create_graph(session_id)
        validation = manager.validate_graph(session_id)
        edges_data = {}
        for eid, e in graph.edges.items():
            edges_data[eid] = {
                "id": e.id,
                "type": e.type,
                "input_vertex": e.input_vertex,
                "output_vertex": e.output_vertex,
                "is_reflexive": e.is_reflexive,
                "settings": e.settings,
                "script": getattr(e, 'script', None),
                "trigger_state": getattr(e, 'trigger_state', None),
                "target_state": getattr(e, 'target_state', None),
                "max_retries": getattr(e, 'max_retries', None),
            }

        # Query database store to always reflect latest real-time states and contents
        db_vertices = {v.name: v for v in store.list_vertices(session_id)}
        vertices_data = {}
        for vname, v in graph.vertices.items():
            if vname in db_vertices:
                vertices_data[vname] = db_vertices[vname].to_dict()
            else:
                vertices_data[vname] = v.to_dict()
        for vname, db_v in db_vertices.items():
            if vname not in vertices_data:
                vertices_data[vname] = db_v.to_dict()

        return {
            "session_id": session_id,
            "valid": validation.get("valid", False),
            "tiers": validation.get("tiers", {}),
            "error": validation.get("error"),
            "vertices": vertices_data,
            "edges": edges_data,
        }

    @app.get("/api/sessions/{session_id}/graph/nodes")
    async def list_session_loaded_nodes(session_id: str) -> Dict[str, Any]:
        """List all loaded vertices with provenance tracking for a session."""
        nodes = manager.list_loaded_nodes(session_id)
        return {"session_id": session_id, "nodes": nodes, "total": len(nodes)}

    @app.get("/api/sessions/{session_id}/graph/nodes/{name}/relationships")
    async def get_session_node_relationships(session_id: str, name: str) -> Dict[str, Any]:
        """Retrieve predecessors, successors, and relationship details for a single vertex."""
        try:
            return manager.get_node_relationships(session_id, name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Vertex '{name}' not found in session graph")

    @app.get("/api/sessions/{session_id}/graph/relationships")
    async def get_session_graph_relationships(session_id: str) -> Dict[str, Any]:
        """Retrieve full graph relationship matrix, adjacency lists, and roots/sinks."""
        return manager.get_graph_relationships(session_id)

    @app.get("/api/sessions/{session_id}/graph/dump")
    @app.post("/api/sessions/{session_id}/graph/dump")
    async def dump_session_graph(
        session_id: str,
        path: Optional[str] = Query(default=None, description="Optional filesystem path to dump JSON"),
        payload: Optional[Dict[str, Any]] = Body(default=None),
    ) -> Dict[str, Any]:
        """Dump complete session graph structure and component metadata."""
        target_path = path
        if not target_path and payload and isinstance(payload, dict):
            target_path = payload.get("path")
        if target_path:
            base_dir = Path(app.state.manifest_base_dir or Path.cwd()).resolve()
            # Must end in .json
            if not str(target_path).endswith('.json'):
                raise HTTPException(400, "Dump path must end in .json")
            _validate_path_security(target_path, base_dir)
        dumped = manager.dump_graph(session_id, path=target_path)
        return {"status": "dumped", "session_id": session_id, "graph": dumped}

    @app.post("/api/sessions/{session_id}/graph/vertices")
    async def create_or_update_vertex(
        session_id: str,
        req: VertexCreateOrUpdateRequest,
    ) -> Dict[str, Any]:
        """Add or update a vertex online."""
        async with manager.get_session_lock(session_id):
            v = manager.add_or_update_vertex(
                session_id=session_id,
                name=req.name,
                content=req.content,
                attributes=req.attributes,
                state=req.state,
                processed_count=req.processed_count,
            )
            return {"status": "saved", "vertex": v.to_dict()}

    @app.delete("/api/sessions/{session_id}/graph/vertices/{name}")
    async def delete_vertex(session_id: str, name: str) -> Dict[str, Any]:
        """Delete a vertex online."""
        async with manager.get_session_lock(session_id):
            deleted = manager.delete_vertex(session_id, name)
            return {"deleted": deleted, "name": name}

    @app.post("/api/sessions/{session_id}/graph/edges")
    async def create_or_update_edge(
        session_id: str,
        req: EdgeCreateOrUpdateRequest,
    ) -> Dict[str, Any]:
        """Add or update an edge online."""
        async with manager.get_session_lock(session_id):
            edge = manager.add_or_update_edge(
                session_id=session_id,
                edge_id=req.id,
                edge_type=req.type,
                input_vertex=req.input_vertex,
                output_vertex=req.output_vertex,
                settings=req.settings,
                script=req.script,
                trigger_state=req.trigger_state,
                target_state=req.target_state,
                max_retries=req.max_retries,
            )
            val = manager.validate_graph(session_id)
            return {
                "status": "saved",
                "edge": {
                    "id": edge.id,
                    "type": edge.type,
                    "input": edge.input_vertex,
                    "output": edge.output_vertex,
                },
                "validation": val,
            }

    @app.delete("/api/sessions/{session_id}/graph/edges/{edge_id}")
    async def delete_edge(session_id: str, edge_id: str) -> Dict[str, Any]:
        """Delete an edge online."""
        async with manager.get_session_lock(session_id):
            deleted = manager.delete_edge(session_id, edge_id)
            return {"deleted": deleted, "edge_id": edge_id}

    @app.post("/api/sessions/{session_id}/graph/validate")
    async def validate_graph(session_id: str) -> Dict[str, Any]:
        """Validate DAG topology and tiers for session graph."""
        return manager.validate_graph(session_id)

    # -----------------------------------------------------------------------
    # Dynamic Workflow Execution & Live SSE Events
    # -----------------------------------------------------------------------

    @app.post("/api/sessions/{session_id}/run")
    async def run_session_workflow(
        session_id: str,
        req: WorkflowRunRequest = Body(default_factory=WorkflowRunRequest),
    ) -> Dict[str, Any]:
        """Execute session graph with specified concurrency limit and broadcast events."""
        async with manager.get_session_lock(session_id):
            # 每次执行前都从 store 进行读取更新，不使用内存脏状态
            graph = manager.load_graph_from_store(session_id)
            executor = ExecutorV4(
                graph=graph,
                store=store,
                agent=app.state.agent,
                max_concurrency=req.max_concurrency,
                timeout=req.timeout,
            )
            # P2: Register executor for task cancellation on reentry
            manager.register_executor(session_id, executor)

            # Broadcast events in real-time
            async def stream_and_broadcast():
                async for ev in executor.stream():
                    manager.broadcast_event(session_id, ev)

            try:
                await stream_and_broadcast()
            finally:
                manager.unregister_executor(session_id)
            return executor.result.to_dict()

    @app.get("/api/sessions/{session_id}/events")
    async def stream_session_events(session_id: str) -> StreamingResponse:
        """SSE stream broadcasting execution events in real time."""
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        manager.register_event_queue(session_id, q)

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                while True:
                    ev: GraphEventV4 = await q.get()
                    data = {
                        "event_type": ev.event_type,
                        "edge_id": ev.edge_id,
                        "vertex_name": ev.vertex_name,
                        "payload": ev.payload,
                        "timestamp": ev.timestamp,
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                manager.unregister_event_queue(session_id, q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.post("/api/sessions/{session_id}/graph/vertices/{name}/reenter")
    async def reenter_vertex_route(
        session_id: str,
        name: str,
        req: VertexReentryRequest = Body(default_factory=VertexReentryRequest),
    ) -> Dict[str, Any]:
        """Re-enter a vertex for re-execution, resetting all affected downstream vertices.

        P2 FIX: Cancels any in-flight tasks targeting downstream vertices before
        resetting state, preventing stale results from stomping on the new state.
        """
        async with manager.get_session_lock(session_id):
            # P2: Cancel in-flight tasks for the reentered vertex and its downstream
            cancelled_edges: List[str] = []
            executor = manager.get_running_executor(session_id)
            if executor:
                # Determine affected downstream vertices
                graph = manager.get_or_create_graph(session_id)
                affected_set: Set[str] = set()
                # The reentered vertex itself
                affected_set.add(name)
                # All downstream vertices reachable from it
                visited: Set[str] = set()
                queue: List[str] = [name]
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    for e in graph.get_outgoing_edges(current):
                        if not e.is_reflexive:
                            affected_set.add(e.output_vertex)
                            queue.append(e.output_vertex)
                cancelled_edges = executor.cancel_downstream_tasks(affected_set)

            affected = manager.reenter_vertex(
                session_id=session_id,
                vertex_name=name,
                new_content=req.new_content,
                reset_state=req.reset_state,
                clear_content=req.clear_content,
            )
            return {
                "status": "reentered",
                "reentered_vertex": name,
                "affected_downstream_vertices": affected,
                "reset_state": req.reset_state,
                "cancelled_in_flight_edges": cancelled_edges,
            }

    @app.patch("/api/sessions/{session_id}/graph/edges/{edge_id}/reconnect")
    async def reconnect_edge_route(
        session_id: str,
        edge_id: str,
        req: EdgeReconnectRequest,
    ) -> Dict[str, Any]:
        """Dynamically reconnect an existing edge to new endpoints."""
        async with manager.get_session_lock(session_id):
            success = manager.reconnect_edge(
                session_id=session_id,
                edge_id=edge_id,
                new_input_vertex=req.new_input_vertex,
                new_output_vertex=req.new_output_vertex,
            )
            if not success:
                raise HTTPException(404, f"Edge '{edge_id}' not found in session '{session_id}'")
            return {"status": "reconnected", "edge_id": edge_id}

    @app.post("/api/sessions/{session_id}/graph/subgraphs/splice")
    async def splice_subgraph_route(
        session_id: str,
        req: SubgraphSpliceRequest,
    ) -> Dict[str, Any]:
        """Splice (inline) a subgraph in place of an existing vertex."""
        async with manager.get_session_lock(session_id):
            subgraph: GraphV4
            if req.subgraph_manifest:
                subgraph = DiscreteGraphLoaderV4.load_from_manifest(req.subgraph_manifest)
            elif req.subgraph_data:
                subgraph = DiscreteGraphLoaderV4.load_from_dict(req.subgraph_data)
            else:
                raise HTTPException(400, "Either subgraph_manifest or subgraph_data must be provided")
            result = manager.splice_subgraph(
                session_id=session_id,
                target_vertex_name=req.target_vertex,
                subgraph=subgraph,
                name_prefix=req.name_prefix,
                entry_vertex_name=req.entry_vertex,
                exit_vertex_name=req.exit_vertex,
            )
            return {"status": "spliced", "result": result}

    @app.post("/api/sessions/{session_id}/graph/subgraphs/insert")
    async def insert_subgraph_route(
        session_id: str,
        req: SubgraphInsertRequest,
    ) -> Dict[str, Any]:
        """Insert an independent subgraph with explicit boundary bindings."""
        async with manager.get_session_lock(session_id):
            subgraph: GraphV4
            if req.subgraph_manifest:
                subgraph = DiscreteGraphLoaderV4.load_from_manifest(req.subgraph_manifest)
            elif req.subgraph_data:
                subgraph = DiscreteGraphLoaderV4.load_from_dict(req.subgraph_data)
            else:
                raise HTTPException(400, "Either subgraph_manifest or subgraph_data must be provided")

            result = manager.insert_subgraph(
                session_id=session_id,
                subgraph=subgraph,
                incoming_bindings=req.incoming_bindings,
                outgoing_bindings=req.outgoing_bindings,
                name_prefix=req.name_prefix,
            )
            return {"status": "inserted", "result": result}

    @app.post("/api/sessions/{session_id}/graph/subgraphs/add")
    @app.post("/api/sessions/{session_id}/graph/subgraphs")
    async def add_subgraph_route(
        session_id: str,
        req: SubgraphAddRequest,
    ) -> Dict[str, Any]:
        """Add and join an arbitrary subgraph into session graph with optional prefix, connections, and bindings."""
        async with manager.get_session_lock(session_id):
            subgraph: GraphV4
            if req.subgraph_manifest:
                subgraph = DiscreteGraphLoaderV4.load_from_manifest(req.subgraph_manifest)
            elif req.subgraph_data:
                subgraph = DiscreteGraphLoaderV4.load_from_dict(req.subgraph_data)
            else:
                raise HTTPException(400, "Either subgraph_manifest or subgraph_data must be provided")

            result = manager.add_subgraph(
                session_id=session_id,
                subgraph=subgraph,
                name_prefix=req.name_prefix,
                connections=req.connections,
                incoming_bindings=req.incoming_bindings,
                outgoing_bindings=req.outgoing_bindings,
                source=req.source,
            )
            return {"status": "added", "result": result}


    # -----------------------------------------------------------------------
    # SSE Executor with Session Routing & Harness Tool Call Echo
    # -----------------------------------------------------------------------

    @app.post("/api/sse/execute")
    async def execute_via_sse(
        payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Any:
        """Execute workflow via SSEExecutor with session routing and harness echo output."""
        from framework.sse_executor_v4 import SSEExecutorV4

        session_id = payload.get("session_id")
        input_payload = payload.get("input_payload")
        manifest_path = payload.get("manifest_path")
        if manifest_path:
            base_dir = Path(app.state.manifest_base_dir or Path.cwd()).resolve()
            # Must end in .json
            if not str(manifest_path).endswith('.json'):
                raise HTTPException(400, "Manifest path must end in .json")
            _validate_path_security(manifest_path, base_dir)
        max_concurrency = int(payload.get("max_concurrency", 4))
        timeout = float(payload.get("timeout", 120.0))
        is_stream = bool(payload.get("stream", True))

        sse_exec = SSEExecutorV4(manager=manager, store=store)

        if is_stream:
            gen = sse_exec.execute_and_stream(
                session_id=session_id,
                input_payload=input_payload,
                manifest_path=manifest_path,
                max_concurrency=max_concurrency,
                timeout=timeout,
            )
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            return await sse_exec.execute_harness_call(
                session_id=session_id,
                input_payload=input_payload,
                manifest_path=manifest_path,
                max_concurrency=max_concurrency,
                timeout=timeout,
            )

    return app


# ---------------------------------------------------------------------------
# CLI Entrypoint for Standalone Server
# ---------------------------------------------------------------------------

def parse_server_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VEA v4 Online Graph & Database Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--db", default=":memory:", help="SQLite database path")
    return parser.parse_args()


def main() -> None:
    """Run server directly from CLI."""
    import uvicorn
    args = parse_server_args()
    app = create_v4_server(store_or_db=args.db)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
