"""Graph inspection, online mutation, reconnection, reentry, and subgraph routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from fastapi import APIRouter, Body, HTTPException, Query, Request

from framework.graph_v4 import DiscreteGraphLoaderV4, GraphV4
from framework.server.schemas import (
    EdgeCreateOrUpdateRequest,
    EdgeReconnectRequest,
    SubgraphAddRequest,
    SubgraphInsertRequest,
    SubgraphSpliceRequest,
    VertexCreateOrUpdateRequest,
    VertexReentryRequest,
)
from framework.server.security import (
    resolve_manifest_path,
    validate_path_security,
)
from framework.utils.paths import default_manifest_base_dir

router = APIRouter(tags=["graph"])


@router.get("/api/sessions/{session_id}/graph")
async def get_session_graph(session_id: str, request: Request) -> Dict[str, Any]:
    """Retrieve the current graph structure, endpoints, and DAG validation."""
    manager = request.app.state.manager
    store = request.app.state.store
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


@router.get("/api/sessions/{session_id}/graph/nodes")
async def list_session_loaded_nodes(session_id: str, request: Request) -> Dict[str, Any]:
    """List all loaded vertices with provenance tracking for a session."""
    manager = request.app.state.manager
    nodes = manager.list_loaded_nodes(session_id)
    return {"session_id": session_id, "nodes": nodes, "total": len(nodes)}


@router.get("/api/sessions/{session_id}/graph/nodes/{name}/relationships")
async def get_session_node_relationships(session_id: str, name: str, request: Request) -> Dict[str, Any]:
    """Retrieve predecessors, successors, and relationship details for a single vertex."""
    manager = request.app.state.manager
    try:
        return manager.get_node_relationships(session_id, name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Vertex '{name}' not found in session graph")


@router.get("/api/sessions/{session_id}/graph/relationships")
async def get_session_graph_relationships(session_id: str, request: Request) -> Dict[str, Any]:
    """Retrieve full graph relationship matrix, adjacency lists, and roots/sinks."""
    manager = request.app.state.manager
    return manager.get_graph_relationships(session_id)


@router.get("/api/sessions/{session_id}/graph/dump")
@router.post("/api/sessions/{session_id}/graph/dump")
async def dump_session_graph(
    session_id: str,
    request: Request,
    path: Optional[str] = Query(default=None, description="Optional filesystem path to dump JSON"),
    payload: Optional[Dict[str, Any]] = Body(default=None),
) -> Dict[str, Any]:
    """Dump complete session graph structure and component metadata."""
    manager = request.app.state.manager
    target_path = path
    if not target_path and payload and isinstance(payload, dict):
        target_path = payload.get("path")
    if target_path:
        base_dir = Path(request.app.state.manifest_base_dir or default_manifest_base_dir()).resolve()
        if not str(target_path).endswith('.json'):
            raise HTTPException(400, "Dump path must end in .json")
        # Use the resolved path so the check cannot be bypassed between
        # validation and use.
        target_path = str(validate_path_security(target_path, base_dir))
    dumped = manager.dump_graph(session_id, path=target_path)
    return {"status": "dumped", "session_id": session_id, "graph": dumped}


@router.post("/api/sessions/{session_id}/graph/vertices")
async def create_or_update_vertex(
    session_id: str,
    req: VertexCreateOrUpdateRequest,
    request: Request,
) -> Dict[str, Any]:
    """Add or update a vertex online."""
    manager = request.app.state.manager
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


@router.delete("/api/sessions/{session_id}/graph/vertices/{name}")
async def delete_vertex(session_id: str, name: str, request: Request) -> Dict[str, Any]:
    """Delete a vertex online."""
    manager = request.app.state.manager
    async with manager.get_session_lock(session_id):
        deleted = manager.delete_vertex(session_id, name)
        return {"deleted": deleted, "name": name}


@router.post("/api/sessions/{session_id}/graph/edges")
async def create_or_update_edge(
    session_id: str,
    req: EdgeCreateOrUpdateRequest,
    request: Request,
) -> Dict[str, Any]:
    """Add or update an edge online."""
    manager = request.app.state.manager
    async with manager.get_session_lock(session_id):
        try:
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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


@router.delete("/api/sessions/{session_id}/graph/edges/{edge_id}")
async def delete_edge(session_id: str, edge_id: str, request: Request) -> Dict[str, Any]:
    """Delete an edge online."""
    manager = request.app.state.manager
    async with manager.get_session_lock(session_id):
        deleted = manager.delete_edge(session_id, edge_id)
        return {"deleted": deleted, "edge_id": edge_id}


@router.post("/api/sessions/{session_id}/graph/validate")
async def validate_graph(session_id: str, request: Request) -> Dict[str, Any]:
    """Validate DAG topology and tiers for session graph."""
    manager = request.app.state.manager
    return manager.validate_graph(session_id)


@router.post("/api/sessions/{session_id}/graph/vertices/{name}/reenter")
async def reenter_vertex_route(
    session_id: str,
    name: str,
    request: Request,
    req: VertexReentryRequest = Body(default_factory=VertexReentryRequest),
) -> Dict[str, Any]:
    """Re-enter a vertex for re-execution, resetting all affected downstream vertices."""
    manager = request.app.state.manager
    async with manager.get_session_lock(session_id):
        cancelled_edges: List[str] = []
        executor = manager.get_running_executor(session_id)
        if executor:
            graph = manager.get_or_create_graph(session_id)
            affected_set: Set[str] = set()
            affected_set.add(name)
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


@router.patch("/api/sessions/{session_id}/graph/edges/{edge_id}/reconnect")
async def reconnect_edge_route(
    session_id: str,
    edge_id: str,
    req: EdgeReconnectRequest,
    request: Request,
) -> Dict[str, Any]:
    """Dynamically reconnect an existing edge to new endpoints."""
    manager = request.app.state.manager
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


@router.post("/api/sessions/{session_id}/graph/subgraphs/splice")
async def splice_subgraph_route(
    session_id: str,
    req: SubgraphSpliceRequest,
    request: Request,
) -> Dict[str, Any]:
    """Splice (inline) a subgraph in place of an existing vertex."""
    manager = request.app.state.manager
    async with manager.get_session_lock(session_id):
        subgraph: GraphV4
        if req.subgraph_manifest:
            manifest_path = resolve_manifest_path(
                req.subgraph_manifest, request.app.state.manifest_base_dir
            )
            subgraph = DiscreteGraphLoaderV4.load_from_manifest(manifest_path)
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


@router.post("/api/sessions/{session_id}/graph/subgraphs/insert")
async def insert_subgraph_route(
    session_id: str,
    req: SubgraphInsertRequest,
    request: Request,
) -> Dict[str, Any]:
    """Insert an independent subgraph with explicit boundary bindings."""
    manager = request.app.state.manager
    async with manager.get_session_lock(session_id):
        subgraph: GraphV4
        if req.subgraph_manifest:
            manifest_path = resolve_manifest_path(
                req.subgraph_manifest, request.app.state.manifest_base_dir
            )
            subgraph = DiscreteGraphLoaderV4.load_from_manifest(manifest_path)
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


@router.post("/api/sessions/{session_id}/graph/subgraphs/add")
@router.post("/api/sessions/{session_id}/graph/subgraphs")
async def add_subgraph_route(
    session_id: str,
    req: SubgraphAddRequest,
    request: Request,
) -> Dict[str, Any]:
    """Add and join an arbitrary subgraph into session graph with optional prefix, connections, and bindings."""
    manager = request.app.state.manager
    async with manager.get_session_lock(session_id):
        subgraph: GraphV4
        if req.subgraph_manifest:
            manifest_path = resolve_manifest_path(
                req.subgraph_manifest, request.app.state.manifest_base_dir
            )
            subgraph = DiscreteGraphLoaderV4.load_from_manifest(manifest_path)
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
