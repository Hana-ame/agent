"""Database and metrics inspection routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Query, Request

router = APIRouter(tags=["db"])


@router.get("/api/db/stats")
async def get_db_stats(request: Request) -> Dict[str, Any]:
    """Return overall database statistics across all sessions."""
    store = request.app.state.store
    return store.get_db_stats()


@router.get("/api/db/sessions")
async def list_sessions(request: Request) -> List[str]:
    """List distinct session IDs."""
    manager = request.app.state.manager
    return manager.list_active_sessions()


@router.get("/api/db/sessions/{session_id}/vertices")
async def get_db_vertices(
    session_id: str,
    request: Request,
    state: Optional[str] = Query(None, description="Optional state filter"),
) -> List[Dict[str, Any]]:
    """Fetch raw vertices for a session directly from SQLite."""
    store = request.app.state.store
    records = store.list_vertices(session_id=session_id, state=state)
    return [r.to_dict() for r in records]


@router.get("/api/db/sessions/{session_id}/staging")
async def get_db_staging(
    session_id: str,
    request: Request,
    key: Optional[str] = Query(None),
    edge_id: Optional[str] = Query(None),
    vertex_name: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """Query raw records from session_staging table."""
    store = request.app.state.store
    records = store.get_staged(
        session_id=session_id,
        key=key,
        edge_id=edge_id,
        vertex_name=vertex_name,
    )
    return [r.to_dict() for r in records]


@router.post("/api/db/sessions/{session_id}/clear")
async def clear_session_db(session_id: str, request: Request) -> Dict[str, str]:
    """Purge all records for a session from SQLite and in-memory state."""
    store = request.app.state.store
    manager = request.app.state.manager
    store.clear_session(session_id)
    manager._graphs.pop(session_id, None)
    manager._event_broadcasters.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@router.get("/api/db/sessions/{session_id}/metrics")
@router.get("/api/sessions/{session_id}/metrics")
async def get_session_metrics(
    session_id: str,
    request: Request,
    edge_id: Optional[str] = Query(None, description="Filter by edge ID"),
    limit: Optional[int] = Query(None, description="Limit records returned"),
) -> Dict[str, Any]:
    """Fetch edge execution metrics and summary benchmarks for a session."""
    store = request.app.state.store
    records = store.list_edge_metrics(session_id=session_id, edge_id=edge_id, limit=limit)
    summary = store.get_edge_metrics_summary(session_id=session_id)
    return {
        "session_id": session_id,
        "summary": summary,
        "metrics": [r.to_dict() for r in records],
    }


@router.get("/api/metrics/summary")
@router.get("/api/db/metrics/summary")
async def get_global_metrics_summary(request: Request) -> Dict[str, Any]:
    """Return aggregate benchmarking metrics across all sessions."""
    store = request.app.state.store
    return store.get_edge_metrics_summary(session_id=None)
