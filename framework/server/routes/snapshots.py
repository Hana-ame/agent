"""Graph snapshot and time-travel rollback routes."""

from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import APIRouter, Body, HTTPException, Request

router = APIRouter(tags=["snapshots"])


@router.get("/api/sessions/{session_id}/snapshots")
async def list_session_snapshots(session_id: str, request: Request) -> Dict[str, Any]:
    """List all historical complete graph snapshots for a session."""
    manager = request.app.state.manager
    if not manager.snapshot_manager:
        return {"session_id": session_id, "snapshots": []}
    snapshots = manager.snapshot_manager.list_snapshots(session_id)
    return {"session_id": session_id, "snapshots": snapshots}


@router.get("/api/sessions/{session_id}/snapshots/{step}")
async def get_session_snapshot(session_id: str, step: int, request: Request) -> Dict[str, Any]:
    """Get the complete graph specification for a specific historical step."""
    manager = request.app.state.manager
    if not manager.snapshot_manager:
        raise HTTPException(status_code=404, detail="Snapshot manager not configured")
    data = manager.snapshot_manager.get_snapshot_data(session_id, step)
    if not data:
        raise HTTPException(status_code=404, detail=f"Snapshot step {step} not found for session {session_id}")
    return data


@router.post("/api/sessions/{session_id}/snapshots")
async def create_session_snapshot(
    session_id: str,
    request: Request,
    payload: Optional[Dict[str, Any]] = Body(default=None),
) -> Dict[str, Any]:
    """Trigger an explicit snapshot of the current complete graph state."""
    manager = request.app.state.manager
    trigger = (payload or {}).get("trigger", "manual_api")
    path = manager.record_snapshot(session_id, trigger=trigger)
    if not path:
        raise HTTPException(status_code=500, detail="Failed to save snapshot")
    return {"status": "saved", "path": str(path), "session_id": session_id}


@router.post("/api/sessions/{session_id}/snapshots/{step}/restore")
async def restore_session_snapshot(session_id: str, step: int, request: Request) -> Dict[str, Any]:
    """Roll back and restore the complete graph to a historical snapshot step."""
    manager = request.app.state.manager
    if not manager.snapshot_manager:
        raise HTTPException(status_code=404, detail="Snapshot manager not configured")
    async with manager.get_session_lock(session_id):
        graph = manager.snapshot_manager.restore_snapshot(session_id, step, store=manager.store)
        if not graph:
            raise HTTPException(status_code=404, detail=f"Snapshot step {step} could not be restored")
        # Invalidate in-memory graph cache to reload restored state
        manager._graphs.pop(session_id, None)
        return {"status": "restored", "session_id": session_id, "step": step, "vertices": len(graph.vertices)}
