"""Execution and SSE streaming routes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse

from framework.executor_v4 import ExecutorV4, GraphEventV4
from framework.server.schemas import WorkflowRunRequest
from framework.server.security import validate_path_security

router = APIRouter(tags=["execution"])


@router.post("/api/sessions/{session_id}/run")
async def run_session_workflow(
    session_id: str,
    request: Request,
    req: WorkflowRunRequest = Body(default_factory=WorkflowRunRequest),
) -> Dict[str, Any]:
    """Execute session graph with specified concurrency limit and broadcast events."""
    manager = request.app.state.manager
    store = request.app.state.store
    async with manager.get_session_lock(session_id):
        # Reload latest state from store before execution to avoid dirty in-memory state
        graph = manager.load_graph_from_store(session_id)
        executor = ExecutorV4(
            graph=graph,
            store=store,
            agent=request.app.state.agent,
            max_concurrency=req.max_concurrency,
            timeout=req.timeout,
            snapshot_manager=manager.snapshot_manager,
        )
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


@router.get("/api/sessions/{session_id}/events")
async def stream_session_events(session_id: str, request: Request) -> StreamingResponse:
    """SSE stream broadcasting execution events in real time."""
    manager = request.app.state.manager
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


@router.post("/api/sse/execute")
async def execute_via_sse(
    request: Request,
    payload: Dict[str, Any] = Body(default_factory=dict),
) -> Any:
    """Execute workflow via SSEExecutor with session routing and harness echo output."""
    from framework.sse_executor_v4 import SSEExecutorV4

    manager = request.app.state.manager
    store = request.app.state.store

    session_id = payload.get("session_id")
    input_payload = payload.get("input_payload")
    manifest_path = payload.get("manifest_path")
    if manifest_path:
        base_dir = Path(request.app.state.manifest_base_dir or Path.cwd()).resolve()
        # Must end in .json
        if not str(manifest_path).endswith('.json'):
            raise HTTPException(400, "Manifest path must end in .json")
        validate_path_security(manifest_path, base_dir)
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
