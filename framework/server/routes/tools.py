"""Dynamic tool catalog and intent routing routes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Request

from framework.executor_v4 import ExecutorV4
from framework.graph_v4 import DiscreteGraphLoaderV4
from framework.server.helpers import get_effective_catalog_dir, list_tool_catalog
from framework.server.schemas import RouteAndRunRequest
from framework.vertex_v4 import VertexStateV4

logger = logging.getLogger("vertex_edge_agent.server_v4.routes.tools")

router = APIRouter(tags=["tools"])


@router.get("/api/tool-catalog")
async def list_tool_catalog_endpoint(request: Request) -> Dict[str, Any]:
    """List all dynamic tool subgraphs registered in the tool catalog."""
    cat_dir = get_effective_catalog_dir(request.app.state.tool_catalog_dir)
    if not cat_dir:
        return {"catalog_dir": None, "count": 0, "tools": []}
    tools = list_tool_catalog(cat_dir)
    return {
        "catalog_dir": str(cat_dir),
        "count": len(tools),
        "tools": tools,
    }


@router.get("/api/tool-catalog/{tool_id}")
async def get_tool_manifest_endpoint(tool_id: str, request: Request) -> Dict[str, Any]:
    """Get detailed manifest for a specific tool in the catalog."""
    cat_dir = get_effective_catalog_dir(request.app.state.tool_catalog_dir)
    if not cat_dir:
        raise HTTPException(404, "Tool catalog not configured or directory missing")
    fpath = cat_dir / f"{tool_id}.json"
    if not fpath.exists():
        raise HTTPException(404, f"Tool '{tool_id}' not found in catalog")
    try:
        manifest = json.loads(fpath.read_text(encoding="utf-8"))
        return {
            "id": tool_id,
            "manifest_path": str(fpath.resolve()),
            "manifest": manifest,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to parse tool manifest '{tool_id}': {e}")


@router.post("/api/sessions/{session_id}/route-and-run")
async def route_and_run_session_endpoint(
    session_id: str,
    req: RouteAndRunRequest,
    request: Request,
) -> Dict[str, Any]:
    """Dynamically route user task to catalog tool subgraph, splice into DAG, and execute."""
    cat_dir = get_effective_catalog_dir(request.app.state.tool_catalog_dir)
    if not cat_dir:
        raise HTTPException(500, "Tool catalog directory not configured or does not exist")

    # 1. Determine tool via explicit override, LLM reasoning, or heuristic
    selected_tool = req.tool_id
    route_reason = "Explicit tool specified by user request"
    if not selected_tool:
        if req.use_llm:
            try:
                from examples.dynamic_tool_library.tool_scripts import classify_intent_with_llm
                selected_tool, route_reason = await classify_intent_with_llm(
                    query=req.task,
                    catalog_dir=cat_dir,
                    api_key=req.api_key,
                    base_url=req.base_url or "https://sensenova.moonchan.xyz/v1/chat/completions",
                    model=req.model or "sensenova-6.8-flash-lite",
                    timeout=12.0,
                )
            except Exception as e:
                logger.warning("LLM router failed, falling back to heuristic: %s", e)
                from examples.dynamic_tool_library.tool_scripts import classify_intent
                selected_tool = classify_intent(req.task)
                route_reason = f"Heuristic fallback (LLM exception: {e})"
        else:
            from examples.dynamic_tool_library.tool_scripts import classify_intent
            selected_tool = classify_intent(req.task)
            route_reason = "Rule-based heuristic classifier"

    tool_manifest_file = cat_dir / f"{selected_tool}.json"
    if not tool_manifest_file.exists():
        raise HTTPException(404, f"Tool manifest '{selected_tool}' not found in catalog {cat_dir}")

    try:
        tool_subgraph = DiscreteGraphLoaderV4.load_from_manifest(tool_manifest_file)
    except Exception as e:
        raise HTTPException(500, f"Failed to load tool subgraph from {tool_manifest_file}: {e}")

    manager = request.app.state.manager
    store = request.app.state.store

    async with manager.get_session_lock(session_id):
        # Reset previous session graph store records for clean dynamic execution
        store.clear_session(session_id)
        manager._graphs.pop(session_id, None)

        # Base parent vertices
        manager.add_or_update_vertex(
            session_id=session_id,
            name="v_user_query",
            content=req.task,
            state=VertexStateV4.DATA_READY.value,
            attributes=["start"],
        )
        manager.add_or_update_vertex(
            session_id=session_id,
            name="v_final_output",
            content="",
            state=VertexStateV4.TODO.value,
            attributes=["end"],
        )

        # Splice/insert subgraph
        entry_v = tool_subgraph.metadata.get("entry_vertex", "sub_in")
        exit_v = tool_subgraph.metadata.get("exit_vertex", "sub_out")
        insert_res = manager.insert_subgraph(
            session_id=session_id,
            subgraph=tool_subgraph,
            incoming_bindings={"v_user_query": entry_v},
            outgoing_bindings={exit_v: "v_final_output"},
            name_prefix=f"{selected_tool}_",
        )

        # Load fresh graph
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

        exec_res = executor.result
        v_out = store.get_vertex(session_id, "v_final_output")
        output_content = v_out.content if v_out else ""

        snapshots = (
            manager.snapshot_manager.list_snapshots(session_id)
            if manager.snapshot_manager
            else []
        )

        return {
            "session_id": session_id,
            "task": req.task,
            "routed_tool": selected_tool,
            "route_reason": route_reason,
            "success": exec_res.success,
            "output": output_content,
            "execution_time": exec_res.execution_time,
            "completed_edges": exec_res.completed_edges,
            "errors": exec_res.errors,
            "inserted_vertices": insert_res.get("inserted_vertices", []),
            "inserted_edges": insert_res.get("inserted_edges", []),
            "snapshot_count": len(snapshots),
            "snapshots": snapshots,
        }
