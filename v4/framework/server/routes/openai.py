"""OpenAI-compatible endpoints (/v1/chat/completions, /v1/models)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time as _time
from typing import Any, AsyncGenerator, Dict, List, Optional
import uuid as _uuid

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from framework.edge_v4 import ToolEdgeV4
from framework.edges.registry import is_tool_edge
from framework.executor_v4 import ExecutorV4
from framework.graph_v4 import DiscreteGraphLoaderV4, GraphV4
from framework.server.helpers import get_effective_catalog_dir
from framework.vertex_v4 import VertexAttributeV4, VertexStateV4, VertexStoreV4

logger = logging.getLogger("vertex_edge_agent.server_v4.routes.openai")

router = APIRouter(tags=["openai"])

# -- graph template registry (model name -> manifest/config) -----------
_graph_templates: Dict[str, Dict[str, Any]] = {}


def register_graph_template(
    name: str,
    manifest_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    _graph_templates[name] = {
        "manifest_path": manifest_path,
        "config": config,
    }


def _openai_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{_uuid.uuid4().hex[:24]}"


def _resolve_session_id(
    req_body: Dict[str, Any],
    request: Any,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Extract session ID from header, body extension, or derive from root message."""
    sid = None
    if hasattr(request, "headers"):
        sid = request.headers.get("x-session-id")
    if not sid:
        sid = req_body.get("session_id")
    if not sid:
        sid = req_body.get("user")
    if not sid and messages:
        from framework.http_executor_v4 import HttpHarnessExecutorV4
        sid = HttpHarnessExecutorV4.derive_session_id(messages)
    if not sid:
        sid = f"sess_{_uuid.uuid4().hex[:10]}"
    return sid


def _extract_user_content(messages: List[Dict[str, Any]]) -> str:
    """Extract the latest user message content."""
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"]
    # Fallback to last message
    if messages and messages[-1].get("content"):
        return messages[-1]["content"]
    return ""


def _format_edge_thinking(edge_results: Dict[str, Any]) -> str:
    """Format edge execution results into a readable thinking/reasoning trace.

    Each edge's output is shown as a numbered step with its status,
    duration, and a truncated preview of the output content.
    """
    if not edge_results:
        return ""

    lines = []
    step_num = 0
    for edge_id, result in edge_results.items():
        step_num += 1
        success = result.get("success", False)
        status = "✅" if success else "❌"
        error = result.get("error")
        output = result.get("output", "")
        metadata = result.get("metadata", {})
        duration = metadata.get("duration_ms", metadata.get("execution_time_ms"))

        line = f"[Step {step_num}] {status} **{edge_id}**"
        if duration:
            line += f" ({duration:.0f}ms)"
        lines.append(line)

        if success and output:
            # Truncate long outputs for readability
            preview = output[:500]
            if len(output) > 500:
                preview += f" ... (+{len(output) - 500} chars)"
            lines.append(f"  → `{preview}`")
        elif error:
            lines.append(f"  ✗ {error[:200]}")

        lines.append("")

    return "\n".join(lines)


def _get_end_vertex_content(
    _store: VertexStoreV4,
    _session_id: str,
    _graph: GraphV4,
) -> Optional[str]:
    """Retrieve content from END vertices or sink vertices."""
    for v in _graph.vertices.values():
        if v.has_attribute(VertexAttributeV4.END):
            rec = _store.get_vertex(_session_id, v.name)
            if rec and rec.content and rec.state == VertexStateV4.DATA_READY.value:
                return rec.content
    # Fallback: find vertices with no outgoing forward edges
    for v in _graph.vertices.values():
        outgoing_fwd = [
            e for e in _graph.get_outgoing_edges(v.name)
            if not e.is_reflexive
        ]
        incoming_fwd = [
            e for e in _graph.get_incoming_edges(v.name)
            if not e.is_reflexive
        ]
        if not outgoing_fwd and incoming_fwd:
            rec = _store.get_vertex(_session_id, v.name)
            if rec and rec.content and rec.state == VertexStateV4.DATA_READY.value:
                return rec.content
    return None


def _inject_user_input(
    _store: VertexStoreV4,
    _session_id: str,
    _graph: GraphV4,
    user_content: str,
) -> None:
    """Inject user content into START vertices."""
    for v in _graph.vertices.values():
        if v.has_attribute(VertexAttributeV4.START):
            _store.update_vertex_content(
                session_id=_session_id,
                name=v.name,
                content=user_content,
                state=VertexStateV4.DATA_READY.value,
            )


@router.get("/v1/models")
async def openai_list_models(request: Request) -> Dict[str, Any]:
    """OpenAI-compatible model listing.

    Returns registered graph templates and active sessions.
    """
    manager = request.app.state.manager
    models = []
    # Built-in templates
    for name in sorted(_graph_templates.keys()):
        models.append({
            "id": name,
            "object": "model",
            "created": int(_time.time()),
            "owned_by": "vea-v4",
        })
    # Active sessions are also addressable as "models"
    for sid in manager.list_active_sessions():
        if sid not in _graph_templates:
            models.append({
                "id": sid,
                "object": "model",
                "created": int(_time.time()),
                "owned_by": "vea-v4-session",
            })
    # Always include a default entry
    if not any(m["id"] == "default" for m in models):
        models.insert(0, {
            "id": "default",
            "object": "model",
            "created": int(_time.time()),
            "owned_by": "vea-v4",
        })
    return {"object": "list", "data": models}


@router.post("/v1/chat/completions")
async def openai_chat_completions(
    request: Request,
    req_body: Dict[str, Any] = Body(...),
) -> Any:
    """OpenAI-compatible chat completions endpoint on the V4 engine.

    Supports three execution modes via ``vea_execution_mode``:
    - ``"full"`` (default): Run graph to completion, return final content.
    - ``"per_tier"``: Execute one topological tier, return tool_call if
      more work remains (for agent harness hop-by-hop loops).
    - ``"per_edge"``: Execute a single edge per call (finest granularity).

    Session state is preserved across requests via ``session_id``
    (in body, ``user`` field, or ``X-Session-ID`` header).
    """
    manager = request.app.state.manager
    store = request.app.state.store
    agent = request.app.state.agent

    # Parse request fields
    model = req_body.get("model", "default")
    messages = req_body.get("messages", [])
    is_stream = req_body.get("stream", False)
    execution_mode = req_body.get("vea_execution_mode", "full")

    if not messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")

    session_id = _resolve_session_id(req_body, request, messages=messages)
    request_id = _openai_id()

    # Resolve or create graph for this session
    template = _graph_templates.get(model)
    manifest_path = None
    if template and template.get("manifest_path"):
        manifest_path = template["manifest_path"]

    # Load graph from manifest if needed, otherwise from store/memory
    if manifest_path and Path(manifest_path).exists():
        # Only load from manifest if session doesn't already exist
        existing = manager._graphs.get(session_id)
        if existing is None or not existing.vertices:
            graph = DiscreteGraphLoaderV4.load_from_manifest(
                manifest_path=manifest_path,
                override_session_id=session_id,
            )
            manager._graphs[session_id] = graph
            DiscreteGraphLoaderV4.populate_store(graph, store)
            graph = manager.load_graph_from_store(session_id)
        else:
            graph = manager.load_graph_from_store(session_id)
    elif template and template.get("config"):
        existing = manager._graphs.get(session_id)
        if existing is None or not existing.vertices:
            graph = DiscreteGraphLoaderV4.load_from_dict(
                template["config"],
                override_session_id=session_id,
            )
            manager._graphs[session_id] = graph
            DiscreteGraphLoaderV4.populate_store(graph, store)
            graph = manager.load_graph_from_store(session_id)
        else:
            graph = manager.load_graph_from_store(session_id)
    # Check for dynamic tool routing request
    user_content = _extract_user_content(messages)
    dynamic_route = req_body.get("vea_dynamic_route", False) or model in ("dynamic-router", "auto-router")
    if dynamic_route and user_content:
        cat_dir = get_effective_catalog_dir(request.app.state.tool_catalog_dir)
        if cat_dir:
            try:
                from framework.tool_catalog.router import classify_intent_with_llm
                tool_name, _ = await classify_intent_with_llm(user_content, cat_dir)
            except Exception:
                from framework.tool_catalog.router import classify_intent
                tool_name = classify_intent(user_content)

            tool_manifest_file = cat_dir / f"{tool_name}.json"
            if tool_manifest_file.exists():
                try:
                    tool_subgraph = DiscreteGraphLoaderV4.load_from_manifest(tool_manifest_file)
                    store.clear_session(session_id)
                    manager._graphs.pop(session_id, None)
                    manager.add_or_update_vertex(
                        session_id=session_id,
                        name="v_user_query",
                        content=user_content,
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
                    entry_v = tool_subgraph.metadata.get("entry_vertex", "sub_in")
                    exit_v = tool_subgraph.metadata.get("exit_vertex", "sub_out")
                    manager.insert_subgraph(
                        session_id=session_id,
                        subgraph=tool_subgraph,
                        incoming_bindings={"v_user_query": entry_v},
                        outgoing_bindings={exit_v: "v_final_output"},
                        name_prefix=f"{tool_name}_",
                    )
                    graph = manager.load_graph_from_store(session_id)
                except Exception as exc:
                    logger.warning("Dynamic tool routing subgraph load failed: %s", exc)
                    graph = manager.get_or_create_graph(session_id)
            else:
                graph = manager.get_or_create_graph(session_id)
        else:
            graph = manager.get_or_create_graph(session_id)
    else:
        graph = manager.get_or_create_graph(session_id)

    # Only inject if the latest message is a user message (not a tool result)
    latest_msg = messages[-1] if messages else {}
    if latest_msg.get("role") in ("user", "system") and user_content:
        _inject_user_input(store, session_id, graph, user_content)

    # --- Streaming mode ---
    if is_stream:
        async def _openai_v4_stream() -> AsyncGenerator[str, None]:
            # Initial role chunk
            init_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(_time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(init_chunk)}\n\n"

            executor = ExecutorV4(
                graph=graph,
                store=store,
                agent=agent,
                max_concurrency=4,
                timeout=120.0,
            )
            manager.register_executor(session_id, executor)
            try:
                async for event in executor.stream():
                    manager.broadcast_event(session_id, event)
                    if event.event_type == "edge_completed":
                        payload = event.payload or {}
                        output = payload.get("output")
                        if isinstance(output, str) and output:
                            # Edge output goes to reasoning_content
                            delta_chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(_time.time()),
                                "model": model,
                                "choices": [{"index": 0, "delta": {"reasoning_content": output}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(delta_chunk)}\n\n"
            finally:
                manager.unregister_executor(session_id)

            # Final content from END vertex goes to content
            final_content = _get_end_vertex_content(store, session_id, graph)
            if final_content:
                final_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(_time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": final_content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"

            # Done chunk
            done_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(_time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _openai_v4_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- Non-streaming: full, harness, or hop-by-hop ---
    has_tool_edges = any(
        is_tool_edge(e)
        for e in graph.edges.values()
    )
    is_tool_reply = latest_msg.get("role") == "tool"

    async with manager.get_session_lock(session_id):
        # If graph has tool edges or incoming request is a tool execution result,
        # execute via HttpHarnessExecutorV4 (passive request-driven execution)
        if has_tool_edges or is_tool_reply or execution_mode in ("harness", "step"):
            from framework.http_executor_v4 import HttpHarnessExecutorV4
            http_exec = HttpHarnessExecutorV4(store=store, agent=agent)
            step_res = await http_exec.step(session_id, graph, messages, model=model)
            return JSONResponse(step_res)

        executor = ExecutorV4(
            graph=graph,
            store=store,
            agent=agent,
            max_concurrency=4,
            timeout=120.0,
        )
        manager.register_executor(session_id, executor)

        def _get_usage_dict() -> Dict[str, int]:
            try:
                summ = store.get_edge_metrics_summary(session_id)
                return {
                    "prompt_tokens": summ.get("total_prompt_tokens", 0),
                    "completion_tokens": summ.get("total_completion_tokens", 0),
                    "total_tokens": summ.get("total_tokens", 0),
                }
            except Exception:
                return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            if execution_mode in ("per_tier", "per_edge"):
                # Hop-by-hop: execute one tier, then return
                hop_result = await executor.run_one_tier()

                # Check if graph is now terminal
                is_done = executor.is_terminal()
                if is_done and hop_result.success:
                    # Graph complete — return final content
                    final_content = _get_end_vertex_content(store, session_id, graph) or ""
                    return JSONResponse({
                        "id": request_id,
                        "object": "chat.completion",
                        "created": int(_time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": final_content,
                            },
                            "finish_reason": "stop",
                        }],
                        "usage": _get_usage_dict(),
                        "system_fingerprint": f"vea-v4-{session_id}",
                    })
                else:
                    # More work remains — return advance_graph tool_call
                    advance_args = {
                        "session_id": session_id,
                        "completed_edges": hop_result.completed_edges,
                        "vertex_states": hop_result.vertex_states,
                        "execution_time": hop_result.execution_time,
                        "errors": hop_result.errors,
                    }
                    return JSONResponse({
                        "id": request_id,
                        "object": "chat.completion",
                        "created": int(_time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": f"call_{_uuid.uuid4().hex[:10]}",
                                    "type": "function",
                                    "function": {
                                        "name": "advance_graph",
                                        "arguments": json.dumps(advance_args),
                                    },
                                }],
                            },
                            "finish_reason": "tool_calls",
                        }],
                        "usage": _get_usage_dict(),
                        "system_fingerprint": f"vea-v4-{session_id}",
                    })
            else:
                # Full execution mode — run to completion
                result = await executor.run()
                final_content = _get_end_vertex_content(store, session_id, graph) or ""
                if not final_content and result.vertex_contents:
                    # Fallback: use last vertex content
                    for vname in reversed(list(result.vertex_contents.keys())):
                        if result.vertex_contents[vname]:
                            final_content = result.vertex_contents[vname]
                            break

                # Build thinking trace from edge results
                thinking = _format_edge_thinking(result.edge_results)

                return JSONResponse({
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(_time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_content,
                            "reasoning_content": thinking,
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": _get_usage_dict(),
                    "system_fingerprint": f"vea-v4-{session_id}",
                })
        finally:
            manager.unregister_executor(session_id)
