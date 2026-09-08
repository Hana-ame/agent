"""V4 HTTP Harness-Driven Executor.

Provides passive, request-driven graph execution tailored for Agent Harness frameworks.
Every incoming HTTP request advances the graph by one step/edge:
- If an edge requires external tools (ToolEdgeV4 / LLMToolEdgeV4), it yields the tool_call
  to the harness with finish_reason="tool_calls".
- When the harness executes the tool in its sandbox and calls the API with role: "tool",
  the executor settles the result, transitions the vertex to DATA_READY, and advances
  to the next eligible edge.
- Observability: message.content reports what each edge did and what data it produced.
- Zero internal while-loops; execution is completely driven by incoming HTTP requests.
- Purely handshake-driven; no tiering concepts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from framework.edge_v4 import (
    CodeEdgeV4,
    EdgeResultV4,
    EdgeV4,
    LLMEdgeV4,
    LLMToolEdgeV4,
    ReflexiveEdgeV4,
    ToolEdgeV4,
)
from framework.graph_v4 import GraphV4
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.http_executor_v4")


class HttpHarnessExecutorV4:
    """Request-driven executor that advances the graph step-by-step per HTTP turn."""

    def __init__(
        self,
        store: VertexStoreV4,
        agent: Optional[Any] = None,
    ):
        self.store = store
        self.agent = agent

    @staticmethod
    def derive_session_id(
        messages: List[Dict[str, Any]],
        provided_session_id: Optional[str] = None,
        header_session_id: Optional[str] = None,
        user: Optional[str] = None,
    ) -> str:
        """Derive or resolve deterministic session ID.

        If a standard OpenAI client / harness does not pass custom headers,
        the session is deterministically derived from the root user message
        (hash of the first prompt), ensuring all subsequent turns in the same
        harness conversation seamlessly bind to the same SQLite graph session.
        """
        if header_session_id:
            return header_session_id
        if provided_session_id:
            return provided_session_id
        if user:
            return user

        # Extract root conversation prompt for fingerprinting
        first_content = ""
        for msg in messages:
            if msg.get("role") in ("user", "system") and msg.get("content"):
                first_content = str(msg["content"]).strip()
                break

        if first_content:
            digest = hashlib.sha256(first_content.encode("utf-8")).hexdigest()[:12]
            return f"sess_{digest}"

        return f"sess_{uuid.uuid4().hex[:10]}"

    def _get_eligible_edges(self, session_id: str, graph: GraphV4) -> List[Tuple[int, int, EdgeV4]]:
        """Identify edges whose trigger prerequisites are met.

        Ranked strictly by:
          1. Urgency rank (0: reflexive recovery, 1: todo urgent, 2: todo)
          2. Edge priority descending (-edge.priority)
          3. Edge ID ascending
        No tiering concept.
        """
        candidates: List[Tuple[int, int, EdgeV4]] = []

        for edge in graph.edges.values():
            edge_prio = edge.priority

            if hasattr(graph, "is_vertex_active"):
                if not graph.is_vertex_active(edge.input_vertex) or not graph.is_vertex_active(edge.output_vertex):
                    continue

            if edge.is_reflexive or isinstance(edge, ReflexiveEdgeV4):
                target_v = self.store.get_vertex(session_id, edge.output_vertex)
                trigger_state = getattr(edge, "trigger_state", VertexStateV4.REJECT.value)
                if target_v and target_v.state == trigger_state:
                    candidates.append((0, -edge_prio, edge))
            else:
                in_v = self.store.get_vertex(session_id, edge.input_vertex)
                out_v = self.store.get_vertex(session_id, edge.output_vertex)
                if in_v and out_v:
                    upstream_ready = in_v.state == VertexStateV4.DATA_READY.value
                    if upstream_ready:
                        if out_v.state == VertexStateV4.TODO_URGENT.value:
                            candidates.append((1, -edge_prio, edge))
                        elif out_v.state == VertexStateV4.TODO.value:
                            candidates.append((2, -edge_prio, edge))

        candidates.sort(key=lambda item: (item[0], item[1], item[2].id))
        return candidates

    def _is_terminal(self, session_id: str, graph: GraphV4) -> Tuple[bool, bool]:
        """Check if graph has completed or deadlocked."""
        all_vertices = self.store.list_vertices(session_id)
        if not all_vertices:
            return True, True

        states = {v.name: v.state for v in all_vertices}

        def _is_active(name: str) -> bool:
            if hasattr(graph, "is_vertex_active"):
                return graph.is_vertex_active(name)
            return True

        end_vertices = [
            v for v in all_vertices
            if _is_active(v.name)
            and (
                v.has_attribute(VertexAttributeV4.END)
                or (
                    not [e for e in graph.get_outgoing_edges(v.name) if not e.is_reflexive]
                    and [e for e in graph.get_incoming_edges(v.name) if not e.is_reflexive]
                )
            )
        ]

        has_forbidden = any(
            s == VertexStateV4.FORBIDDEN.value
            for v_name, s in states.items()
            if _is_active(v_name)
        )

        if end_vertices and all(v.state == VertexStateV4.DATA_READY.value for v in end_vertices) and not has_forbidden:
            return True, True

        has_pending = any(
            s in (VertexStateV4.TODO.value, VertexStateV4.TODO_URGENT.value, VertexStateV4.REJECT.value)
            for v_name, s in states.items()
            if _is_active(v_name)
        )

        if not has_pending:
            is_success = (
                not has_forbidden
                and all(v.state == VertexStateV4.DATA_READY.value for v in end_vertices)
            ) if end_vertices else not has_forbidden
            return True, is_success

        eligible = self._get_eligible_edges(session_id, graph)
        if not eligible:
            logger.warning("[HttpHarnessExecutorV4] Deadlock: pending demands exist but no eligible edges.")
            return True, False

        return False, False

    def _get_end_vertex_content(self, session_id: str, graph: GraphV4) -> str:
        """Fetch content from END or sink vertices."""
        for v in graph.vertices.values():
            if v.has_attribute(VertexAttributeV4.END):
                rec = self.store.get_vertex(session_id, v.name)
                if rec and rec.content and rec.state == VertexStateV4.DATA_READY.value:
                    return rec.content

        for v in graph.vertices.values():
            outgoing = [e for e in graph.get_outgoing_edges(v.name) if not e.is_reflexive]
            incoming = [e for e in graph.get_incoming_edges(v.name) if not e.is_reflexive]
            if not outgoing and incoming:
                rec = self.store.get_vertex(session_id, v.name)
                if rec and rec.content and rec.state == VertexStateV4.DATA_READY.value:
                    return rec.content

        all_v = self.store.list_vertices(session_id)
        for v in reversed(all_v):
            if v.content:
                return v.content
        return ""

    def _get_usage(self, session_id: str) -> Dict[str, int]:
        """Aggregate total usage tokens for the session from the edge_metrics table."""
        try:
            summary = self.store.get_edge_metrics_summary(session_id)
            return {
                "prompt_tokens": summary.get("total_prompt_tokens", 0),
                "completion_tokens": summary.get("total_completion_tokens", 0),
                "total_tokens": summary.get("total_tokens", 0),
            }
        except Exception:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def step(
        self,
        session_id: str,
        graph: GraphV4,
        messages: List[Dict[str, Any]],
        model: str = "default",
    ) -> Dict[str, Any]:
        """Execute one pulse of the graph driven by the incoming harness request.

        Returns an OpenAI-compliant chat completion dictionary.
        """
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        edge_reports: List[str] = []
        latest_msg = messages[-1] if messages else {}

        # -------------------------------------------------------------------
        # Phase 1: Handle Tool Result (if harness returned role: "tool")
        # -------------------------------------------------------------------
        if latest_msg.get("role") == "tool":
            tool_call_id = latest_msg.get("tool_call_id")
            tool_content = str(latest_msg.get("content", ""))

            # Query pending tool call from staging table
            staged = self.store.get_staged(session_id=session_id, key="pending_tool_call")
            if staged:
                latest_staged = staged[-1]
                try:
                    pending_info = json.loads(latest_staged.value)
                except Exception:
                    pending_info = {}

                edge_id = pending_info.get("edge_id")
                edge = graph.edges.get(edge_id) if edge_id else None

                if edge:
                    pending_start = pending_info.get("start_time")
                    elapsed_ms = (time.perf_counter() - float(pending_start)) * 1000.0 if pending_start else 0.0

                    # Settle the edge with the tool output from harness
                    if isinstance(edge, ToolEdgeV4):
                        res = await edge.run(session_id=session_id, store=self.store, tool_output=tool_content)
                    else:
                        self.store.apply_merge_strategy(
                            session_id=session_id,
                            name=edge.output_vertex,
                            incoming_content=tool_content,
                        )
                        self.store.update_vertex_state(session_id, edge.output_vertex, VertexStateV4.DATA_READY.value)
                        self.store.increment_processed_count(session_id, edge.output_vertex)
                        res = EdgeResultV4(edge_id=edge.id, success=True, output=tool_content)

                    # Record edge metric in store
                    try:
                        p_tok = int(res.metadata.get("prompt_tokens") or res.metadata.get("usage", {}).get("prompt_tokens", 0))
                        c_tok = int(res.metadata.get("completion_tokens") or res.metadata.get("usage", {}).get("completion_tokens", 0))
                        t_tok = int(res.metadata.get("total_tokens") or res.metadata.get("usage", {}).get("total_tokens", p_tok + c_tok))
                        cost = float(res.metadata.get("cost_usd") or res.metadata.get("cost", 0.0))
                        self.store.record_edge_metric(
                            session_id=session_id,
                            edge_id=edge.id,
                            edge_type=edge.type,
                            input_vertex=edge.input_vertex,
                            output_vertex=edge.output_vertex,
                            execution_time_ms=elapsed_ms,
                            prompt_tokens=p_tok,
                            completion_tokens=c_tok,
                            total_tokens=t_tok,
                            cost_usd=cost,
                            success=res.success,
                            error=res.error,
                            metadata={"tool_call_id": tool_call_id, **res.metadata},
                        )
                    except Exception as err:
                        logger.warning("[HttpHarnessExecutorV4] Failed recording metric for '%s': %s", edge.id, err)

                    # Stage completed record
                    self.store.stage_output(
                        session_id=session_id,
                        edge_id=edge.id,
                        key="edge_completed",
                        value=tool_content,
                        vertex_name=edge.output_vertex,
                        metadata={"tool_call_id": tool_call_id, "success": res.success},
                    )

                    preview = (tool_content[:200] + "...") if len(tool_content) > 200 else tool_content
                    edge_reports.append(
                        f"[Edge Completed: {edge.id} ({edge.type})]\n"
                        f"• Input: '{edge.input_vertex}' -> Output: '{edge.output_vertex}'\n"
                        f"• Tool Output: {preview}"
                    )

        # -------------------------------------------------------------------
        # Phase 2: Ingest Initial User Input into START Vertices
        # -------------------------------------------------------------------
        elif latest_msg.get("role") in ("user", "system"):
            user_text = str(latest_msg.get("content", ""))
            if user_text:
                for v in graph.vertices.values():
                    if v.has_attribute(VertexAttributeV4.START):
                        rec = self.store.get_vertex(session_id, v.name)
                        if rec and rec.state != VertexStateV4.DATA_READY.value:
                            self.store.update_vertex_content(
                                session_id=session_id,
                                name=v.name,
                                content=user_text,
                                state=VertexStateV4.DATA_READY.value,
                            )

        # -------------------------------------------------------------------
        # Phase 3: Loop through local edges until a Tool Call or Terminal is reached
        # -------------------------------------------------------------------
        while True:
            # Check terminal state
            is_done, is_success = self._is_terminal(session_id, graph)
            if is_done:
                final_content = self._get_end_vertex_content(session_id, graph)
                full_report = "\n\n".join(edge_reports)
                full_content = f"{full_report}\n\n{final_content}" if full_report else final_content

                return {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content,
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": self._get_usage(session_id),
                    "system_fingerprint": f"vea-v4-{session_id}",
                }

            # Select next eligible edge
            eligible = self._get_eligible_edges(session_id, graph)
            if not eligible:
                # No edge can run right now
                final_content = self._get_end_vertex_content(session_id, graph)
                full_report = "\n\n".join(edge_reports)
                return {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_report or final_content or "Workflow completed.",
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": self._get_usage(session_id),
                    "system_fingerprint": f"vea-v4-{session_id}",
                }

            _rank, _prio, next_edge = eligible[0]

            # Case A: Declarative Tool Edge -> Yield Tool Call to Harness
            if isinstance(next_edge, ToolEdgeV4):
                in_v = self.store.get_vertex(session_id, next_edge.input_vertex)
                in_content = in_v.content if in_v else ""
                call_id = f"call_{uuid.uuid4().hex[:12]}"
                tool_call = next_edge.build_tool_call(in_v_content=in_content, call_id=call_id)

                # Stage pending call so the next request can resolve it
                stage_payload = {
                    "call_id": call_id,
                    "edge_id": next_edge.id,
                    "tool_name": next_edge.tool_name,
                    "input_vertex": next_edge.input_vertex,
                    "output_vertex": next_edge.output_vertex,
                    "start_time": time.perf_counter(),
                }
                self.store.stage_output(
                    session_id=session_id,
                    edge_id=next_edge.id,
                    key="pending_tool_call",
                    value=json.dumps(stage_payload),
                    vertex_name=next_edge.output_vertex,
                )

                edge_intro = f"[Edge: {next_edge.id}] Requesting tool '{next_edge.tool_name}' execution in harness sandbox..."
                full_report = "\n\n".join(edge_reports)
                combined_content = f"{full_report}\n\n{edge_intro}" if full_report else edge_intro

                return {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": combined_content,
                            "tool_calls": [tool_call],
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": self._get_usage(session_id),
                    "system_fingerprint": f"vea-v4-{session_id}",
                }

            # Case B: LLM Tool Edge -> If LLM emits tool_calls, yield to Harness
            elif isinstance(next_edge, LLMToolEdgeV4):
                in_v = self.store.get_vertex(session_id, next_edge.input_vertex)
                rendered = (next_edge.prompt_template or "{input}").replace("{input}", in_v.content if in_v else "")

                agent = getattr(next_edge, "agent", None) or self.agent
                if agent and hasattr(agent, "chat_with_tools"):
                    llm_resp = await agent.chat_with_tools(rendered, tools=next_edge.tools)
                    tc = next_edge.build_tool_call_from_llm_response(llm_resp)
                    if tc:
                        stage_payload = {
                            "call_id": tc.get("id"),
                            "edge_id": next_edge.id,
                            "tool_name": tc.get("function", {}).get("name"),
                            "input_vertex": next_edge.input_vertex,
                            "output_vertex": next_edge.output_vertex,
                            "start_time": time.perf_counter(),
                        }
                        self.store.stage_output(
                            session_id=session_id,
                            edge_id=next_edge.id,
                            key="pending_tool_call",
                            value=json.dumps(stage_payload),
                            vertex_name=next_edge.output_vertex,
                        )
                        full_report = "\n\n".join(edge_reports)
                        return {
                            "id": request_id,
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": full_report or None,
                                    "tool_calls": [tc],
                                },
                                "finish_reason": "tool_calls",
                            }],
                            "usage": self._get_usage(session_id),
                            "system_fingerprint": f"vea-v4-{session_id}",
                        }

                # Fallback: run edge standard compute
                start_t = time.perf_counter()
                res = await next_edge.run(session_id=session_id, store=self.store, agent=agent)
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0

                try:
                    p_tok = int(res.metadata.get("prompt_tokens") or res.metadata.get("usage", {}).get("prompt_tokens", 0))
                    c_tok = int(res.metadata.get("completion_tokens") or res.metadata.get("usage", {}).get("completion_tokens", 0))
                    t_tok = int(res.metadata.get("total_tokens") or res.metadata.get("usage", {}).get("total_tokens", p_tok + c_tok))
                    cost = float(res.metadata.get("cost_usd") or res.metadata.get("cost", 0.0))
                    self.store.record_edge_metric(
                        session_id=session_id,
                        edge_id=next_edge.id,
                        edge_type=next_edge.type,
                        input_vertex=next_edge.input_vertex,
                        output_vertex=next_edge.output_vertex,
                        execution_time_ms=elapsed_ms,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        total_tokens=t_tok,
                        cost_usd=cost,
                        success=res.success,
                        error=res.error,
                        metadata=res.metadata,
                    )
                except Exception as err:
                    logger.warning("[HttpHarnessExecutorV4] Failed recording metric for '%s': %s", next_edge.id, err)

                edge_reports.append(
                    f"[Edge Completed: {next_edge.id} ({next_edge.type})]\n"
                    f"• Output: {str(res.output)[:200]}"
                )
                continue

            # Case C: Local server edge (CodeEdgeV4, reflexive, etc.)
            else:
                start_t = time.perf_counter()
                agent = getattr(next_edge, "agent", None) or self.agent
                res = await next_edge.run(session_id=session_id, store=self.store, agent=agent)
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0

                try:
                    p_tok = int(res.metadata.get("prompt_tokens") or res.metadata.get("usage", {}).get("prompt_tokens", 0))
                    c_tok = int(res.metadata.get("completion_tokens") or res.metadata.get("usage", {}).get("completion_tokens", 0))
                    t_tok = int(res.metadata.get("total_tokens") or res.metadata.get("usage", {}).get("total_tokens", p_tok + c_tok))
                    cost = float(res.metadata.get("cost_usd") or res.metadata.get("cost", 0.0))
                    self.store.record_edge_metric(
                        session_id=session_id,
                        edge_id=next_edge.id,
                        edge_type=next_edge.type,
                        input_vertex=next_edge.input_vertex,
                        output_vertex=next_edge.output_vertex,
                        execution_time_ms=elapsed_ms,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        total_tokens=t_tok,
                        cost_usd=cost,
                        success=res.success,
                        error=res.error,
                        metadata=res.metadata,
                    )
                except Exception as err:
                    logger.warning("[HttpHarnessExecutorV4] Failed recording metric for '%s': %s", next_edge.id, err)

                output_prev = (str(res.output)[:200] + "...") if len(str(res.output)) > 200 else str(res.output)
                edge_reports.append(
                    f"[Edge Completed: {next_edge.id} ({next_edge.type})]\n"
                    f"• Input: '{next_edge.input_vertex}' -> Output: '{next_edge.output_vertex}'\n"
                    f"• Result: {output_prev}"
                )
                # Continue loop to see if the next edge needs to yield a tool call!
                continue
