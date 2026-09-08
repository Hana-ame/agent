"""V4 SSE Executor with Session Routing, Subgraph Vertices, and Harness Echo Integration.

Features:
1. Dynamic Session Routing: Resolves session_id, automatically instantiating a new session graph if absent.
2. Subgraph Vertex Hierarchy: Vertices with attribute 'subgraph' load nested child graphs tracked under the session.
3. DAG-Ordered Execution: Coordinates edges according to topological levels.
4. Harness Echo Compatibility: Emits SSE streams and outputs structured as tool call echo 'info' payloads.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, Union

from framework.edge_v4 import CodeEdgeV4
from framework.executor_v4 import ExecutorV4, GraphEventV4
from framework.graph_v4 import DiscreteGraphLoaderV4, GraphV4
from framework.server_v4 import SessionGraphManagerV4
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.sse_executor_v4")


@dataclass
class ToolCallEcho:
    """Tool call payload formatted for agent harness frameworks."""

    call_id: str
    function_name: str = "echo"
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_openai_dict(self) -> Dict[str, Any]:
        """Format as standard OpenAI tool_call."""
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.function_name,
                "arguments": json.dumps(self.arguments),
            },
        }

    def to_sse_chunk(self, chunk_id: str, finish_reason: Optional[str] = None) -> str:
        """Format as OpenAI streaming chunk SSE data line."""
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "vea-v4-sse-executor",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": self.call_id,
                                "type": "function",
                                "function": {
                                    "name": self.function_name,
                                    "arguments": json.dumps(self.arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"


class SSEExecutorV4:
    """Session-aware SSE executor with subgraph resolution and harness echo streaming."""

    def __init__(
        self,
        manager: SessionGraphManagerV4,
        store: Optional[VertexStoreV4] = None,
        default_manifest: Optional[Union[str, Path]] = None,
    ):
        self.manager = manager
        self.store = store or manager.store
        self.default_manifest = default_manifest

    def resolve_session_and_graph(
        self,
        session_id: Optional[str] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[str, GraphV4]:
        """Retrieve existing session graph or dynamically construct a new one."""
        if not session_id:
            session_id = f"sess_{uuid.uuid4().hex[:10]}"

        manifest_to_use = manifest_path or self.default_manifest

        # Load from manifest if provided
        if manifest_to_use and Path(manifest_to_use).exists():
            graph = DiscreteGraphLoaderV4.load_from_manifest(
                manifest_path=manifest_to_use,
                override_session_id=session_id,
            )
            DiscreteGraphLoaderV4.populate_store(graph, self.store)
            return session_id, self.manager.load_graph_from_store(session_id)

        # Otherwise read and update fresh from store
        graph = self.manager.get_or_create_graph(session_id)
        return session_id, graph

    def resolve_subgraph_vertices(self, session_id: str, graph: GraphV4) -> None:
        """Discover vertices with attribute 'subgraph' and link their nested child graphs."""
        for v in list(graph.vertices.values()):
            if v.has_attribute(VertexAttributeV4.SUBGRAPH):
                sub_session_id = f"{session_id}::{v.name}"
                sub_manifest_path = None
                sub_input_map = {}
                sub_output_map = {}

                if v.content:
                    try:
                        content_dict = json.loads(v.content)
                        sub_manifest_path = content_dict.get("subgraph_manifest")
                        sub_input_map = content_dict.get("input_map", {})
                        sub_output_map = content_dict.get("output_map", {})
                    except Exception:
                        sub_manifest_path = v.content.strip()

                if sub_manifest_path and Path(sub_manifest_path).exists():
                    sub_graph = DiscreteGraphLoaderV4.load_from_manifest(
                        manifest_path=sub_manifest_path,
                        override_session_id=sub_session_id,
                    )
                    DiscreteGraphLoaderV4.populate_store(sub_graph, self.store)
                    self.manager._graphs[sub_session_id] = sub_graph

                    # Recursively resolve any deeper nested subgraphs within the child graph
                    self.resolve_subgraph_vertices(sub_session_id, sub_graph)

                    # Attach bridge executor edge for the subgraph
                    bridge_edge_id = f"bridge_subgraph_{v.name}"
                    if bridge_edge_id not in graph.edges:
                        # Capture variables for closure
                        _sub_graph = sub_graph
                        _sub_session_id = sub_session_id
                        _store = self.store
                        _input_map = dict(sub_input_map)
                        _output_map = dict(sub_output_map)
                        _parent_vertex_name = v.name

                        async def run_subgraph_bridge(
                            content: str,
                            settings: dict,
                            staging: dict,
                            sg=_sub_graph,
                            ssid=_sub_session_id,
                            st=_store,
                            inp_map=_input_map,
                            out_map=_output_map,
                            pv_name=_parent_vertex_name,
                            session_id=session_id,
                        ) -> str:
                            """Bridge edge that forwards data through a child subgraph."""
                            # 1. Route input into child subgraph
                            parsed_json = None
                            try:
                                parsed_json = json.loads(content) if content else {}
                            except Exception:
                                parsed_json = None

                            if inp_map and isinstance(parsed_json, dict):
                                for source_key, target_node in inp_map.items():
                                    val_to_inject = parsed_json.get(source_key, content)
                                    st.update_vertex_content(
                                        session_id=ssid,
                                        name=target_node,
                                        content=str(val_to_inject),
                                        state=VertexStateV4.DATA_READY.value,
                                    )
                            else:
                                sub_start_nodes = [
                                    sv for sv in sg.vertices.values()
                                    if sv.has_attribute(VertexAttributeV4.START)
                                ]
                                for sv in sub_start_nodes:
                                    st.update_vertex_content(
                                        session_id=ssid,
                                        name=sv.name,
                                        content=content,
                                        state=VertexStateV4.DATA_READY.value,
                                    )

                            # 2. Execute child subgraph
                            sub_executor = ExecutorV4(
                                graph=sg,
                                store=st,
                                max_concurrency=2,
                            )
                            sub_res = await sub_executor.run()

                            # 3. Collect output from child subgraph
                            final_output = ""
                            if out_map:
                                collected: Dict[str, Any] = {}
                                for target_node, dest_key in out_map.items():
                                    rec = st.get_vertex(ssid, target_node)
                                    collected[dest_key] = rec.content if rec else ""
                                final_output = json.dumps(collected) if len(collected) > 1 else list(collected.values())[0] if collected else ""
                            else:
                                sub_end_nodes = [
                                    sv for sv in sg.vertices.values()
                                    if sv.has_attribute(VertexAttributeV4.END)
                                ]
                                if sub_end_nodes:
                                    end_name = sub_end_nodes[0].name
                                    end_rec = st.get_vertex(ssid, end_name)
                                    final_output = end_rec.content if end_rec else ""
                                elif sub_res.vertex_contents:
                                    final_output = sub_res.vertex_contents.get(
                                        list(sub_res.vertex_contents.keys())[-1], ""
                                    )

                            # 4. Synchronize back to parent vertex in SQLite
                            st.update_vertex_content(
                                session_id=session_id,
                                name=pv_name,
                                content=str(final_output),
                                state=VertexStateV4.DATA_READY.value,
                            )
                            return str(final_output)

                        # Create a proxy output vertex for the bridge edge so it's not reflexive
                        proxy_name = f"{v.name}__bridge_out"
                        if proxy_name not in graph.vertices:
                            proxy_v = VertexRecordV4(
                                id=0,
                                session_id=graph.session_id,
                                name=proxy_name,
                                content="",
                                attributes=[],
                                state=VertexStateV4.TODO.value,
                            )
                            graph.add_vertex(proxy_v)
                            self.store.save_vertex(
                                session_id=graph.session_id,
                                name=proxy_name,
                                content="",
                                state=VertexStateV4.TODO.value,
                            )

                        bridge_edge = CodeEdgeV4(
                            edge_id=bridge_edge_id,
                            input_vertex=v.name,
                            output_vertex=proxy_name,
                            script=run_subgraph_bridge,
                        )
                        graph.add_edge(bridge_edge)
                        self.store.save_edge(
                            session_id=graph.session_id,
                            edge_id=bridge_edge_id,
                            edge_type="code",
                            input_vertex=v.name,
                            output_vertex=proxy_name,
                        )

                        # Rewire downstream edges that originally depended on v.name to proxy_name
                        for edge in list(graph.edges.values()):
                            if edge.id != bridge_edge_id and edge.input_vertex == v.name:
                                edge.input_vertex = proxy_name
                                self.store.save_edge(
                                    session_id=graph.session_id,
                                    edge_id=edge.id,
                                    edge_type=edge.type,
                                    input_vertex=proxy_name,
                                    output_vertex=edge.output_vertex,
                                    script=getattr(edge, "script", None) if isinstance(getattr(edge, "script", None), str) else None,
                                    trigger_state=getattr(edge, "trigger_state", None),
                                    target_state=getattr(edge, "target_state", None),
                                    max_retries=getattr(edge, "max_retries", 3),
                                    settings=edge.settings,
                                )

                        try:
                            graph.compute_dag_tiers()
                        except Exception as e:
                            logger.warning('Exception ignored: %s', e)

                        logger.info("[SSEExecutorV4] Registered subgraph '%s' bridge for vertex '%s'", _sub_session_id, v.name)

    def _inject_input_payload(
        self,
        session_id: str,
        graph: GraphV4,
        input_payload: Optional[Union[str, Dict[str, Any]]],
    ) -> None:
        """Inject input payload into start vertices."""
        if input_payload is None:
            return
        raw_input = (
            json.dumps(input_payload)
            if isinstance(input_payload, dict)
            else str(input_payload)
        )
        start_vertices = [
            v for v in graph.vertices.values()
            if v.has_attribute(VertexAttributeV4.START)
        ]
        for sv in start_vertices:
            self.store.update_vertex_content(
                session_id=session_id,
                name=sv.name,
                content=raw_input,
                state=VertexStateV4.DATA_READY.value,
            )

    async def execute_and_stream(
        self,
        session_id: Optional[str] = None,
        input_payload: Optional[Union[str, Dict[str, Any]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        max_concurrency: int = 4,
        timeout: float = 120.0,
    ) -> AsyncGenerator[str, None]:
        """Execute edges in DAG order and yield SSE stream with tool call echo 'info' format."""
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        call_id = f"call_{uuid.uuid4().hex[:10]}"

        try:
            sess_id, graph = self.resolve_session_and_graph(session_id, manifest_path)
            self.resolve_subgraph_vertices(sess_id, graph)
            self._inject_input_payload(sess_id, graph, input_payload)
        except Exception as setup_err:
            error_echo = ToolCallEcho(
                call_id=call_id,
                function_name="echo",
                arguments={"info": {"event": "setup_error", "error": str(setup_err)}},
            )
            yield error_echo.to_sse_chunk(chunk_id, finish_reason="tool_calls")
            yield "data: [DONE]\n\n"
            return

        # Emit initial start event
        start_echo = ToolCallEcho(
            call_id=call_id,
            function_name="echo",
            arguments={
                "info": {
                    "event": "workflow_started",
                    "session_id": sess_id,
                    "concurrency": max_concurrency,
                    "graph_vertices": list(graph.vertices.keys()),
                }
            },
        )
        yield start_echo.to_sse_chunk(chunk_id)

        executor = ExecutorV4(
            graph=graph,
            store=self.store,
            max_concurrency=max_concurrency,
            timeout=timeout,
        )

        # Stream real-time edge execution steps
        stream_error_occurred = False
        try:
            async for event in executor.stream():
                self.manager.broadcast_event(sess_id, event)
                echo_chunk = ToolCallEcho(
                    call_id=call_id,
                    function_name="echo",
                    arguments={
                        "info": {
                            "event": event.event_type,
                            "session_id": sess_id,
                            "edge_id": event.edge_id,
                            "vertex_name": event.vertex_name,
                            "payload": event.payload,
                        }
                    },
                )
                yield echo_chunk.to_sse_chunk(chunk_id)
        except Exception as stream_err:
            stream_error_occurred = True
            error_echo = ToolCallEcho(
                call_id=call_id,
                function_name="echo",
                arguments={"info": {"event": "stream_error", "error": str(stream_err)}},
            )
            yield error_echo.to_sse_chunk(chunk_id, finish_reason="tool_calls")
            yield "data: [DONE]\n\n"

        if not stream_error_occurred:
            # Final result collection
            result = executor.result
            final_info = {
                "event": "workflow_finished",
                "session_id": sess_id,
                "success": result.success,
                "execution_time": result.execution_time,
                "completed_edges": result.completed_edges,
                "vertex_states": result.vertex_states,
                "vertex_contents": result.vertex_contents,
                "errors": result.errors,
            }

            # Terminal tool call echo
            final_echo = ToolCallEcho(
                call_id=call_id,
                function_name="echo",
                arguments={"info": final_info},
            )
            yield final_echo.to_sse_chunk(chunk_id, finish_reason="tool_calls")
            yield "data: [DONE]\n\n"

    async def execute_harness_call(
        self,
        session_id: Optional[str] = None,
        input_payload: Optional[Union[str, Dict[str, Any]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        max_concurrency: int = 4,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Execute workflow and return non-streaming tool call echo 'info' payload."""
        call_id = f"call_{uuid.uuid4().hex[:10]}"
        try:
            sess_id, graph = self.resolve_session_and_graph(session_id, manifest_path)
            self.resolve_subgraph_vertices(sess_id, graph)

            self._inject_input_payload(sess_id, graph, input_payload)

            executor = ExecutorV4(
                graph=graph,
                store=self.store,
                max_concurrency=max_concurrency,
                timeout=timeout,
            )
            result = await executor.run()

            echo = ToolCallEcho(
                call_id=call_id,
                function_name="echo",
                arguments={
                    "info": {
                        "session_id": sess_id,
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "completed_edges": result.completed_edges,
                        "vertex_states": result.vertex_states,
                        "vertex_contents": result.vertex_contents,
                        "errors": result.errors,
                    }
                },
            )
            return echo.to_openai_dict()
        except Exception as exc:
            logger.error("[SSEExecutorV4] execute_harness_call failed: %s", exc)
            error_echo = ToolCallEcho(
                call_id=call_id,
                function_name="echo",
                arguments={
                    "info": {
                        "session_id": session_id,
                        "success": False,
                        "error": str(exc),
                    }
                },
            )
            return error_echo.to_openai_dict()
