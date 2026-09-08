"""V4 Workflow Orchestrator with Session Routing, Subgraph Resolution, and Input Injection.

Encapsulates workflow DAG execution, session state management, and recursive subgraph
resolution as a pure domain service independent of HTTP/SSE transport protocols.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from framework.edge_v4 import CodeEdgeV4
from framework.executor_v4 import ExecutorV4, GraphEventV4
from framework.graph_v4 import DiscreteGraphLoaderV4, GraphV4
from framework.graph_manager_v4 import SessionGraphManagerV4
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.workflow_executor_v4")

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)


@dataclass
class WorkflowEventV4:
    """Domain event emitted during workflow execution."""

    event_type: str
    session_id: str
    edge_id: Optional[str] = None
    vertex_name: Optional[str] = None
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResultV4:
    """Consolidated result returned after workflow completion."""

    session_id: str
    success: bool
    execution_time: float
    completed_edges: List[str] = field(default_factory=list)
    vertex_states: Dict[str, str] = field(default_factory=dict)
    vertex_contents: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "session_id": self.session_id,
            "success": self.success,
            "execution_time": self.execution_time,
            "completed_edges": list(self.completed_edges),
            "vertex_states": dict(self.vertex_states),
            "vertex_contents": dict(self.vertex_contents),
            "errors": dict(self.errors),
        }


class WorkflowExecutorV4:
    """Pure workflow orchestrator handling session routing, subgraphs, and step execution."""

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
        if manifest_to_use:
            m_path = Path(manifest_to_use)
            if not m_path.is_absolute() and not m_path.exists():
                cand = Path(_REPO_ROOT) / m_path
                if cand.exists():
                    manifest_to_use = str(cand)

        # Load from manifest if provided
        if manifest_to_use and Path(manifest_to_use).exists():
            graph = DiscreteGraphLoaderV4.load_from_manifest(
                manifest_path=manifest_to_use,
                override_session_id=session_id,
            )
            self.manager._graphs[session_id] = graph
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
                        sub_manifest_path = (
                            content_dict.get("subgraph_manifest")
                            or content_dict.get("subgraph_dir")
                            or content_dict.get("directory")
                        )
                        sub_input_map = content_dict.get("input_map", {})
                        sub_output_map = content_dict.get("output_map", {})
                    except Exception:
                        sub_manifest_path = v.content.strip()

                if sub_manifest_path:
                    cand = Path(sub_manifest_path)
                    if not cand.is_absolute() and not cand.exists():
                        # 1. Try relative to vertex's loaded source file directory
                        v_info = graph.loaded_nodes.get(v.name) if hasattr(graph, "loaded_nodes") else None
                        if v_info and str(v_info.get("source", "")).startswith("file:"):
                            src_dir = Path(str(v_info["source"])[5:]).resolve().parent
                            if (src_dir / cand).exists():
                                cand = (src_dir / cand).resolve()
                        # 2. Try relative to parent graph's base_dir if present
                        if not cand.exists():
                            p_base = graph.metadata.get("base_dir")
                            if p_base and (Path(p_base) / cand).exists():
                                cand = (Path(p_base) / cand).resolve()
                        # 3. Try relative to repo root
                        if not cand.exists() and (Path(_REPO_ROOT) / cand).exists():
                            cand = (Path(_REPO_ROOT) / cand).resolve()
                    if cand.exists():
                        sub_manifest_path = str(cand)

                if sub_manifest_path and Path(sub_manifest_path).exists():
                    sub_p = Path(sub_manifest_path)
                    if sub_p.is_dir():
                        sub_graph = DiscreteGraphLoaderV4.load_from_directory(
                            directory_path=sub_p,
                            override_session_id=sub_session_id,
                        )
                    else:
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

                            sub_executor = ExecutorV4(
                                graph=sg,
                                store=st,
                                max_concurrency=2,
                            )
                            sub_res = await sub_executor.run()

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

                        logger.info("[WorkflowExecutorV4] Registered subgraph '%s' bridge for vertex '%s'", _sub_session_id, v.name)

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

    async def stream(
        self,
        session_id: Optional[str] = None,
        input_payload: Optional[Union[str, Dict[str, Any]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        max_concurrency: int = 4,
        timeout: float = 120.0,
    ) -> AsyncGenerator[WorkflowEventV4, None]:
        """Execute edges in DAG order and yield pure WorkflowEventV4 events."""
        try:
            sess_id, graph = self.resolve_session_and_graph(session_id, manifest_path)
            self.resolve_subgraph_vertices(sess_id, graph)
            self._inject_input_payload(sess_id, graph, input_payload)
        except Exception as setup_err:
            yield WorkflowEventV4(
                event_type="setup_error",
                session_id=session_id or "",
                payload={"error": str(setup_err)},
            )
            return

        yield WorkflowEventV4(
            event_type="workflow_started",
            session_id=sess_id,
            metadata={
                "concurrency": max_concurrency,
                "graph_vertices": list(graph.vertices.keys()),
            },
        )

        executor = ExecutorV4(
            graph=graph,
            store=self.store,
            max_concurrency=max_concurrency,
            timeout=timeout,
        )

        stream_error_occurred = False
        try:
            async for event in executor.stream():
                self.manager.broadcast_event(sess_id, event)
                yield WorkflowEventV4(
                    event_type=event.event_type,
                    session_id=sess_id,
                    edge_id=event.edge_id,
                    vertex_name=event.vertex_name,
                    payload=event.payload,
                )
        except Exception as stream_err:
            stream_error_occurred = True
            yield WorkflowEventV4(
                event_type="stream_error",
                session_id=sess_id,
                payload={"error": str(stream_err)},
            )

        if not stream_error_occurred:
            result = executor.result
            yield WorkflowEventV4(
                event_type="workflow_finished",
                session_id=sess_id,
                payload={
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "completed_edges": result.completed_edges,
                    "vertex_states": result.vertex_states,
                    "vertex_contents": result.vertex_contents,
                    "errors": result.errors,
                },
            )

    async def run(
        self,
        session_id: Optional[str] = None,
        input_payload: Optional[Union[str, Dict[str, Any]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        max_concurrency: int = 4,
        timeout: float = 120.0,
    ) -> WorkflowResultV4:
        """Execute workflow non-streaming and return clean WorkflowResultV4."""
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
        return WorkflowResultV4(
            session_id=sess_id,
            success=result.success,
            execution_time=result.execution_time,
            completed_edges=result.completed_edges,
            vertex_states=result.vertex_states,
            vertex_contents=result.vertex_contents,
            errors=result.errors,
        )
