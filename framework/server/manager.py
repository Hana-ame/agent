"""Session graph manager for isolated, SQLite-synchronized GraphV4 execution."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from framework.edge_v4 import EdgeV4, ReflexiveEdgeV4
from framework.executor_v4 import ExecutorV4, GraphEventV4
from framework.graph_v4 import DiscreteGraphLoaderV4, GraphTopologyError, GraphV4
from framework.snapshot_v4 import GraphSnapshotManagerV4
from framework.vertex_v4 import VertexRecordV4, VertexStateV4, VertexStoreV4

logger = logging.getLogger("vertex_edge_agent.server_v4.manager")

class SessionGraphManagerV4:
    """Maintains isolated GraphV4 instances per session synchronized with VertexStoreV4."""

    def __init__(self, store: VertexStoreV4, snapshot_dir: Optional[Union[str, Path]] = "snapshots"):
        self.store = store
        self._graphs: Dict[str, GraphV4] = {}
        self._event_broadcasters: Dict[str, List[asyncio.Queue]] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        # P2: Track running executors per session for task cancellation on reentry
        self._running_executors: Dict[str, "ExecutorV4"] = {}
        self.snapshot_dir = snapshot_dir
        self.snapshot_manager: Optional[GraphSnapshotManagerV4] = (
            GraphSnapshotManagerV4(base_dir=snapshot_dir) if snapshot_dir else None
        )

    def record_snapshot(self, session_id: str, trigger: str) -> Optional[Path]:
        """Record a complete graph snapshot to a local JSON file for this session."""
        if not self.snapshot_manager:
            return None
        try:
            graph = self.load_graph_from_store(session_id)
            return self.snapshot_manager.save_snapshot(
                graph=graph,
                trigger=trigger,
                store=self.store,
            )
        except Exception as e:
            logger.warning("Failed recording snapshot for session '%s' (%s): %s", session_id, trigger, e)
            return None

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
                    if old_edge:
                        if callable(getattr(old_edge, "script", None)):
                            edge.script = old_edge.script
                        if hasattr(old_edge, "arguments") and old_edge.arguments and not getattr(edge, "arguments", None):
                            edge.arguments = old_edge.arguments
                        if hasattr(old_edge, "tool_name") and old_edge.tool_name:
                            edge.tool_name = old_edge.tool_name
                        if hasattr(old_edge, "arguments_template") and old_edge.arguments_template:
                            edge.arguments_template = old_edge.arguments_template
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
        self.record_snapshot(session_id, trigger=f"vertex_saved:{name}")
        return db_record

    def delete_vertex(self, session_id: str, name: str) -> bool:
        """Delete vertex from database, cleaning up connected edges in SQLite."""
        for er in self.store.list_edges(session_id):
            if er.input_vertex == name or er.output_vertex == name:
                self.store.delete_edge(session_id, er.edge_id)
        deleted = self.store.delete_vertex(session_id, name)
        if deleted:
            self.record_snapshot(session_id, trigger=f"vertex_deleted:{name}")
        return deleted

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
        self.record_snapshot(session_id, trigger=f"edge_saved:{edge_id}")
        return edge

    def delete_edge(self, session_id: str, edge_id: str) -> bool:
        """Delete edge from SQLite database."""
        deleted = self.store.delete_edge(session_id, edge_id)
        if deleted:
            self.record_snapshot(session_id, trigger=f"edge_deleted:{edge_id}")
        return deleted

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
        self.record_snapshot(session_id, trigger=f"edge_reconnected:{edge_id}")
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


