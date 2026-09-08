"""Discrete graph loader from filesystem manifests, directories, and SQLite store."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.edges.base import EdgeV4
from framework.edges.code import CodeEdgeV4
from framework.edges.reflexive import ReflexiveEdgeV4
from framework.vertex_v4 import VertexStateV4, VertexStoreV4

if TYPE_CHECKING:
    from framework.graphs.core import GraphV4

logger = logging.getLogger("vertex_edge_agent.graphs.loader")


class DiscreteGraphLoaderV4:
    """Loads discrete vertex and edge JSON components specified by a master manifest."""

    @classmethod
    def load_from_dict(
        cls,
        data: Dict[str, Any],
        base_dir: Optional[Union[str, Path]] = None,
        override_session_id: Optional[str] = None,
    ) -> "GraphV4":
        """Construct GraphV4 from in-memory dictionary definition."""
        from framework.graphs.core import GraphV4

        session_id = override_session_id or data.get("session_id", "default_session")
        name = (data.get("metadata") or {}).get("name", "subgraph")
        meta = dict(data.get("metadata") or {})
        b_dir = Path(base_dir).resolve() if base_dir else None
        if b_dir:
            meta.setdefault("base_dir", str(b_dir))
        graph = GraphV4(session_id=session_id, name=name, metadata=meta)

        def _resolve_item_path(item: Union[str, Path]) -> Path:
            p = Path(item)
            if b_dir and (b_dir / p).exists():
                return (b_dir / p).resolve()
            if p.is_absolute() and p.exists():
                return p.resolve()
            if p.exists():
                return p.resolve()
            if (Path(_REPO_ROOT) / p).exists():
                return (Path(_REPO_ROOT) / p).resolve()
            target = (b_dir / p) if b_dir else p
            raise FileNotFoundError(f"Configuration path not found: {target}")

        # Load discrete vertices
        for v_item in data.get("vertices", []):
            if isinstance(v_item, (str, Path)):
                v_path = _resolve_item_path(v_item)
                if v_path.is_dir():
                    for vf in sorted(v_path.glob("*.json")):
                        graph.add_vertex(vf)
                else:
                    graph.add_vertex(v_path)
            elif isinstance(v_item, dict):
                graph.add_vertex(v_item, source="manifest_dict")

        # Load discrete edges
        for e_item in data.get("edges", []):
            if isinstance(e_item, (str, Path)):
                e_path = _resolve_item_path(e_item)
                if e_path.is_dir():
                    for ef in sorted(e_path.glob("*.json")):
                        graph.add_edge(ef)
                else:
                    graph.add_edge(e_path)
            elif isinstance(e_item, dict):
                edge_instance = EdgeV4.from_config(e_item, base_dir=str(b_dir) if b_dir else None)
                graph.add_edge(edge_instance)

        graph.validate(strict_dag=False)
        return graph

    @classmethod
    def load_from_directory(
        cls,
        directory_path: Union[str, Path],
        override_session_id: Optional[str] = None,
    ) -> "GraphV4":
        """Construct GraphV4 by loading all vertex and edge configurations in a directory."""
        from framework.graphs.core import GraphV4

        d_p = Path(directory_path)
        if not d_p.is_absolute() and not d_p.exists():
            cand = Path(_REPO_ROOT) / d_p
            if cand.exists():
                d_p = cand
        if not d_p.is_dir():
            raise FileNotFoundError(f"Graph directory not found: {directory_path}")
        d_p = d_p.resolve()

        # Check for standard manifest file in directory
        for m_name in ("graph.json", "manifest.json"):
            cand_m = d_p / m_name
            if cand_m.is_file():
                return cls.load_from_manifest(cand_m, override_session_id=override_session_id)

        session_id = override_session_id or d_p.name
        graph = GraphV4(session_id=session_id, name=d_p.name, metadata={"base_dir": str(d_p)})

        # Check for vertices subdir
        for v_sub in ("vertices", "vertex", "nodes"):
            sub_dir = d_p / v_sub
            if sub_dir.is_dir():
                for vf in sorted(sub_dir.glob("*.json")):
                    graph.add_vertex(vf)
                break

        # Check for edges subdir
        for e_sub in ("edges", "edge"):
            sub_dir = d_p / e_sub
            if sub_dir.is_dir():
                for ef in sorted(sub_dir.glob("*.json")):
                    graph.add_edge(ef)
                break

        # Top-level loose configs
        for jf in sorted(d_p.glob("*.json")):
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                if (
                    any(k in item for k in ("input_vertex", "output_vertex", "edge_id", "source", "destination"))
                    or item.get("type") in ("code", "reflexive", "llm", "llm_chat", "llm_generate", "llm_process", "llm_callable")
                ):
                    edge_instance = EdgeV4.from_config(item, base_dir=str(d_p))
                    graph.add_edge(edge_instance)
                elif "name" in item:
                    graph.add_vertex(item, source=f"file:{jf}")

        graph.validate(strict_dag=False)
        return graph

    @classmethod
    def load_from_manifest(
        cls,
        manifest_path: Union[str, Path],
        override_session_id: Optional[str] = None,
    ) -> "GraphV4":
        """Parse master manifest JSON and all discrete vertex/edge JSON files."""
        path = Path(manifest_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Master manifest not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.load_from_dict(
            data=data,
            base_dir=path.parent,
            override_session_id=override_session_id or data.get("session_id", path.stem),
        )

    @classmethod
    def populate_store(cls, graph: "GraphV4", store: VertexStoreV4) -> None:
        """Seed all graph vertices and edges into SQLite store."""
        for v in graph.vertices.values():
            db_record = store.save_vertex(
                session_id=graph.session_id,
                name=v.name,
                content=v.content,
                attributes=v.attributes,
                state=v.state,
                processed_count=v.processed_count,
            )
            v.id = db_record.id

        for e in graph.edges.values():
            script_val = getattr(e, "script", None)
            script_str = script_val if isinstance(script_val, str) else None
            trigger_state = getattr(e, "trigger_state", None)
            target_state = getattr(e, "target_state", None)
            max_retries = getattr(e, "max_retries", 3)
            store.save_edge(
                session_id=graph.session_id,
                edge_id=e.id,
                edge_type=e.type,
                input_vertex=e.input_vertex,
                output_vertex=e.output_vertex,
                script=script_str,
                trigger_state=trigger_state,
                target_state=target_state,
                max_retries=max_retries,
                settings=e.settings,
            )


def load_from_store_fn(cls: Any, store: VertexStoreV4, session_id: str, name: str = "v4_graph") -> "GraphV4":
    """Hydrate a GraphV4 instance from SQLite persisted vertices and edges."""
    from framework.chat_llm_edge_v4 import ChatLLMEdgeV4
    from framework.generate_llm_edge_v4 import GenerateLLMEdgeV4
    from framework.process_llm_edge_v4 import ProcessLLMEdgeV4
    from framework.callable_llm_edge_v4 import CallableLLMEdgeV4
    from framework.edges.tool import ToolEdgeV4, LLMToolEdgeV4
    from framework.edges.llm import LLMEdgeV4

    graph = cls(session_id=session_id, name=name)
    for v in store.list_vertices(session_id):
        graph.add_vertex(v, source="sqlite_store")

    for er in store.list_edges(session_id):
        edge_type = er.edge_type
        if edge_type == "reflexive" or er.input_vertex == er.output_vertex:
            edge: EdgeV4 = ReflexiveEdgeV4(
                edge_id=er.edge_id,
                vertex_name=er.input_vertex,
                trigger_state=er.trigger_state or VertexStateV4.REJECT.value,
                target_state=er.target_state or VertexStateV4.TODO_URGENT.value,
                max_retries=er.max_retries,
                script=er.script,
                settings=er.settings,
            )
        elif edge_type in ("tool", "tool_call"):
            edge = ToolEdgeV4(
                edge_id=er.edge_id,
                input_vertex=er.input_vertex,
                output_vertex=er.output_vertex,
                tool_name=er.settings.get("tool_name", er.settings.get("tool", "bash")),
                arguments=er.settings.get("arguments", er.settings.get("args", {})),
                arguments_template=er.settings.get("arguments_template"),
                settings=er.settings,
            )
        elif edge_type in ("llm_tool", "llm_tool_call"):
            edge = LLMToolEdgeV4(
                edge_id=er.edge_id,
                input_vertex=er.input_vertex,
                output_vertex=er.output_vertex,
                model=er.settings.get("model", "sensenova-6.8-flash-lite"),
                prompt_template=er.settings.get("prompt"),
                tools=er.settings.get("tools", []),
                settings=er.settings,
            )
        elif edge_type in ("llm", "llm_chat", "llm_generate", "llm_process", "llm_callable"):
            cls_map = {
                "llm": LLMEdgeV4,
                "llm_chat": ChatLLMEdgeV4,
                "llm_generate": GenerateLLMEdgeV4,
                "llm_process": ProcessLLMEdgeV4,
                "llm_callable": CallableLLMEdgeV4,
            }
            target_cls = cls_map.get(edge_type, LLMEdgeV4)
            edge = target_cls(
                edge_id=er.edge_id,
                input_vertex=er.input_vertex,
                output_vertex=er.output_vertex,
                model=er.settings.get("model", "sensenova-6.8-flash-lite"),
                prompt_template=er.settings.get("prompt"),
                settings=er.settings,
                agent_mode=er.settings.get("agent_mode"),
            )
        else:
            edge = CodeEdgeV4(
                edge_id=er.edge_id,
                input_vertex=er.input_vertex,
                output_vertex=er.output_vertex,
                script=er.script,
                settings=er.settings,
            )
        graph.add_edge(edge)

    try:
        graph.compute_dag_tiers()
    except Exception as e:
        logger.warning("Failed to compute DAG tiers during load_from_store: %s", e)
    return graph
