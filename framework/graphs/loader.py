"""Discrete graph loader from filesystem manifests, directories, and SQLite store."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

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


def _manifest_session_id(data: Dict[str, Any]) -> Optional[str]:
    """Session declared by a manifest.

    The canonical key is ``"session"`` — the same one the CLI uses (``--session``)
    and the same one the standalone edge runner accepts. ``"session_id"`` is kept
    as a deprecated alias because existing manifests already use it; ``session_id``
    is still the internal field name throughout the storage layer
    (``VertexStoreV4``, ``EdgeRecordV4``), which this does not change.
    """
    return data.get("session") or data.get("session_id")


class DiscreteGraphLoaderV4:
    """Loads discrete vertex and edge JSON components specified by a master manifest."""

    @classmethod
    def load_from_dict(
        cls,
        data: Dict[str, Any],
        base_dir: Optional[Union[str, Path]] = None,
        override_session_id: Optional[str] = None,
        script_roots: Optional[Sequence[Union[str, Path]]] = None,
    ) -> "GraphV4":
        """Construct GraphV4 from in-memory dictionary definition.

        ``script_roots`` restricts where a dynamic edge spec (``"my_edge.py:MyEdge"``)
        may load from and is searched when the spec is relative. Omit for the
        defaults: repository root plus ``VEA_SCRIPT_ROOTS``.
        """
        from framework.graphs.core import GraphV4

        session_id = override_session_id or _manifest_session_id(data) or "default_session"
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
                edge_instance = EdgeV4.from_config(
                    e_item, base_dir=str(b_dir) if b_dir else None, script_roots=script_roots
                )
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
                from framework.edges.registry import is_registered_edge_type

                if (
                    any(k in item for k in ("input_vertex", "output_vertex", "edge_id", "source", "destination"))
                    or is_registered_edge_type(item.get("type"))
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
        script_roots: Optional[Sequence[Union[str, Path]]] = None,
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
            override_session_id=override_session_id or _manifest_session_id(data) or path.stem,
            script_roots=script_roots,
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


def load_from_store_fn(
    cls: Any,
    store: VertexStoreV4,
    session_id: str,
    name: str = "v4_graph",
    script_roots: Optional[Sequence[Union[str, Path]]] = None,
) -> "GraphV4":
    """Hydrate a GraphV4 instance from SQLite persisted vertices and edges.

    Edge construction goes through the same registry-driven
    :meth:`EdgeV4.from_config` path as manifest loading, so every registered edge
    type round-trips identically (including ``llm_tool`` and its tool catalog).
    """
    from framework.edges.base import EdgeV4

    graph = cls(session_id=session_id, name=name)
    # Remember where this graph's edge scripts may load from, so a later
    # rehydrate (ExecutorV4's store sync) resolves dynamic edges identically.
    graph.script_roots = [Path(p).resolve() for p in script_roots] if script_roots else None
    for v in store.list_vertices(session_id):
        graph.add_vertex(v, source="sqlite_store")

    for er in store.list_edges(session_id):
        settings = dict(er.settings or {})
        cfg: Dict[str, Any] = {
            # Keep the type the user declared ("my_edge.py:MyEdge").
            # EdgeV4.from_config prefers the absolute "_edge_class_spec" recorded
            # in settings for *loading*, so a relative spec still resolves from a
            # different cwd without the persisted type being rewritten.
            "id": er.edge_id,
            "type": er.edge_type,
            "input_vertex": er.input_vertex,
            "output_vertex": er.output_vertex,
            "script": er.script,
            "trigger_state": er.trigger_state,
            "target_state": er.target_state,
            "max_retries": er.max_retries,
            "settings": settings,
        }
        try:
            edge = EdgeV4.from_config(cfg, script_roots=script_roots)
        except Exception as exc:
            logger.warning(
                "Skipping edge '%s' (type '%s'): %s", er.edge_id, er.edge_type, exc
            )
            continue
        graph.add_edge(edge)

    try:
        graph.compute_dag_tiers()
    except Exception as e:
        logger.warning("Failed to compute DAG tiers during load_from_store: %s", e)
    return graph
