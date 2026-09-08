"""V4 Complete Graph Snapshot Manager.

Persists the complete state of a GraphV4 (including all vertices, edges,
metadata, relationships, tiers, and snapshot metadata) as standalone JSON files
in a structured directory hierarchy:

    snapshots/{session_id}/step_{step:04d}_{trigger}.json

Each snapshot is a 100% valid, full graph specification that can be loaded
directly via DiscreteGraphLoaderV4 or used for time-travel rollback/restore.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from framework.graph_v4 import DiscreteGraphLoaderV4, GraphV4
from framework.vertex_v4 import VertexStoreV4

logger = logging.getLogger(__name__)


def _sanitize_trigger(trigger: str) -> str:
    """Sanitize trigger string for use in filename."""
    s = re.sub(r"[^a-zA-Z0-9_\-]", "_", trigger.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:40] if s else "snapshot"


class GraphSnapshotManagerV4:
    """Manages complete GraphV4 historical snapshots saved as local JSON files."""

    def __init__(self, base_dir: Union[str, Path] = "snapshots") -> None:
        self.base_dir = Path(base_dir).resolve()

    def get_session_dir(self, session_id: str) -> Path:
        """Return and ensure directory for a session's snapshots."""
        s_dir = self.base_dir / session_id
        s_dir.mkdir(parents=True, exist_ok=True)
        return s_dir

    def get_next_step(self, session_id: str) -> int:
        """Determine next sequential step number for a session."""
        s_dir = self.base_dir / session_id
        if not s_dir.exists():
            return 0
        existing_steps: List[int] = []
        for p in s_dir.glob("step_*.json"):
            m = re.match(r"^step_(\d+)", p.name)
            if m:
                try:
                    existing_steps.append(int(m.group(1)))
                except ValueError:
                    pass
        return (max(existing_steps) + 1) if existing_steps else 0

    def save_snapshot(
        self,
        graph: GraphV4,
        trigger: str = "manual",
        store: Optional[VertexStoreV4] = None,
        metadata: Optional[Dict[str, Any]] = None,
        custom_step: Optional[int] = None,
        indent: int = 2,
    ) -> Path:
        """Serialize and persist the complete graph as a standalone JSON snapshot.

        Args:
            graph: The GraphV4 instance.
            trigger: Identifier for the event triggering this snapshot (e.g.
                     'execution_start', 'edge_completed:e1', 'vertex_added:v1').
            store: Optional VertexStoreV4. If provided, latest live vertex contents
                   and states are reloaded from SQLite to ensure 100% fidelity.
            metadata: Optional additional key-values to record in snapshot_metadata.
            custom_step: Optional explicit step index override.
            indent: JSON indentation.

        Returns:
            Path to the saved snapshot file.
        """
        session_id = graph.session_id
        s_dir = self.get_session_dir(session_id)

        step = custom_step if custom_step is not None else self.get_next_step(session_id)
        clean_trig = _sanitize_trigger(trigger)
        filename = f"step_{step:04d}_{clean_trig}.json"
        target_path = s_dir / filename

        # If store is provided, rehydrate from SQLite to capture latest vertex state
        if store is not None:
            try:
                target_graph = GraphV4.load_from_store(store, session_id, name=graph.name)
            except Exception as e:
                logger.warning("Could not rehydrate graph from store for snapshot: %s", e)
                target_graph = graph
        else:
            target_graph = graph

        # Complete graph dump
        graph_data = target_graph.dump()

        now_iso = datetime.now(timezone.utc).isoformat()
        snapshot_meta: Dict[str, Any] = {
            "session_id": session_id,
            "step": step,
            "trigger": trigger,
            "timestamp": now_iso,
            "filename": filename,
        }
        if metadata:
            snapshot_meta.update(metadata)

        # Embed snapshot metadata directly into the complete graph document
        graph_data["snapshot_metadata"] = snapshot_meta
        graph_data["snapshot_step"] = step
        graph_data["snapshot_trigger"] = trigger
        graph_data["snapshot_timestamp"] = now_iso

        # Atomic file write
        temp_path = target_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=indent, ensure_ascii=False)
        temp_path.replace(target_path)

        logger.debug("[GraphSnapshotManagerV4] Saved snapshot: %s (trigger: %s)", target_path, trigger)
        return target_path

    def list_snapshots(self, session_id: str) -> List[Dict[str, Any]]:
        """List all historical complete graph snapshots for a session, sorted by step."""
        s_dir = self.base_dir / session_id
        if not s_dir.exists():
            return []

        snapshots: List[Dict[str, Any]] = []
        for p in sorted(s_dir.glob("step_*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("snapshot_metadata", {})
                step = meta.get("step")
                if step is None:
                    m = re.match(r"^step_(\d+)", p.name)
                    step = int(m.group(1)) if m else 0

                snapshots.append({
                    "step": step,
                    "trigger": meta.get("trigger", data.get("snapshot_trigger", "unknown")),
                    "timestamp": meta.get("timestamp", data.get("snapshot_timestamp", "")),
                    "filename": p.name,
                    "path": str(p.resolve()),
                    "vertex_count": len(data.get("vertices", [])),
                    "edge_count": len(data.get("edges", [])),
                })
            except Exception as err:
                logger.warning("Failed reading snapshot file '%s': %s", p, err)

        snapshots.sort(key=lambda s: s["step"])
        return snapshots

    def get_snapshot_data(self, session_id: str, step: int) -> Optional[Dict[str, Any]]:
        """Retrieve complete raw graph snapshot dictionary for a specific step."""
        s_dir = self.base_dir / session_id
        if not s_dir.exists():
            return None

        prefix = f"step_{step:04d}"
        for p in s_dir.glob(f"{prefix}*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as err:
                logger.warning("Error reading snapshot %s: %s", p, err)
                return None

        # Fallback to non-padded pattern
        for p in s_dir.glob(f"step_{step}_*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as err:
                logger.warning("Error reading snapshot %s: %s", p, err)
                return None

        return None

    def load_snapshot_as_graph(self, session_id: str, step: int) -> Optional[GraphV4]:
        """Load a historical snapshot back into an active GraphV4 instance."""
        data = self.get_snapshot_data(session_id, step)
        if data is None:
            return None
        return DiscreteGraphLoaderV4.load_from_dict(data, override_session_id=session_id)

    def restore_snapshot(
        self,
        session_id: str,
        step: int,
        store: VertexStoreV4,
    ) -> Optional[GraphV4]:
        """Roll back and restore the live SQLite store and memory graph to a historical snapshot.

        1. Loads complete graph specification from step_{step}.json.
        2. Clears current session in store.
        3. Repopulates store with all vertices, states, and edges from the historical snapshot.

        Returns:
            The restored GraphV4 instance.
        """
        graph = self.load_snapshot_as_graph(session_id, step)
        if graph is None:
            return None

        # Clear current live state in SQLite
        store.clear_session(session_id)

        # Repopulate from historical complete graph
        DiscreteGraphLoaderV4.populate_store(graph, store)

        logger.info(
            "[GraphSnapshotManagerV4] Restored session '%s' to step %d (%s vertices, %s edges)",
            session_id,
            step,
            len(graph.vertices),
            len(graph.edges),
        )
        return graph
