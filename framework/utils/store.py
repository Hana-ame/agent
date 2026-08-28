"""State store module — SQLite-backed snapshot persistence.

Provides ``SQLiteStateStore`` for saving and loading ``GraphSnapshot``
objects, enabling resumable execution and audit trails.

Schema
------
runs:
    run_id TEXT  — unique execution identifier
    status TEXT  — 'running' | 'paused' | 'completed' | 'failed' | 'awaiting_approval'
    graph_config TEXT  — JSON of original graph config (for documentation; graph
                         is always reconstructed externally)
    created_at / updated_at TEXT

snapshots:
    run_id, step, trigger, state JSON, created_at
"""

import datetime
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vertex_edge_agent.store")


# ---------------------------------------------------------------------------
# Snapshot data class
# ---------------------------------------------------------------------------

@dataclass
class GraphSnapshot:
    """Immutable point-in-time view of a graph's execution state."""

    run_id: str
    step: int
    trigger: str             # human-readable label, e.g. "vertex:A:done"
    timestamp: str           # ISO-8601 UTC string (filled by store on save)
    vertex_states: Dict[str, Dict]  # vertex_id -> serialised state dict
    edge_states:   Dict[str, Dict]  # edge_id   -> serialised state dict
    graph_config:  Optional[Dict] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Base State Store Interface
# ---------------------------------------------------------------------------

class BaseStateStore(ABC):
    """Abstract interface for execution state and checkpoint stores."""

    @abstractmethod
    def create_run(self, run_id: str, graph_config: Optional[Dict] = None) -> None:
        """Register a new run."""
        pass

    @abstractmethod
    def update_run_status(self, run_id: str, status: str) -> None:
        """Update run status."""
        pass

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get run metadata dict."""
        pass

    @abstractmethod
    def list_runs(self) -> List[Dict]:
        """List all runs."""
        pass

    @abstractmethod
    def save_snapshot(self, snapshot: GraphSnapshot) -> None:
        """Persist an execution snapshot."""
        pass

    @abstractmethod
    def load_latest_snapshot(self, run_id: str) -> Optional[GraphSnapshot]:
        """Retrieve the latest snapshot for a run."""
        pass

    @abstractmethod
    def snapshot_count(self, run_id: str) -> int:
        """Get total snapshot count for a run."""
        pass

    @abstractmethod
    def load_all_snapshots(self, run_id: str) -> List[GraphSnapshot]:
        """Retrieve all snapshots for a run."""
        pass


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------

class SQLiteStateStore(BaseStateStore):
    """Lightweight SQLite-backed store for execution checkpoints.

    Args:
        db_path: Filesystem path for the SQLite database file.
                 Use ``":memory:"`` for an in-memory database (tests).
    """

    def __init__(self, db_path: str = "checkpoints.db"):
        self.db_path = db_path
        self._conn = None
        if db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_db()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id      TEXT PRIMARY KEY,
                    status      TEXT NOT NULL DEFAULT 'running',
                    graph_config TEXT,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id     TEXT NOT NULL,
                    step       INTEGER NOT NULL,
                    trigger    TEXT NOT NULL,
                    state      TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)
            conn.commit()
        logger.debug("[Store] DB initialised at '%s'", self.db_path)

    def _connect(self):
        if self._conn:
            return self._conn
        return sqlite3.connect(self.db_path, check_same_thread=False)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------
    def create_run(
        self,
        run_id: str,
        graph_config: Optional[Dict] = None,
    ) -> None:
        """Register a new run.  Idempotent (IGNORE if already exists)."""
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO runs "
                "(run_id, status, graph_config, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    "running",
                    json.dumps(graph_config) if graph_config else None,
                    now,
                    now,
                ),
            )
            conn.commit()
        logger.info("[Store] Run '%s' registered", run_id)

    def update_run_status(self, run_id: str, status: str) -> None:
        """Update the run's status string."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
                (status, _now(), run_id),
            )
            conn.commit()
        logger.debug("[Store] Run '%s' status → %s", run_id, status)

    def get_run(self, run_id: str) -> Optional[Dict]:
        """Return run metadata dict, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id, status, graph_config, created_at, updated_at "
                "FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "run_id": row[0],
            "status": row[1],
            "graph_config": json.loads(row[2]) if row[2] else None,
            "created_at": row[3],
            "updated_at": row[4],
        }

    def list_runs(self) -> List[Dict]:
        """Return all runs, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, status, created_at, updated_at FROM runs "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [
            {"run_id": r[0], "status": r[1], "created_at": r[2], "updated_at": r[3]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------
    def save_snapshot(self, snapshot: GraphSnapshot) -> None:
        """Persist a snapshot; fills in ``snapshot.timestamp`` with wall-clock."""
        now = _now()
        state_json = json.dumps({
            "vertex_states": snapshot.vertex_states,
            "edge_states":   snapshot.edge_states,
        })
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO snapshots (run_id, step, trigger, state, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (snapshot.run_id, snapshot.step, snapshot.trigger, state_json, now),
            )
            conn.execute(
                "UPDATE runs SET updated_at=? WHERE run_id=?",
                (now, snapshot.run_id),
            )
            conn.commit()
        logger.debug(
            "[Store] Snapshot saved  run=%s  step=%d  trigger=%s",
            snapshot.run_id, snapshot.step, snapshot.trigger,
        )

    def load_latest_snapshot(self, run_id: str) -> Optional[GraphSnapshot]:
        """Return the most-recent snapshot for *run_id*, or ``None``."""
        with self._connect() as conn:
            run_row = conn.execute(
                "SELECT graph_config FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if not run_row:
                return None
            graph_config = json.loads(run_row[0]) if run_row[0] else None

            snap_row = conn.execute(
                "SELECT step, trigger, state, created_at FROM snapshots "
                "WHERE run_id=? ORDER BY step DESC LIMIT 1",
                (run_id,),
            ).fetchone()

        if not snap_row:
            return None

        state = json.loads(snap_row[2])
        return GraphSnapshot(
            run_id=run_id,
            step=snap_row[0],
            trigger=snap_row[1],
            timestamp=snap_row[3],
            vertex_states=state["vertex_states"],
            edge_states=state["edge_states"],
            graph_config=graph_config,
        )

    def snapshot_count(self, run_id: str) -> int:
        """Return the total number of snapshots saved for *run_id*."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM snapshots WHERE run_id=?",
                (run_id,),
            ).fetchone()
        return row[0] if row else 0

    def load_all_snapshots(self, run_id: str) -> List[GraphSnapshot]:
        """Return every snapshot for *run_id*, ordered by step ascending."""
        with self._connect() as conn:
            run_row = conn.execute(
                "SELECT graph_config FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            graph_config = json.loads(run_row[0]) if (run_row and run_row[0]) else None

            rows = conn.execute(
                "SELECT step, trigger, state, created_at FROM snapshots "
                "WHERE run_id=? ORDER BY step ASC",
                (run_id,),
            ).fetchall()

        snaps = []
        for row in rows:
            state = json.loads(row[2])
            snaps.append(GraphSnapshot(
                run_id=run_id,
                step=row[0],
                trigger=row[1],
                timestamp=row[3],
                vertex_states=state["vertex_states"],
                edge_states=state["edge_states"],
                graph_config=graph_config,
            ))
        return snaps


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
