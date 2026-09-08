"""V4 SQLite3-backed storage engine for vertices and session staging.

Provides key-indexed persistence for Vertex records and an isolated
scratchpad staging table attributed to producing edges.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


class VertexStateV4(str, Enum):
    """Execution and lifecycle states for V4 vertices."""

    DATA_READY = "data ready"
    IDLE = "idle"
    FORBIDDEN = "forbidden"
    TODO = "todo"
    TODO_URGENT = "todo urgent"
    REJECT = "reject"
    PRUNING = "pruning"


class VertexAttributeV4(str, Enum):
    """Functional tags that classify vertex roles and data formats."""

    START = "start"
    END = "end"
    LLM_RESULT = "llm result"
    LLM_PROMPT = "llm prompt"
    JSON = "json"
    PLAIN_TEXT = "plain text"
    SUBGRAPH = "subgraph"


@dataclass
class VertexRecordV4:
    """Represents a row in the vertices table."""

    id: int
    session_id: str
    name: str
    content: str = ""
    attributes: List[str] = field(default_factory=list)
    state: str = VertexStateV4.IDLE.value
    processed_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def has_attribute(self, attr: Union[str, VertexAttributeV4]) -> bool:
        """Check if vertex contains the specified attribute."""
        target = attr.value if isinstance(attr, VertexAttributeV4) else str(attr)
        return target in self.attributes

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "name": self.name,
            "content": self.content,
            "attributes": list(self.attributes),
            "state": self.state,
            "processed_count": self.processed_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class StagingRecordV4:
    """Represents an entry in the session_staging scratchpad table."""

    id: int
    session_id: str
    edge_id: str
    vertex_name: Optional[str]
    key: str
    value: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize staging entry to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "edge_id": self.edge_id,
            "vertex_name": self.vertex_name,
            "key": self.key,
            "value": self.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class EdgeRecordV4:
    """Represents an entry in the edges SQLite table."""

    id: int
    session_id: str
    edge_id: str
    edge_type: str
    input_vertex: str
    output_vertex: str
    script: Optional[str] = None
    trigger_state: Optional[str] = None
    target_state: Optional[str] = None
    max_retries: int = 3
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge record to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "input_vertex": self.input_vertex,
            "output_vertex": self.output_vertex,
            "script": self.script,
            "trigger_state": self.trigger_state,
            "target_state": self.target_state,
            "max_retries": self.max_retries,
            "settings": self.settings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class VertexStoreV4:
    """SQLite3 key-indexed storage engine for V4 vertices, edges, and session staging."""

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode, manual transactions where needed
        )
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _initialize_schema(self) -> None:
        """Create tables and performance indexes if not present."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode = WAL;")
            cur.execute("PRAGMA synchronous = NORMAL;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vertices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    attributes TEXT NOT NULL DEFAULT '[]',
                    state TEXT NOT NULL DEFAULT 'idle' CHECK (state IN ('data ready', 'idle', 'forbidden', 'todo', 'todo urgent', 'reject', 'pruning')),
                    processed_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(session_id, name)
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_vertices_session_state ON vertices(session_id, state);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_vertices_session_name ON vertices(session_id, name);"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_staging (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    edge_id TEXT NOT NULL,
                    vertex_name TEXT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_staging_session_edge ON session_staging(session_id, edge_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_staging_session_key ON session_staging(session_id, key);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_staging_session_vertex ON session_staging(session_id, vertex_name);"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    edge_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL DEFAULT 'code',
                    input_vertex TEXT NOT NULL,
                    output_vertex TEXT NOT NULL,
                    script TEXT,
                    trigger_state TEXT,
                    target_state TEXT,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    settings TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(session_id, edge_id)
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_session ON edges(session_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_session_endpoints ON edges(session_id, input_vertex, output_vertex);"
            )

    # ------------------------------------------------------------------
    # Vertex Operations
    # ------------------------------------------------------------------

    def save_vertex(
        self,
        session_id: str,
        name: str,
        content: str = "",
        attributes: Optional[Sequence[Union[str, VertexAttributeV4]]] = None,
        state: Union[str, VertexStateV4] = VertexStateV4.IDLE,
        processed_count: int = 0,
    ) -> VertexRecordV4:
        """Insert or update a vertex record."""
        state_str = state.value if isinstance(state, VertexStateV4) else str(state)
        attr_list = [
            a.value if isinstance(a, VertexAttributeV4) else str(a)
            for a in (attributes or [])
        ]
        attr_json = json.dumps(attr_list)
        now = self._now_iso()

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO vertices (session_id, name, content, attributes, state, processed_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, name) DO UPDATE SET
                    content = excluded.content,
                    attributes = excluded.attributes,
                    state = excluded.state,
                    processed_count = CASE WHEN excluded.processed_count > 0 THEN excluded.processed_count ELSE vertices.processed_count END,
                    updated_at = excluded.updated_at
                RETURNING id, session_id, name, content, attributes, state, processed_count, created_at, updated_at;
                """,
                (session_id, name, content, attr_json, state_str, processed_count, now, now),
            )
            row = cur.fetchone()
            return self._row_to_vertex(row)

    def get_vertex(self, session_id: str, name: str) -> Optional[VertexRecordV4]:
        """Fetch a single vertex record by session_id and name."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM vertices WHERE session_id = ? AND name = ?;",
                (session_id, name),
            )
            row = cur.fetchone()
            return self._row_to_vertex(row) if row else None

    def list_vertices(
        self,
        session_id: str,
        state: Optional[Union[str, VertexStateV4]] = None,
    ) -> List[VertexRecordV4]:
        """List all vertices in a session, optionally filtered by state."""
        with self._lock:
            cur = self._conn.cursor()
            if state is not None:
                state_str = state.value if isinstance(state, VertexStateV4) else str(state)
                cur.execute(
                    "SELECT * FROM vertices WHERE session_id = ? AND state = ? ORDER BY id ASC;",
                    (session_id, state_str),
                )
            else:
                cur.execute(
                    "SELECT * FROM vertices WHERE session_id = ? ORDER BY id ASC;",
                    (session_id,),
                )
            rows = cur.fetchall()
            return [self._row_to_vertex(r) for r in rows]

    def update_vertex_state(
        self,
        session_id: str,
        name: str,
        state: Union[str, VertexStateV4],
    ) -> bool:
        """Update lifecycle state of a vertex."""
        state_str = state.value if isinstance(state, VertexStateV4) else str(state)
        now = self._now_iso()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE vertices
                SET state = ?, updated_at = ?
                WHERE session_id = ? AND name = ?;
                """,
                (state_str, now, session_id, name),
            )
            return cur.rowcount > 0

    def update_vertex_content(
        self,
        session_id: str,
        name: str,
        content: str,
        state: Optional[Union[str, VertexStateV4]] = None,
        increment_count: bool = False,
    ) -> bool:
        """Update content and optionally update state and increment processed count."""
        now = self._now_iso()
        with self._lock:
            cur = self._conn.cursor()
            updates = ["content = ?", "updated_at = ?"]
            params: List[Any] = [content, now]

            if state is not None:
                state_str = state.value if isinstance(state, VertexStateV4) else str(state)
                updates.append("state = ?")
                params.append(state_str)

            if increment_count:
                updates.append("processed_count = processed_count + 1")

            params.extend([session_id, name])
            query = f"UPDATE vertices SET {', '.join(updates)} WHERE session_id = ? AND name = ?;"
            cur.execute(query, params)
            return cur.rowcount > 0

    def increment_processed_count(self, session_id: str, name: str) -> int:
        """Atomically increment processing count and return new value."""
        now = self._now_iso()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE vertices
                SET processed_count = processed_count + 1, updated_at = ?
                WHERE session_id = ? AND name = ?
                RETURNING processed_count;
                """,
                (now, session_id, name),
            )
            row = cur.fetchone()
            return int(row["processed_count"]) if row else 0

    def delete_vertex(self, session_id: str, name: str) -> bool:
        """Delete a single vertex record by session_id and name, including associated staging entries."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "DELETE FROM session_staging WHERE session_id = ? AND vertex_name = ?;",
                (session_id, name),
            )
            cur.execute(
                "DELETE FROM vertices WHERE session_id = ? AND name = ?;",
                (session_id, name),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Edge Operations
    # ------------------------------------------------------------------

    def save_edge(
        self,
        session_id: str,
        edge_id: str,
        edge_type: str = "code",
        input_vertex: str = "",
        output_vertex: str = "",
        script: Optional[str] = None,
        trigger_state: Optional[str] = None,
        target_state: Optional[str] = None,
        max_retries: int = 3,
        settings: Optional[Dict[str, Any]] = None,
    ) -> EdgeRecordV4:
        """Insert or update an edge record."""
        settings_json = json.dumps(settings or {})
        now = self._now_iso()

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO edges (
                    session_id, edge_id, edge_type, input_vertex, output_vertex,
                    script, trigger_state, target_state, max_retries, settings,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, edge_id) DO UPDATE SET
                    edge_type = excluded.edge_type,
                    input_vertex = excluded.input_vertex,
                    output_vertex = excluded.output_vertex,
                    script = excluded.script,
                    trigger_state = excluded.trigger_state,
                    target_state = excluded.target_state,
                    max_retries = excluded.max_retries,
                    settings = excluded.settings,
                    updated_at = excluded.updated_at
                RETURNING id, session_id, edge_id, edge_type, input_vertex, output_vertex,
                          script, trigger_state, target_state, max_retries, settings, created_at, updated_at;
                """,
                (
                    session_id, edge_id, edge_type, input_vertex, output_vertex,
                    script, trigger_state, target_state, max_retries, settings_json,
                    now, now,
                ),
            )
            row = cur.fetchone()
            return self._row_to_edge(row)

    def get_edge(self, session_id: str, edge_id: str) -> Optional[EdgeRecordV4]:
        """Fetch a single edge record by session_id and edge_id."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM edges WHERE session_id = ? AND edge_id = ?;",
                (session_id, edge_id),
            )
            row = cur.fetchone()
            return self._row_to_edge(row) if row else None

    def list_edges(self, session_id: str) -> List[EdgeRecordV4]:
        """List all edges registered for a session."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM edges WHERE session_id = ? ORDER BY id ASC;",
                (session_id,),
            )
            return [self._row_to_edge(r) for r in cur.fetchall()]

    def delete_edge(self, session_id: str, edge_id: str) -> bool:
        """Delete a single edge record by session_id and edge_id."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "DELETE FROM edges WHERE session_id = ? AND edge_id = ?;",
                (session_id, edge_id),
            )
            return cur.rowcount > 0

    def list_sessions(self) -> List[str]:
        """List distinct session IDs across vertices, edges, and staging tables."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT DISTINCT session_id FROM vertices "
                "UNION SELECT DISTINCT session_id FROM edges "
                "UNION SELECT DISTINCT session_id FROM session_staging;"
            )
            return sorted([row[0] for row in cur.fetchall()])

    def get_db_stats(self) -> Dict[str, int]:
        """Return aggregate statistics across all sessions."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM vertices;")
            total_vertices = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM edges;")
            total_edges = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM session_staging;")
            total_staging = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM ("
                "  SELECT DISTINCT session_id FROM vertices "
                "  UNION SELECT DISTINCT session_id FROM edges "
                "  UNION SELECT DISTINCT session_id FROM session_staging"
                ");"
            )
            active_sessions = cur.fetchone()[0]
        return {
            "total_vertices": total_vertices,
            "total_edges": total_edges,
            "total_staging": total_staging,
            "active_sessions": active_sessions,
        }

    # ------------------------------------------------------------------
    # Session Staging Operations
    # ------------------------------------------------------------------

    def stage_output(
        self,
        session_id: str,
        edge_id: str,
        key: str,
        value: str,
        vertex_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record intermediate data or diagnostics tagged with producing edge."""
        meta_json = json.dumps(metadata or {})
        now = self._now_iso()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO session_staging (session_id, edge_id, vertex_name, key, value, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                RETURNING id;
                """,
                (session_id, edge_id, vertex_name, key, value, meta_json, now),
            )
            row = cur.fetchone()
            return int(row["id"])

    def get_staged(
        self,
        session_id: str,
        key: Optional[str] = None,
        edge_id: Optional[str] = None,
        vertex_name: Optional[str] = None,
    ) -> List[StagingRecordV4]:
        """Query staged scratchpad records within a session."""
        with self._lock:
            clauses = ["session_id = ?"]
            params: List[Any] = [session_id]

            if key is not None:
                clauses.append("key = ?")
                params.append(key)
            if edge_id is not None:
                clauses.append("edge_id = ?")
                params.append(edge_id)
            if vertex_name is not None:
                clauses.append("vertex_name = ?")
                params.append(vertex_name)

            query = f"SELECT * FROM session_staging WHERE {' AND '.join(clauses)} ORDER BY id ASC;"
            cur = self._conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
            return [self._row_to_staging(r) for r in rows]

    def get_latest_staged_for_vertex(
        self,
        session_id: str,
        vertex_name: str,
        key: Optional[str] = None,
    ) -> Optional[StagingRecordV4]:
        """Fetch the most recent staged entry for a specific vertex."""
        with self._lock:
            clauses = ["session_id = ?", "vertex_name = ?"]
            params: List[Any] = [session_id, vertex_name]

            if key is not None:
                clauses.append("key = ?")
                params.append(key)

            query = f"SELECT * FROM session_staging WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT 1;"
            cur = self._conn.cursor()
            cur.execute(query, params)
            row = cur.fetchone()
            return self._row_to_staging(row) if row else None

    # ------------------------------------------------------------------
    # Maintenance & Cleanup
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> None:
        """Remove all vertices, edges, and staging data for a session."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN;")
            try:
                cur.execute("DELETE FROM vertices WHERE session_id = ?;", (session_id,))
                cur.execute("DELETE FROM edges WHERE session_id = ?;", (session_id,))
                cur.execute("DELETE FROM session_staging WHERE session_id = ?;", (session_id,))
                cur.execute("COMMIT;")
            except Exception:
                cur.execute("ROLLBACK;")
                raise

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> 'VertexStoreV4':
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _row_to_vertex(self, row: sqlite3.Row) -> VertexRecordV4:
        attrs = []
        raw_attrs = row["attributes"]
        if raw_attrs:
            try:
                attrs = json.loads(raw_attrs)
            except Exception:
                attrs = []
        return VertexRecordV4(
            id=row["id"],
            session_id=row["session_id"],
            name=row["name"],
            content=row["content"],
            attributes=attrs,
            state=row["state"],
            processed_count=row["processed_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_edge(self, row: sqlite3.Row) -> EdgeRecordV4:
        settings = {}
        raw_settings = row["settings"]
        if raw_settings:
            try:
                settings = json.loads(raw_settings)
            except Exception:
                settings = {}
        return EdgeRecordV4(
            id=row["id"],
            session_id=row["session_id"],
            edge_id=row["edge_id"],
            edge_type=row["edge_type"],
            input_vertex=row["input_vertex"],
            output_vertex=row["output_vertex"],
            script=row["script"],
            trigger_state=row["trigger_state"],
            target_state=row["target_state"],
            max_retries=row["max_retries"],
            settings=settings,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_staging(self, row: sqlite3.Row) -> StagingRecordV4:
        metadata = {}
        raw_meta = row["metadata"]
        if raw_meta:
            try:
                metadata = json.loads(raw_meta)
            except Exception:
                metadata = {}
        return StagingRecordV4(
            id=row["id"],
            session_id=row["session_id"],
            edge_id=row["edge_id"],
            vertex_name=row["vertex_name"],
            key=row["key"],
            value=row["value"],
            metadata=metadata,
            created_at=row["created_at"],
        )


# Alias for backward compatibility or direct instantiation
VertexV4 = VertexRecordV4
EdgeV4Record = EdgeRecordV4
