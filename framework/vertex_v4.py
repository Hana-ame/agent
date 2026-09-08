"""V4 SQLite3-backed storage engine for vertices and session staging.

Provides key-indexed persistence for Vertex records and an isolated
scratchpad staging table attributed to producing edges.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


class VertexStateV4(str, Enum):
    """Execution, lifecycle, and topological traversal states for V4 vertices."""

    DATA_READY = "data ready"
    IDLE = "idle"
    FORBIDDEN = "forbidden"
    TODO = "todo"
    TODO_URGENT = "todo urgent"
    REJECT = "reject"
    PRUNING = "pruning"

    # State coloring and DFS traversal states
    WHITE = "white"
    GRAY = "gray"
    BLACK = "black"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            color_ints = {
                VertexStateV4.WHITE: 0,
                VertexStateV4.GRAY: 1,
                VertexStateV4.BLACK: 2,
            }
            if self in color_ints:
                return color_ints[self] == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


# Unified alias for state coloring functionality
NodeColor = VertexStateV4



class VertexAttributeV4(str, Enum):
    """Functional tags that classify vertex roles and data formats."""

    START = "start"
    END = "end"
    LLM_RESULT = "llm result"
    LLM_PROMPT = "llm prompt"
    JSON = "json"
    PLAIN_TEXT = "plain text"
    SUBGRAPH = "subgraph"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ORPHAN = "orphan"


class MergeStrategyV4(str, Enum):
    """Fan-in merge strategies for vertices receiving from multiple upstream edges."""
    OVERWRITE = "overwrite"      # Default: last write wins
    JSON_MERGE = "json_merge"    # Merge JSON objects: {**existing, **incoming}
    LIST_APPEND = "list_append"  # Append to JSON list
    REDUCER_SCRIPT = "reducer_script"  # Custom reducer function


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
        self._is_memory = (self.db_path == ":memory:")
        self._write_lock = threading.RLock()
        self._conn_lock = threading.RLock()
        self._mem_lock = threading.RLock()
        self._local = threading.local()
        self._connections: List[sqlite3.Connection] = []
        if self._is_memory:
            self._mem_conn = sqlite3.connect(
                ":memory:",
                check_same_thread=False,
                isolation_level=None,
            )
            self._mem_conn.row_factory = sqlite3.Row
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _read_lock(self):
        """Context manager for read operations (lock-free for WAL disk mode)."""
        if self._is_memory:
            return self._mem_lock
        return nullcontext()

    def _write_lock_ctx(self):
        """Context manager for write operations."""
        if self._is_memory:
            return self._mem_lock
        return self._write_lock

    def _get_connection(self) -> sqlite3.Connection:
        if self._is_memory:
            return self._mem_conn
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(
                self.db_path,
                isolation_level=None,  # Autocommit mode, manual transactions where needed
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            self._local.conn = conn
            with self._conn_lock:
                self._connections.append(conn)
        return self._local.conn

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _initialize_schema(self) -> None:
        """Create tables and performance indexes if not present."""
        with self._write_lock_ctx():
            cur = self._get_connection().cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vertices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    attributes TEXT NOT NULL DEFAULT '[]',
                    state TEXT NOT NULL DEFAULT 'idle' CHECK (state IN ('data ready', 'idle', 'forbidden', 'todo', 'todo urgent', 'reject', 'pruning', 'white', 'gray', 'black')),
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
        """Insert or update a vertex record.
        
        Note: Passing processed_count=0 (the default) preserves the existing count on upsert.
        To modify the processed_count after creation, explicitly call increment_processed_count().
        """
        state_str = self._validate_state(state)
        attr_list = self._validate_attributes(attributes)
        attr_json = json.dumps(attr_list)
        now = self._now_iso()

        with self._write_lock_ctx():
            cur = self._get_connection().cursor()
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
        with self._read_lock():
            cur = self._get_connection().cursor()
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
        with self._read_lock():
            cur = self._get_connection().cursor()
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
        state_str = self._validate_state(state)
        now = self._now_iso()
        with self._write_lock_ctx():
            cur = self._get_connection().cursor()
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
        with self._write_lock_ctx():
            cur = self._get_connection().cursor()
            updates = ["content = ?", "updated_at = ?"]
            params: List[Any] = [content, now]

            if state is not None:
                state_str = self._validate_state(state)
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
        with self._write_lock_ctx():
            cur = self._get_connection().cursor()
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

    def apply_merge_strategy(
        self,
        session_id: str,
        name: str,
        incoming_content: str,
        strategy: str = "overwrite",
        reducer_fn: Optional[Callable] = None,
    ) -> str:
        """Apply fan-in merge strategy when multiple edges write to the same vertex."""
        with self._write_lock_ctx():
            if strategy == MergeStrategyV4.OVERWRITE.value or strategy == "overwrite":
                self.update_vertex_content(session_id, name, incoming_content)
                return incoming_content
            
            existing = self.get_vertex(session_id, name)
            existing_content = existing.content if existing else ""
            
            if strategy == MergeStrategyV4.JSON_MERGE.value or strategy == "json_merge":
                try:
                    existing_dict = json.loads(existing_content) if existing_content else {}
                    incoming_dict = json.loads(incoming_content) if incoming_content else {}
                    merged = {**existing_dict, **incoming_dict}
                    merged_content = json.dumps(merged)
                except (json.JSONDecodeError, TypeError):
                    merged_content = incoming_content  # Fall back to overwrite
                self.update_vertex_content(session_id, name, merged_content)
                return merged_content
            
            elif strategy == MergeStrategyV4.LIST_APPEND.value or strategy == "list_append":
                try:
                    existing_list = json.loads(existing_content) if existing_content else []
                    if not isinstance(existing_list, list):
                        existing_list = [existing_list] if existing_content else []
                    existing_list.append(json.loads(incoming_content) if incoming_content else incoming_content)
                except (json.JSONDecodeError, TypeError):
                    existing_list = [existing_content, incoming_content] if existing_content else [incoming_content]
                merged_content = json.dumps(existing_list)
                self.update_vertex_content(session_id, name, merged_content)
                return merged_content
            
            elif strategy == MergeStrategyV4.REDUCER_SCRIPT.value or strategy == "reducer_script":
                if reducer_fn is None:
                    self.update_vertex_content(session_id, name, incoming_content)
                    return incoming_content
                result = reducer_fn(existing_content, incoming_content)
                self.update_vertex_content(session_id, name, str(result))
                return str(result)
            
            # Unknown strategy — fallback to overwrite
            self.update_vertex_content(session_id, name, incoming_content)
            return incoming_content

    def delete_vertex(self, session_id: str, name: str) -> bool:
        """Delete a single vertex record by session_id and name, including associated staging entries."""
        with self._write_lock_ctx():
            cur = self._get_connection().cursor()
            cur.execute("BEGIN;")
            try:
                cur.execute(
                    "DELETE FROM session_staging WHERE session_id = ? AND vertex_name = ?;",
                    (session_id, name),
                )
                cur.execute(
                    "DELETE FROM vertices WHERE session_id = ? AND name = ?;",
                    (session_id, name),
                )
                deleted = cur.rowcount > 0
                cur.execute("COMMIT;")
                return deleted
            except Exception:
                cur.execute("ROLLBACK;")
                raise

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

        with self._write_lock_ctx():
            cur = self._get_connection().cursor()
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
        with self._read_lock():
            cur = self._get_connection().cursor()
            cur.execute(
                "SELECT * FROM edges WHERE session_id = ? AND edge_id = ?;",
                (session_id, edge_id),
            )
            row = cur.fetchone()
            return self._row_to_edge(row) if row else None

    def list_edges(self, session_id: str) -> List[EdgeRecordV4]:
        """List all edges registered for a session."""
        with self._read_lock():
            cur = self._get_connection().cursor()
            cur.execute(
                "SELECT * FROM edges WHERE session_id = ? ORDER BY id ASC;",
                (session_id,),
            )
            return [self._row_to_edge(r) for r in cur.fetchall()]

    def delete_edge(self, session_id: str, edge_id: str) -> bool:
        """Delete a single edge record by session_id and edge_id."""
        with self._write_lock_ctx():
            cur = self._get_connection().cursor()
            cur.execute(
                "DELETE FROM edges WHERE session_id = ? AND edge_id = ?;",
                (session_id, edge_id),
            )
            return cur.rowcount > 0

    def list_sessions(self) -> List[str]:
        """List distinct session IDs across vertices, edges, and staging tables."""
        with self._read_lock():
            cur = self._get_connection().cursor()
            cur.execute(
                "SELECT DISTINCT session_id FROM vertices "
                "UNION SELECT DISTINCT session_id FROM edges "
                "UNION SELECT DISTINCT session_id FROM session_staging;"
            )
            return sorted([row[0] for row in cur.fetchall()])

    def get_db_stats(self) -> Dict[str, int]:
        """Return aggregate statistics across all sessions."""
        with self._read_lock():
            cur = self._get_connection().cursor()
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
        with self._write_lock_ctx():
            cur = self._get_connection().cursor()
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
        with self._read_lock():
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
            cur = self._get_connection().cursor()
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
        with self._read_lock():
            clauses = ["session_id = ?", "vertex_name = ?"]
            params: List[Any] = [session_id, vertex_name]

            if key is not None:
                clauses.append("key = ?")
                params.append(key)

            query = f"SELECT * FROM session_staging WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT 1;"
            cur = self._get_connection().cursor()
            cur.execute(query, params)
            row = cur.fetchone()
            return self._row_to_staging(row) if row else None

    # ------------------------------------------------------------------
    # Maintenance & Cleanup
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> None:
        """Remove all vertices, edges, and staging data for a session."""
        with self._write_lock_ctx():
            cur = self._get_connection().cursor()
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
        """Close SQLite database connection pool."""
        if self._is_memory:
            with self._mem_lock:
                if hasattr(self, "_mem_conn") and self._mem_conn:
                    try:
                        self._mem_conn.close()
                    except Exception as e:
                        logger.warning(f"Error closing in-memory connection: {e}")
        else:
            with self._conn_lock:
                for conn in self._connections:
                    try:
                        conn.close()
                    except Exception as e:
                        logger.warning(f"Error closing connection: {e}")
                self._connections.clear()
                if hasattr(self._local, "conn"):
                    del self._local.conn

    def __enter__(self) -> 'VertexStoreV4':
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _validate_state(self, state: Union[str, VertexStateV4]) -> str:
        state_str = state.value if isinstance(state, VertexStateV4) else str(state)
        normalized = state_str.replace("_", " ").lower()
        valid_states = {s.value for s in VertexStateV4}
        if normalized in valid_states:
            return normalized
        if state_str not in valid_states:
            raise ValueError(f"Invalid state: {state_str}. Must be one of {valid_states}")
        return state_str

    def _validate_attributes(self, attributes: Optional[Sequence[Union[str, VertexAttributeV4]]]) -> List[str]:
        attr_list = [
            a.value if isinstance(a, VertexAttributeV4) else str(a)
            for a in (attributes or [])
        ]
        valid_attrs = {a.value for a in VertexAttributeV4}
        result = []
        for attr in attr_list:
            norm = attr.replace("_", " ").lower()
            if norm in valid_attrs:
                result.append(norm)
            elif attr in valid_attrs:
                result.append(attr)
            else:
                raise ValueError(f"Invalid attribute: {attr}. Must be one of {valid_attrs}")
        return result

    def _row_to_vertex(self, row: sqlite3.Row) -> VertexRecordV4:
        attrs = []
        raw_attrs = row["attributes"]
        if raw_attrs:
            try:
                attrs = json.loads(raw_attrs)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode attributes JSON for vertex {row['name']}: {e}")
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
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode settings JSON for edge {row['edge_id']}: {e}")
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
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode metadata JSON for staging id {row['id']}: {e}")
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
VertexV4 = VertexRecordV4  # Deprecated: use VertexRecordV4 directly
EdgeV4Record = EdgeRecordV4  # Deprecated: use EdgeRecordV4 directly
