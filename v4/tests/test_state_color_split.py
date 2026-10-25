"""Regression tests for the lifecycle-state / traversal-colour split (review §4).

Before the split, ``detect_cycles_and_order`` wrote WHITE/GRAY/BLACK into
``vertex.state`` — the same column that drives the executor's handshake — and the
loader persisted those colours, so an ``idle`` vertex became ``black``.
"""

from __future__ import annotations

import sqlite3

import pytest

from framework.graph_v4 import DiscreteGraphLoaderV4, GraphV4, NodeColor
from framework.vertex_v4 import (
    TraversalColor,
    VertexStateV4,
    VertexStoreV4,
)


class TestEnums:
    def test_lifecycle_states_do_not_contain_colors(self):
        values = {s.value for s in VertexStateV4}
        assert not ({"white", "gray", "black"} & values)
        assert {"data ready", "idle", "todo", "todo urgent", "reject", "forbidden"} <= values

    def test_traversal_colors_are_their_own_enum(self):
        assert {c.value for c in TraversalColor} == {"white", "gray", "black"}
        assert NodeColor is TraversalColor
        assert NodeColor.BLACK == 2
        assert NodeColor.WHITE == 0

    def test_store_rejects_color_as_lifecycle_state(self):
        store = VertexStoreV4(":memory:")
        with pytest.raises(ValueError):
            store.save_vertex("s", "v", content="x", state="black")
        with pytest.raises(ValueError):
            store.update_vertex_state("s", "v", "gray")


class TestValidationDoesNotCorruptState:
    def test_validation_preserves_idle_state(self):
        graph = GraphV4(session_id="s")
        graph.add_vertex({"name": "a", "content": "x", "state": "idle"})
        graph.add_vertex({"name": "b", "content": "", "state": "idle"})
        graph.add_edge({"id": "e", "type": "code", "input_vertex": "a", "output_vertex": "b"})

        graph.validate(strict_dag=False)

        assert {v.name: v.state for v in graph.vertices.values()} == {"a": "idle", "b": "idle"}
        # Colours are still reported, just not persisted onto the vertices.
        assert graph.node_states["a"] == TraversalColor.BLACK

    def test_loader_and_dump_preserve_declared_state(self):
        graph = DiscreteGraphLoaderV4.load_from_dict(
            {
                "vertices": [
                    {"name": "a", "content": "x", "state": "idle"},
                    {"name": "b", "content": "", "state": "todo"},
                ],
                "edges": [{"id": "e", "type": "code", "input_vertex": "a", "output_vertex": "b"}],
            },
            override_session_id="s",
        )
        assert graph.vertices["a"].state == "idle"
        store = VertexStoreV4(":memory:")
        DiscreteGraphLoaderV4.populate_store(graph, store)
        assert store.get_vertex("s", "a").state == "idle"
        dumped = graph.dump()
        states = {v["name"]: v["state"] for v in dumped["vertices"]}
        assert states["a"] == "idle"
        assert "black" not in states.values()

    def test_data_ready_state_is_untouched(self):
        graph = GraphV4(session_id="s")
        graph.add_vertex({"name": "a", "content": "x", "state": "data ready"})
        graph.add_vertex({"name": "b", "content": "", "state": "todo"})
        graph.add_edge({"id": "e", "type": "code", "input_vertex": "a", "output_vertex": "b"})
        graph.validate(strict_dag=False)
        assert graph.vertices["a"].state == "data ready"
        assert graph.vertices["b"].state == "todo"


class TestSchemaMigration:
    _OLD_DDL = """
        CREATE TABLE vertices (
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

    def _make_legacy_db(self, path) -> None:
        conn = sqlite3.connect(str(path))
        conn.executescript(self._OLD_DDL)
        conn.execute(
            "INSERT INTO vertices (session_id, name, content, attributes, state, processed_count) "
            "VALUES ('s', 'colored', 'x', '[]', 'black', 2);"
        )
        conn.execute(
            "INSERT INTO vertices (session_id, name, content, attributes, state, processed_count) "
            "VALUES ('s', 'real', 'y', '[\"start\"]', 'data ready', 5);"
        )
        conn.commit()
        conn.close()

    def test_legacy_colors_are_migrated_to_idle(self, tmp_path):
        db = tmp_path / "legacy.db"
        self._make_legacy_db(db)

        store = VertexStoreV4(str(db))
        try:
            colored = store.get_vertex("s", "colored")
            real = store.get_vertex("s", "real")
            assert colored is not None and real is not None
            assert colored.state == VertexStateV4.IDLE.value
            assert colored.processed_count == 2
            # Untouched rows keep their state, attributes and counter.
            assert real.state == VertexStateV4.DATA_READY.value
            assert real.attributes == ["start"]
            assert real.processed_count == 5

            # The rebuilt table no longer accepts traversal colours.
            with pytest.raises(ValueError):
                store.save_vertex("s", "colored", content="z", state="black")
        finally:
            store.close()

    def test_migration_is_idempotent_and_versioned(self, tmp_path):
        db = tmp_path / "legacy2.db"
        self._make_legacy_db(db)

        store = VertexStoreV4(str(db))
        try:
            conn = store._get_connection()
            version = conn.execute("PRAGMA user_version;").fetchone()[0]
            assert version == 1
            ddl = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='vertices';"
            ).fetchone()[0]
            assert "white" not in ddl.lower()
            assert "black" not in ddl.lower()
        finally:
            store.close()

        # Re-opening an already-migrated database is a no-op.
        store2 = VertexStoreV4(str(db))
        try:
            assert store2.get_vertex("s", "real").state == VertexStateV4.DATA_READY.value
        finally:
            store2.close()
