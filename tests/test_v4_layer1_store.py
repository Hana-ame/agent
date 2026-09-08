"""Layer 1: Storage Layer (VertexStoreV4) Comprehensive Test Suite.

Validates:
1. Schema integrity & database check constraints (valid/invalid states, unique keys).
2. Vertex and Edge CRUD lifecycle (upsert, read, update, increment, delete).
3. Session staging scratchpad attribution and multi-criteria queries.
4. Session isolation and atomic cascade cleanup.
5. SQLite WAL multi-reader concurrency without lock blocking on disk databases.
6. Multi-threaded in-memory database stability without table isolation errors.
7. Resilience against malformed JSON data.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
from typing import List

import pytest

from framework.vertex_v4 import (
    EdgeRecordV4,
    StagingRecordV4,
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)


@pytest.fixture
def mem_store() -> VertexStoreV4:
    """Fixture providing an in-memory SQLite store."""
    store = VertexStoreV4(":memory:")
    yield store
    store.close()


@pytest.fixture
def disk_store() -> VertexStoreV4:
    """Fixture providing a temporary disk-backed SQLite store with WAL enabled."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = VertexStoreV4(db_path)
    yield store
    store.close()
    for ext in ["", "-wal", "-shm"]:
        target = db_path + ext
        if os.path.exists(target):
            try:
                os.unlink(target)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 1. Schema Integrity & Constraints
# ---------------------------------------------------------------------------

def test_state_check_constraint_valid_states(mem_store: VertexStoreV4) -> None:
    """Verify that all 7 official lifecycle states are accepted."""
    session_id = "test_valid_states"
    for state in VertexStateV4:
        v = mem_store.save_vertex(
            session_id=session_id,
            name=f"v_{state.name}",
            content=f"content_{state.value}",
            state=state,
        )
        assert v.state == state.value

    # Direct query verification
    records = mem_store.list_vertices(session_id)
    assert len(records) == len(VertexStateV4)


def test_state_validation_rejects_invalid_state(mem_store: VertexStoreV4) -> None:
    """Verify that invalid states are rejected by pre-validation and CHECK constraints."""
    session_id = "test_invalid_state"
    with pytest.raises(ValueError, match="Invalid state"):
        mem_store.save_vertex(session_id, "bad_v", state="running")

    with pytest.raises(ValueError, match="Invalid state"):
        mem_store.update_vertex_state(session_id, "bad_v", state="unknown_state")


def test_attribute_validation_valid_and_invalid(mem_store: VertexStoreV4) -> None:
    """Verify that vertex attributes are strictly validated against VertexAttributeV4."""
    session_id = "test_attributes"
    # Valid attributes
    v = mem_store.save_vertex(
        session_id=session_id,
        name="attr_vertex",
        attributes=[VertexAttributeV4.START, VertexAttributeV4.LLM_PROMPT],
    )
    assert v.attributes == ["start", "llm prompt"]

    # Invalid attribute string
    with pytest.raises(ValueError, match="Invalid attribute"):
        mem_store.save_vertex(
            session_id=session_id,
            name="bad_attr",
            attributes=["unsupported_tag"],
        )


def test_unique_constraint_and_upsert_vertex(mem_store: VertexStoreV4) -> None:
    """Verify that (session_id, name) is unique and save_vertex performs atomic upsert."""
    session_id = "test_upsert_vertex"
    v1 = mem_store.save_vertex(session_id, "node_a", content="initial", state=VertexStateV4.IDLE)
    assert v1.content == "initial"
    assert v1.state == VertexStateV4.IDLE.value

    v2 = mem_store.save_vertex(session_id, "node_a", content="updated", state=VertexStateV4.DATA_READY)
    assert v2.id == v1.id
    assert v2.content == "updated"
    assert v2.state == VertexStateV4.DATA_READY.value

    # Processed count preservation: passing default 0 preserves existing count
    mem_store.increment_processed_count(session_id, "node_a")
    v3 = mem_store.get_vertex(session_id, "node_a")
    assert v3 is not None
    assert v3.processed_count == 1

    v4 = mem_store.save_vertex(session_id, "node_a", content="v4 content")
    assert v4.processed_count == 1  # Preserved!


def test_unique_constraint_and_upsert_edge(mem_store: VertexStoreV4) -> None:
    """Verify that (session_id, edge_id) is unique and save_edge performs atomic upsert."""
    session_id = "test_upsert_edge"
    e1 = mem_store.save_edge(
        session_id=session_id,
        edge_id="e1",
        input_vertex="A",
        output_vertex="B",
        settings={"param": 1},
    )
    assert e1.settings == {"param": 1}

    e2 = mem_store.save_edge(
        session_id=session_id,
        edge_id="e1",
        input_vertex="A",
        output_vertex="C",
        settings={"param": 2},
    )
    assert e2.id == e1.id
    assert e2.output_vertex == "C"
    assert e2.settings == {"param": 2}


# ---------------------------------------------------------------------------
# 2. CRUD Operations & Lifecycle Transitions
# ---------------------------------------------------------------------------

def test_vertex_crud_lifecycle(mem_store: VertexStoreV4) -> None:
    """Test comprehensive vertex CRUD operations and partial field updates."""
    sess = "sess_crud"
    # Create
    v = mem_store.save_vertex(sess, "test_node", content="payload", state=VertexStateV4.TODO)
    assert v.name == "test_node"

    # Read
    fetched = mem_store.get_vertex(sess, "test_node")
    assert fetched is not None
    assert fetched.content == "payload"
    assert fetched.state == VertexStateV4.TODO.value

    # Update State
    updated_state = mem_store.update_vertex_state(sess, "test_node", VertexStateV4.DATA_READY)
    assert updated_state is True
    assert mem_store.get_vertex(sess, "test_node").state == VertexStateV4.DATA_READY.value

    # Update Content with increment_count
    updated_content = mem_store.update_vertex_content(
        sess, "test_node", content="new_payload", increment_count=True
    )
    assert updated_content is True
    v_updated = mem_store.get_vertex(sess, "test_node")
    assert v_updated.content == "new_payload"
    assert v_updated.processed_count == 1

    # List with filtering
    mem_store.save_vertex(sess, "idle_node", state=VertexStateV4.IDLE)
    data_ready_list = mem_store.list_vertices(sess, state=VertexStateV4.DATA_READY)
    assert len(data_ready_list) == 1
    assert data_ready_list[0].name == "test_node"

    # Delete
    deleted = mem_store.delete_vertex(sess, "test_node")
    assert deleted is True
    assert mem_store.get_vertex(sess, "test_node") is None


def test_edge_crud_lifecycle(mem_store: VertexStoreV4) -> None:
    """Test edge creation, retrieval, listing, and deletion."""
    sess = "sess_edge_crud"
    e = mem_store.save_edge(
        session_id=sess,
        edge_id="edge_transform",
        edge_type="code",
        input_vertex="v_in",
        output_vertex="v_out",
        script="examples.demo.transform",
        settings={"batch_size": 16},
    )
    assert e.edge_id == "edge_transform"
    assert e.script == "examples.demo.transform"

    # Fetch
    fetched = mem_store.get_edge(sess, "edge_transform")
    assert fetched is not None
    assert fetched.settings["batch_size"] == 16

    # List
    all_edges = mem_store.list_edges(sess)
    assert len(all_edges) == 1
    assert all_edges[0].edge_id == "edge_transform"

    # Delete
    deleted = mem_store.delete_edge(sess, "edge_transform")
    assert deleted is True
    assert mem_store.get_edge(sess, "edge_transform") is None


def test_session_staging_attribution_and_queries(mem_store: VertexStoreV4) -> None:
    """Test recording intermediate staging data and multi-criteria retrieval."""
    sess = "sess_staging"
    id1 = mem_store.stage_output(
        session_id=sess,
        edge_id="edge_1",
        vertex_name="out_v",
        key="prompt_trace",
        value="Prompt: hello",
        metadata={"tokens": 12},
    )
    assert id1 > 0

    id2 = mem_store.stage_output(
        session_id=sess,
        edge_id="edge_1",
        vertex_name="out_v",
        key="prompt_trace",
        value="Prompt: hello v2",
        metadata={"tokens": 14},
    )
    assert id2 > id1

    # Query latest staged for vertex
    latest = mem_store.get_latest_staged_for_vertex(sess, "out_v", key="prompt_trace")
    assert latest is not None
    assert latest.value == "Prompt: hello v2"
    assert latest.metadata == {"tokens": 14}

    # Query with edge_id filter
    staged_edge = mem_store.get_staged(sess, edge_id="edge_1")
    assert len(staged_edge) == 2


# ---------------------------------------------------------------------------
# 3. Session Isolation & Cascade Cleanup
# ---------------------------------------------------------------------------

def test_session_data_isolation(mem_store: VertexStoreV4) -> None:
    """Ensure vertices, edges, and staging are isolated across sessions."""
    sess_a = "session_alpha"
    sess_b = "session_beta"

    mem_store.save_vertex(sess_a, "shared_name", content="Alpha Data")
    mem_store.save_vertex(sess_b, "shared_name", content="Beta Data")

    assert mem_store.get_vertex(sess_a, "shared_name").content == "Alpha Data"
    assert mem_store.get_vertex(sess_b, "shared_name").content == "Beta Data"

    sessions = mem_store.list_sessions()
    assert sess_a in sessions
    assert sess_b in sessions


def test_clear_session_atomic(mem_store: VertexStoreV4) -> None:
    """Ensure clear_session wipes all data for the session and leaves others intact."""
    sess_a = "sess_to_clear"
    sess_b = "sess_to_keep"

    mem_store.save_vertex(sess_a, "v1", "payload")
    mem_store.save_edge(sess_a, "e1", input_vertex="v1", output_vertex="v2")
    mem_store.stage_output(sess_a, "e1", "k", "v")

    mem_store.save_vertex(sess_b, "v_keep", "safe")

    mem_store.clear_session(sess_a)

    assert len(mem_store.list_vertices(sess_a)) == 0
    assert len(mem_store.list_edges(sess_a)) == 0
    assert len(mem_store.get_staged(sess_a)) == 0

    assert len(mem_store.list_vertices(sess_b)) == 1


def test_delete_vertex_cascades_staging(mem_store: VertexStoreV4) -> None:
    """Verify that deleting a vertex automatically cleans up its attributed staging entries."""
    sess = "sess_cascade"
    mem_store.save_vertex(sess, "target_node", "data")
    mem_store.stage_output(sess, "edge_x", "k1", "v1", vertex_name="target_node")
    mem_store.stage_output(sess, "edge_x", "k2", "v2", vertex_name="other_node")

    deleted = mem_store.delete_vertex(sess, "target_node")
    assert deleted is True

    staged = mem_store.get_staged(sess)
    assert len(staged) == 1
    assert staged[0].vertex_name == "other_node"


def test_db_stats_aggregation(mem_store: VertexStoreV4) -> None:
    """Test get_db_stats returns correct aggregate counts across sessions."""
    mem_store.save_vertex("s1", "v1")
    mem_store.save_vertex("s1", "v2")
    mem_store.save_vertex("s2", "v1")
    mem_store.save_edge("s1", "e1", input_vertex="v1", output_vertex="v2")
    mem_store.stage_output("s1", "e1", "key", "val")

    stats = mem_store.get_db_stats()
    assert stats["total_vertices"] == 3
    assert stats["total_edges"] == 1
    assert stats["total_staging"] == 1
    assert stats["active_sessions"] == 2


# ---------------------------------------------------------------------------
# 4. Concurrency & WAL Scalability
# ---------------------------------------------------------------------------

def test_disk_wal_concurrent_readers_during_active_write(disk_store: VertexStoreV4) -> None:
    """Verify that SQLite WAL mode allows concurrent reads without waiting for write transactions.
    
    A writer holds an active transaction while a reader accesses the database.
    The read must return immediately with a consistent snapshot (< 50ms).
    """
    sess = "wal_concurrency_test"
    disk_store.save_vertex(sess, "v_shared", content="initial_snapshot", state=VertexStateV4.IDLE)

    write_started = threading.Event()
    release_write = threading.Event()

    def long_write_task() -> None:
        with disk_store._write_lock_ctx():
            conn = disk_store._get_connection()
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                "UPDATE vertices SET content = 'uncommitted_in_progress' WHERE session_id = ? AND name = 'v_shared';",
                (sess,),
            )
            write_started.set()
            release_write.wait(timeout=2.0)
            conn.execute("COMMIT;")

    writer_thread = threading.Thread(target=long_write_task)
    writer_thread.start()

    write_started.wait(timeout=1.0)

    # Reader thread performs read while writer transaction is active
    reader_results: List[tuple] = []

    def reader_task() -> None:
        t0 = time.time()
        v = disk_store.get_vertex(sess, "v_shared")
        elapsed = time.time() - t0
        reader_results.append((v, elapsed))

    reader_thread = threading.Thread(target=reader_task)
    reader_thread.start()
    reader_thread.join(timeout=1.0)

    release_write.set()
    writer_thread.join(timeout=1.0)

    assert len(reader_results) == 1
    read_vertex, read_duration = reader_results[0]
    assert read_vertex is not None
    # WAL reads consistent committed snapshot
    assert read_vertex.content == "initial_snapshot"
    # Non-blocking: duration should be well under 0.1s
    assert read_duration < 0.1


def test_in_memory_multi_threaded_stability(mem_store: VertexStoreV4) -> None:
    """Verify that :memory: databases are thread-safe and share state across threads."""
    sess = "mem_thread_test"
    mem_store.save_vertex(sess, "root", "root_data")

    def worker(worker_id: int) -> None:
        mem_store.save_vertex(sess, f"worker_{worker_id}", f"data_{worker_id}")
        mem_store.stage_output(sess, f"edge_{worker_id}", "key", f"val_{worker_id}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    vertices = mem_store.list_vertices(sess)
    assert len(vertices) == 9  # root + 8 workers
    staging = mem_store.get_staged(sess)
    assert len(staging) == 8


def test_concurrent_writes_serialized_gracefully(disk_store: VertexStoreV4) -> None:
    """Verify high-concurrency writes from 10 parallel threads succeed without lock errors."""
    sess = "stress_write_test"
    errors: List[Exception] = []

    def writer(idx: int) -> None:
        try:
            for j in range(5):
                disk_store.save_vertex(
                    session_id=sess,
                    name=f"node_{idx}_{j}",
                    content=f"val_{idx}_{j}",
                    state=VertexStateV4.DATA_READY,
                )
                disk_store.stage_output(
                    session_id=sess,
                    edge_id=f"e_{idx}_{j}",
                    key="log",
                    value=f"entry_{idx}_{j}",
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    all_vertices = disk_store.list_vertices(sess)
    assert len(all_vertices) == 50  # 10 threads * 5 vertices


# ---------------------------------------------------------------------------
# 5. Resilience to Corrupted JSON
# ---------------------------------------------------------------------------

def test_malformed_json_attributes_graceful_fallback(disk_store: VertexStoreV4) -> None:
    """Verify that malformed JSON in attributes or settings falls back safely without crashing."""
    sess = "corrupt_json_test"
    disk_store.save_vertex(sess, "node_raw", "content")

    # Manually corrupt JSON in SQLite
    conn = disk_store._get_connection()
    with disk_store._write_lock_ctx():
        conn.execute(
            "UPDATE vertices SET attributes = '{corrupted_not_json' WHERE session_id = ? AND name = 'node_raw';",
            (sess,),
        )

    v = disk_store.get_vertex(sess, "node_raw")
    assert v is not None
    # Graceful fallback to empty list
    assert v.attributes == []
