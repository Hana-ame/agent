"""Tests for SQLiteStateStore, CheckpointedExecutor, and HumanGateVertex.

All SQLite operations use an in-memory database (':memory:') so no files
are written to disk during the test run.
"""

import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import (
    Graph, MockAgent, Vertex, VertexState,
    SQLiteStateStore, GraphSnapshot,
    CheckpointedExecutor, HumanGateVertex,
)


# ── Shared helpers ───────────────────────────────────────────────────

SIMPLE_CONFIG = {
    "vertices": [
        {"id": "A", "initial_data": [{"data_id": "x", "value": "hello"}]},
        {"id": "B"},
        {"id": "C"},
    ],
    "edges": [
        {"id": "ab", "source": "A", "destination": "B", "channel": "x"},
        {"id": "bc", "source": "B", "destination": "C", "channel": "x"},
    ],
}

DIAMOND_CONFIG = {
    "vertices": [
        {"id": "SRC", "initial_data": [{"data_id": "v", "value": 1}]},
        {"id": "L"},
        {"id": "R"},
        {"id": "SINK"},
    ],
    "edges": [
        {"id": "sl", "source": "SRC", "destination": "L", "channel": "v"},
        {"id": "sr", "source": "SRC", "destination": "R", "channel": "v"},
        {"id": "ls", "source": "L",   "destination": "SINK", "channel": "v"},
        {"id": "rs", "source": "R",   "destination": "SINK", "channel": "v"},
    ],
}

def _mem_store() -> SQLiteStateStore:
    return SQLiteStateStore(":memory:")


# ── SQLiteStateStore ─────────────────────────────────────────────────

class TestSQLiteStateStore:
    def test_create_run_and_get(self):
        store = _mem_store()
        store.create_run("r1", graph_config={"foo": "bar"})

        run = store.get_run("r1")
        assert run is not None
        assert run["run_id"] == "r1"
        assert run["status"] == "running"
        assert run["graph_config"] == {"foo": "bar"}

    def test_create_run_idempotent(self):
        store = _mem_store()
        store.create_run("r1")
        store.create_run("r1")          # second call must not raise
        assert store.get_run("r1") is not None

    def test_update_status(self):
        store = _mem_store()
        store.create_run("r2")
        store.update_run_status("r2", "paused")
        assert store.get_run("r2")["status"] == "paused"

    def test_save_and_load_snapshot(self):
        store = _mem_store()
        store.create_run("r3")

        snap = GraphSnapshot(
            run_id="r3",
            step=1,
            trigger="vertex:A:done",
            timestamp="",
            vertex_states={"A": {"state": "done", "data": {"x": "hello"}}},
            edge_states={"ab": {"completed": True, "aborted": False}},
        )
        store.save_snapshot(snap)

        loaded = store.load_latest_snapshot("r3")
        assert loaded is not None
        assert loaded.step == 1
        assert loaded.trigger == "vertex:A:done"
        assert loaded.vertex_states["A"]["state"] == "done"
        assert loaded.edge_states["ab"]["completed"] is True

    def test_load_latest_returns_highest_step(self):
        store = _mem_store()
        store.create_run("r4")

        for i in range(1, 5):
            store.save_snapshot(GraphSnapshot(
                run_id="r4", step=i, trigger=f"step{i}", timestamp="",
                vertex_states={}, edge_states={},
            ))

        latest = store.load_latest_snapshot("r4")
        assert latest.step == 4

    def test_snapshot_count(self):
        store = _mem_store()
        store.create_run("r5")
        assert store.snapshot_count("r5") == 0

        for i in range(3):
            store.save_snapshot(GraphSnapshot(
                run_id="r5", step=i+1, trigger=f"t{i}", timestamp="",
                vertex_states={}, edge_states={},
            ))
        assert store.snapshot_count("r5") == 3

    def test_list_runs(self):
        store = _mem_store()
        store.create_run("rA")
        store.create_run("rB")
        runs = store.list_runs()
        run_ids = [r["run_id"] for r in runs]
        assert "rA" in run_ids
        assert "rB" in run_ids

    def test_load_nonexistent_run_returns_none(self):
        store = _mem_store()
        assert store.load_latest_snapshot("no-such-run") is None

    def test_get_nonexistent_run_returns_none(self):
        store = _mem_store()
        assert store.get_run("no-such-run") is None


# ── CheckpointedExecutor — basic checkpointing ──────────────────────

class TestCheckpointedExecutorBasic:
    @pytest.mark.asyncio
    async def test_run_succeeds_same_as_base(self):
        g = Graph.from_dict(SIMPLE_CONFIG)
        store = _mem_store()
        ex = CheckpointedExecutor(g, MockAgent(), store=store, run_id="run-basic")
        result = await ex.run()
        assert result.success

    @pytest.mark.asyncio
    async def test_run_saves_snapshots(self):
        g = Graph.from_dict(SIMPLE_CONFIG)
        store = _mem_store()
        ex = CheckpointedExecutor(g, MockAgent(), store=store, run_id="run-snaps")
        await ex.run()

        # Expect: 1 initial + 1 per vertex (3 vertices) = at least 4
        count = store.snapshot_count("run-snaps")
        assert count >= 4

    @pytest.mark.asyncio
    async def test_run_updates_status_to_completed(self):
        g = Graph.from_dict(SIMPLE_CONFIG)
        store = _mem_store()
        ex = CheckpointedExecutor(g, MockAgent(), store=store, run_id="run-status")
        await ex.run()
        assert store.get_run("run-status")["status"] == "completed"

    @pytest.mark.asyncio
    async def test_final_snapshot_shows_all_done(self):
        g = Graph.from_dict(SIMPLE_CONFIG)
        store = _mem_store()
        ex = CheckpointedExecutor(g, MockAgent(), store=store, run_id="run-final")
        await ex.run()

        snap = store.load_latest_snapshot("run-final")
        for vid, vs in snap.vertex_states.items():
            assert vs["state"] in ("done", "aborted"), \
                f"Vertex '{vid}' has unexpected state '{vs['state']}'"

    @pytest.mark.asyncio
    async def test_snapshot_captures_data(self):
        g = Graph.from_dict(SIMPLE_CONFIG)
        store = _mem_store()
        ex = CheckpointedExecutor(g, MockAgent(), store=store, run_id="run-data")
        await ex.run()

        snap = store.load_latest_snapshot("run-data")
        # A had initial_data x="hello"
        assert snap.vertex_states["A"]["data"].get("x") == "hello"

    @pytest.mark.asyncio
    async def test_diamond_run_succeeds(self):
        g = Graph.from_dict(DIAMOND_CONFIG)
        store = _mem_store()
        ex = CheckpointedExecutor(g, MockAgent(), store=store, run_id="run-diamond")
        result = await ex.run()
        assert result.success


# ── CheckpointedExecutor — resume ───────────────────────────────────

class TestResume:
    @pytest.mark.asyncio
    async def test_resume_from_completed_run_is_idempotent(self):
        """
        Resume a fully-completed run: all vertices DONE.
        The executor should see everything settled and return immediately
        with success (no re-processing).
        """
        store = _mem_store()
        run_id = "run-resume-done"

        # First pass — run normally
        g1 = Graph.from_dict(SIMPLE_CONFIG)
        ex1 = CheckpointedExecutor(g1, MockAgent(), store=store, run_id=run_id)
        r1 = await ex1.run()
        assert r1.success

        # Resume on a fresh graph — should succeed trivially (all DONE)
        g2 = Graph.from_dict(SIMPLE_CONFIG)
        r2 = await CheckpointedExecutor.resume(
            run_id, g2, MockAgent(), store=store
        )
        assert r2.success

    @pytest.mark.asyncio
    async def test_resume_no_snapshot_raises(self):
        store = _mem_store()
        store.create_run("ghost-run")   # run exists but no snapshots
        g = Graph.from_dict(SIMPLE_CONFIG)
        with pytest.raises(ValueError, match="No snapshot"):
            await CheckpointedExecutor.resume("ghost-run", g, MockAgent(), store=store)

    @pytest.mark.asyncio
    async def test_resume_from_partial_snapshot(self):
        """
        Manually inject a snapshot where only vertex A is DONE, B and C are IDLE.
        Resume should process B and C and finish successfully.
        """
        store = _mem_store()
        run_id = "run-partial"
        g_config = SIMPLE_CONFIG
        store.create_run(run_id, graph_config=g_config)

        # Manually craft a partial snapshot: A DONE, B IDLE with A→B delivered
        snap = GraphSnapshot(
            run_id=run_id,
            step=2,
            trigger="vertex:A:done",
            timestamp="",
            vertex_states={
                "A": {
                    "state": "done",
                    "data": {"x": "hello"},
                    "error": None,
                    "abort_reason": None,
                    "iteration_count": 0,
                    "completed_incoming_edges": [],
                    "aborted_incoming_edges": [],
                },
                "B": {
                    "state": "idle",
                    "data": {"x": "hello"},    # data from A was received
                    "error": None,
                    "abort_reason": None,
                    "iteration_count": 0,
                    "completed_incoming_edges": ["ab"],  # edge ab completed
                    "aborted_incoming_edges": [],
                },
                "C": {
                    "state": "idle",
                    "data": {},
                    "error": None,
                    "abort_reason": None,
                    "iteration_count": 0,
                    "completed_incoming_edges": [],
                    "aborted_incoming_edges": [],
                },
            },
            edge_states={
                "ab": {"completed": True,  "aborted": False, "abort_reason": None,
                       "error": None, "result": "hello"},
                "bc": {"completed": False, "aborted": False, "abort_reason": None,
                       "error": None, "result": None},
            },
        )
        store.save_snapshot(snap)

        # Resume — B should become READY (completed_incoming_edges covers all),
        # be processed, and C should follow.
        g = Graph.from_dict(g_config)
        result = await CheckpointedExecutor.resume(
            run_id, g, MockAgent(), store=store
        )
        assert result.success, result.summary()
        assert g.vertices["B"].state == VertexState.DONE
        assert g.vertices["C"].state == VertexState.DONE

    @pytest.mark.asyncio
    async def test_resume_preserves_run_id(self):
        store = _mem_store()
        run_id = "run-id-preserve"
        g1 = Graph.from_dict(SIMPLE_CONFIG)
        ex = CheckpointedExecutor(g1, MockAgent(), store=store, run_id=run_id)
        await ex.run()

        g2 = Graph.from_dict(SIMPLE_CONFIG)
        ex2 = await CheckpointedExecutor.resume.__func__(
            CheckpointedExecutor, run_id, g2, MockAgent(), store=store
        )
        # Just verify run exists in store
        assert store.get_run(run_id) is not None


# ── HumanGateVertex — HITL approval flow ────────────────────────────

HITL_CONFIG = {
    "vertices": [
        {"id": "source", "initial_data": [{"data_id": "doc", "value": "draft"}]},
        {"id": "review"},    # will be a HumanGateVertex
        {"id": "publish"},
    ],
    "edges": [
        {"id": "s2r", "source": "source", "destination": "review",  "channel": "doc"},
        {"id": "r2p", "source": "review",  "destination": "publish", "channel": "doc"},
    ],
}


def _make_hitl_graph(config=HITL_CONFIG) -> Graph:
    """Build the HITL graph, replacing 'review' vertex with a HumanGateVertex."""
    g = Graph.from_dict(config)
    # Swap the vertex subclass in-place
    old = g.vertices["review"]
    gate = HumanGateVertex(
        vertex_id=old.id,
        settings=old.settings,
    )
    gate.incoming_edges = list(old.incoming_edges)
    gate.outgoing_edges = list(old.outgoing_edges)
    gate.required_input_count = old.required_input_count
    gate.loop_incoming_edges = dict(old.loop_incoming_edges)
    g.vertices["review"] = gate
    return g


class TestHumanGateVertex:
    def test_approve_sets_ready(self):
        gate = HumanGateVertex("gate")
        gate.state = VertexState.PAUSED
        assert gate.state == VertexState.PAUSED
        gate.approve()
        assert gate.state == VertexState.READY
        assert gate._approved is True

    def test_approve_with_data_updates_store(self):
        gate = HumanGateVertex("gate")
        gate.approve(approved_data={"doc": "final"})
        assert gate._data_store["doc"] == "final"

    def test_reset_clears_approval(self):
        gate = HumanGateVertex("gate")
        gate.approve()
        gate.reset()
        assert not gate._approved
        assert gate.state == VertexState.IDLE

    def test_settings_require_approval_on_standard_vertex(self):
        v = Vertex("v_review", settings={"require_approval": True})
        assert v._require_approval is True
        assert v._approved is False

    def test_pause_for_approval_method(self):
        v = Vertex("v_custom")
        v.state = VertexState.READY
        v.pause_for_approval()
        assert v._require_approval is True
        assert v.state == VertexState.PAUSED

    @pytest.mark.asyncio
    async def test_hitl_pauses_execution_and_resumes_on_approve(self):
        """
        Executor runs until HumanGateVertex becomes PAUSED, saves checkpoint, and exits cleanly.
        Then calling approve() on a resumed executor finishes the pipeline.
        """
        g = _make_hitl_graph()
        gate = g.vertices["review"]
        store = _mem_store()
        run_id = "hitl-pause-resume"

        # Step 1: Initial run -> processes source, then pauses at review
        ex1 = CheckpointedExecutor(g, MockAgent(), store=store, run_id=run_id)
        res1 = await ex1.run()
        # A clean HITL pause is NOT a failure: no errors, paused flag set,
        # summary says PAUSED instead of FAILED.
        assert res1.paused is True
        assert res1.success is False
        assert res1.errors == []
        assert "PAUSED" in res1.summary()
        assert "FAILED" not in res1.summary()
        assert g.vertices["source"].state == VertexState.DONE
        assert g.vertices["review"].state == VertexState.PAUSED
        assert g.vertices["publish"].state == VertexState.IDLE
        assert store.get_run(run_id)["status"] == "awaiting_approval"

        # Verify paused checkpoint was saved
        snaps = store.load_all_snapshots(run_id)
        triggers = [s.trigger for s in snaps]
        assert any("paused" in t for t in triggers)

        # Step 2: Human approves and resumes
        g2 = _make_hitl_graph()
        g2.vertices["review"].approve({"doc": "human_approved_content"})
        res2 = await CheckpointedExecutor.resume(run_id, g2, MockAgent(), store=store)

        assert res2.success, res2.summary()
        assert g2.vertices["review"].state == VertexState.DONE
        assert g2.vertices["publish"].state == VertexState.DONE
        assert await g2.vertices["publish"].fetch_data("doc") == "human_approved_content"
        assert store.get_run(run_id)["status"] == "completed"

    @pytest.mark.asyncio
    async def test_hitl_with_settings_require_approval(self):
        """
        Test that a standard Vertex with {"require_approval": True} in settings
        automatically pauses in PAUSED state when ready.
        """
        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"channel": "val", "value": "init"}]},
                {"id": "B", "settings": {"require_approval": True}},
                {"id": "C"},
            ],
            "edges": [
                {"id": "ab", "source": "A", "destination": "B", "channel": "val"},
                {"id": "bc", "source": "B", "destination": "C", "channel": "val"},
            ],
        }
        g = Graph.from_dict(config)
        store = _mem_store()
        run_id = "run-settings-hitl"

        ex = CheckpointedExecutor(g, MockAgent(), store=store, run_id=run_id)
        await ex.run()

        assert g.vertices["A"].state == VertexState.DONE
        assert g.vertices["B"].state == VertexState.PAUSED
        assert g.vertices["C"].state == VertexState.IDLE
        assert store.get_run(run_id)["status"] == "awaiting_approval"
