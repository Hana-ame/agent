"""Regression tests for the fan-in settlement rewrite (review §2 / E1, E2, H8, H9).

These reproduce the executor defects that the previous ``_fan_in_completed`` /
``_fan_in_counts`` / ``_fan_in_failures`` bookkeeping caused:

* E1 — a late non-participating edge re-opened a settled vertex and re-ran its
  siblings forever (livelock, data loss).
* E2 — one failed edge was added to the completion set and never ran again, and
  its healthy sibling was never dispatched.
* H8 — the terminal shutdown cancelled an in-flight sibling and still reported
  success.
* H9 — an ``except TypeError`` fallback re-ran the edge body on a body TypeError.
"""

from __future__ import annotations

import asyncio

import pytest

from framework.edge_v4 import CodeEdgeV4, EdgeResultV4, EdgeV4
from framework.executor_v4 import ExecutorV4
from framework.graph_manager_v4 import SessionGraphManagerV4
from framework.graph_v4 import GraphV4
from framework.vertex_v4 import VertexStateV4, VertexStoreV4


def _build_store(vertices: dict) -> VertexStoreV4:
    store = VertexStoreV4(":memory:")
    for name, (content, state) in vertices.items():
        store.save_vertex("s", name, content=content, state=state)
    return store


class TestMixedStrategyFanIn:
    @pytest.mark.asyncio
    async def test_mixed_merge_and_overwrite_settles_once(self):
        """E1: overwrite + json_merge + json_merge must settle exactly once."""
        store = _build_store({
            "a": ('{"e1": 1}', VertexStateV4.DATA_READY.value),
            "b": ('{"e2": 2}', VertexStateV4.DATA_READY.value),
            "c": ('{"e3": 3}', VertexStateV4.DATA_READY.value),
            "out": ("{}", VertexStateV4.TODO.value),
        })
        graph = GraphV4("s")
        graph.add_vertex({"name": "a", "content": '{"e1": 1}', "state": "data ready"})
        graph.add_vertex({"name": "b", "content": '{"e2": 2}', "state": "data ready"})
        graph.add_vertex({"name": "c", "content": '{"e3": 3}', "state": "data ready"})
        graph.add_vertex({"name": "out", "content": "{}", "state": "todo", "attributes": ["end"]})

        runs: list[str] = []

        def make(edge_id: str, src: str):
            def fn(content, settings, staging=None):
                runs.append(edge_id)
                return content
            return fn

        graph.add_edge(CodeEdgeV4("e1", "a", "out", script=make("e1", "a"),
                                  settings={"merge_strategy": "json_merge"}))
        graph.add_edge(CodeEdgeV4("e2", "b", "out", script=make("e2", "b"),
                                  settings={"merge_strategy": "json_merge"}))
        graph.add_edge(CodeEdgeV4("e3", "c", "out", script=make("e3", "c")))  # overwrite

        executor = ExecutorV4(graph=graph, store=store, max_concurrency=4, timeout=5.0)
        res = await executor.run()

        assert res.success is True, res.errors
        assert sorted(runs) == ["e1", "e2", "e3"], f"edges re-ran: {runs}"
        out = store.get_vertex("s", "out")
        assert out.state == VertexStateV4.DATA_READY.value
        # Every participant settled exactly once; the final content is whatever
        # the last writer produced (e3 overwrites after the json_merge pair).
        assert out.processed_count == 1

    @pytest.mark.asyncio
    async def test_barrier_waits_for_all_participants(self):
        """A vertex with >1 incoming edge must not settle early."""
        store = _build_store({
            "a": ("A", VertexStateV4.DATA_READY.value),
            "b": ("B", VertexStateV4.DATA_READY.value),
            "out": ("", VertexStateV4.TODO.value),
        })
        graph = GraphV4("s")
        graph.add_vertex({"name": "a", "content": "A", "state": "data ready"})
        graph.add_vertex({"name": "b", "content": "B", "state": "data ready"})
        graph.add_vertex({"name": "out", "content": "", "state": "todo", "attributes": ["end"]})
        graph.add_edge(CodeEdgeV4("e1", "a", "out", script=lambda c, s, st: c,
                                  settings={"merge_strategy": "list_append"}))
        graph.add_edge(CodeEdgeV4("e2", "b", "out", script=lambda c, s, st: c,
                                  settings={"merge_strategy": "list_append"}))

        executor = ExecutorV4(graph=graph, store=store, max_concurrency=2, timeout=5.0)
        res = await executor.run()
        assert res.success is True, res.errors
        assert set(res.completed_edges) == {"e1", "e2"}


class TestFailureDoesNotBlacklist:
    @pytest.mark.asyncio
    async def test_failed_sibling_still_runs_and_edge_not_blacklisted(self):
        """E2: a failing edge must not prevent its sibling from running."""
        store = _build_store({
            "a": ("A", VertexStateV4.DATA_READY.value),
            "b": ("B", VertexStateV4.DATA_READY.value),
            "c": ("", VertexStateV4.TODO.value),
        })
        graph = GraphV4("s")
        graph.add_vertex({"name": "a", "content": "A", "state": "data ready"})
        graph.add_vertex({"name": "b", "content": "B", "state": "data ready"})
        graph.add_vertex({"name": "c", "content": "", "state": "todo", "attributes": ["end"]})

        ran: list[str] = []

        def failing(content, settings, staging=None):
            ran.append("e_fail")
            raise RuntimeError("script blew up")

        def healthy(content, settings, staging=None):
            ran.append("e_ok")
            return content

        graph.add_edge(CodeEdgeV4("e_fail", "a", "c", script=failing))
        graph.add_edge(CodeEdgeV4("e_ok", "b", "c", script=healthy))

        executor = ExecutorV4(graph=graph, store=store, max_concurrency=2, timeout=5.0)
        res = await executor.run()

        assert "e_ok" in ran, "healthy sibling was never dispatched"
        assert store.get_vertex("s", "c").state == VertexStateV4.REJECT.value
        # The failed edge is not permanently excluded from scheduling.
        settlement = executor._settlements.get("c")
        assert settlement is not None
        assert "e_fail" in settlement.failed

    @pytest.mark.asyncio
    async def test_settlement_cycle_resets_after_reentry(self):
        """A re-opened vertex starts a fresh settlement cycle."""
        store = _build_store({
            "a": ("A", VertexStateV4.DATA_READY.value),
            "b": ("B", VertexStateV4.DATA_READY.value),
            "c": ("", VertexStateV4.TODO.value),
        })
        graph = GraphV4("s")
        graph.add_vertex({"name": "a", "content": "A", "state": "data ready"})
        graph.add_vertex({"name": "b", "content": "B", "state": "data ready"})
        graph.add_vertex({"name": "c", "content": "", "state": "todo", "attributes": ["end"]})
        graph.add_edge(CodeEdgeV4("e1", "a", "c", script=lambda c, s, st: c,
                                  settings={"merge_strategy": "json_merge"}))
        graph.add_edge(CodeEdgeV4("e2", "b", "c", script=lambda c, s, st: c,
                                  settings={"merge_strategy": "json_merge"}))

        executor = ExecutorV4(graph=graph, store=store, max_concurrency=2, timeout=5.0)
        await executor.run()
        assert executor._settlements["c"].settled_ids == {"e1", "e2"}

        # Re-open the vertex (as /reenter does) and confirm edges are eligible again.
        store.update_vertex_state("s", "c", VertexStateV4.TODO.value)
        eligible = [e.id for _r, _p, _t, e in executor._get_eligible_edges()]
        assert set(eligible) == {"e1", "e2"}


class TestNoPrematureTermination:
    @pytest.mark.asyncio
    async def test_slow_sibling_is_not_cancelled(self):
        """H8: a slow in-flight edge must finish before success is declared."""
        store = _build_store({
            "a": ("A", VertexStateV4.DATA_READY.value),
            "b": ("B", VertexStateV4.DATA_READY.value),
            "c": ("", VertexStateV4.TODO.value),
        })
        graph = GraphV4("s")
        graph.add_vertex({"name": "a", "content": "A", "state": "data ready"})
        graph.add_vertex({"name": "b", "content": "B", "state": "data ready"})
        graph.add_vertex({"name": "c", "content": "", "state": "todo", "attributes": ["end"]})

        finished: list[str] = []

        async def slow(content, settings, staging=None):
            await asyncio.sleep(0.2)
            finished.append("e_slow")
            return content

        graph.add_edge(CodeEdgeV4("e_fast", "a", "c", script=lambda c, s, st: c))
        graph.add_edge(CodeEdgeV4("e_slow", "b", "c", script=slow))

        executor = ExecutorV4(graph=graph, store=store, max_concurrency=2, timeout=10.0)
        res = await executor.run()

        assert "e_slow" in finished, "slow sibling was cancelled before running"
        assert res.success is True
        assert set(res.completed_edges) == {"e_fast", "e_slow"}


class TestBodyTypeErrorDoesNotRerun:
    @pytest.mark.asyncio
    async def test_body_type_error_runs_edge_once(self):
        """H9: a TypeError raised inside the edge body must not re-run the edge."""
        calls: list[int] = []

        class RaisingEdge(EdgeV4):
            async def run(self, session_id, store, agent=None, auto_transition=True, **kwargs):
                calls.append(1)
                raise TypeError("body bug")

        store = _build_store({
            "a": ("A", VertexStateV4.DATA_READY.value),
            "b": ("", VertexStateV4.TODO.value),
        })
        graph = GraphV4("s")
        graph.add_vertex({"name": "a", "content": "A", "state": "data ready"})
        graph.add_vertex({"name": "b", "content": "", "state": "todo", "attributes": ["end"]})
        graph.add_edge(RaisingEdge("e1", "a", "b"))

        executor = ExecutorV4(graph=graph, store=store, max_concurrency=1, timeout=3.0)
        res = await executor.run()

        assert len(calls) == 1, f"edge body ran {len(calls)} times"
        assert res.success is False


class TestToolEdgeRefused:
    @pytest.mark.asyncio
    async def test_tool_edge_is_not_fabricated(self):
        """H1: ExecutorV4 must not pass upstream content through a tool edge."""
        from framework.edges.tool import ToolEdgeV4

        store = _build_store({
            "a": ("SECRET_UPSTREAM", VertexStateV4.DATA_READY.value),
            "b": ("", VertexStateV4.TODO.value),
        })
        graph = GraphV4("s")
        graph.add_vertex({"name": "a", "content": "SECRET_UPSTREAM", "state": "data ready"})
        graph.add_vertex({"name": "b", "content": "", "state": "todo", "attributes": ["end"]})
        graph.add_edge(ToolEdgeV4("e_tool", "a", "b", tool_name="bash",
                                  arguments={"command": "ls"}))

        executor = ExecutorV4(graph=graph, store=store, max_concurrency=1, timeout=3.0)
        res = await executor.run()

        assert res.success is False
        assert any("HttpHarnessExecutorV4" in e for e in res.errors)
        b = store.get_vertex("s", "b")
        assert b.content != "SECRET_UPSTREAM", "tool result was fabricated from upstream content"
        assert b.state == VertexStateV4.REJECT.value


class TestSubgraphFailurePropagates:
    @pytest.mark.asyncio
    async def test_child_failure_is_not_reported_as_parent_success(self, tmp_path):
        """H2: a failing child graph must fail the bridge edge, not the parent silently succeed."""
        import json as _json

        from framework.sse_executor_v4 import SSEExecutorV4

        child_dir = tmp_path / "child"
        child_dir.mkdir()
        script = child_dir / "boom.py"
        script.write_text("def process(data, settings=None, staging=None):\n    raise RuntimeError('child boom')\n")
        (child_dir / "c_in.json").write_text(_json.dumps(
            {"name": "c_start", "content": "", "attributes": ["start"], "state": "idle"}))
        (child_dir / "c_out.json").write_text(_json.dumps(
            {"name": "c_end", "content": "", "attributes": ["end"], "state": "todo"}))
        (child_dir / "c_edge.json").write_text(_json.dumps(
            {"id": "e_child", "type": "code", "input_vertex": "c_start",
             "output_vertex": "c_end", "script": f"{script}:process"}))
        (child_dir / "graph.json").write_text(_json.dumps({
            "version": "4.0", "metadata": {"name": "child"},
            "vertices": ["c_in.json", "c_out.json"], "edges": ["c_edge.json"],
        }))

        parent = tmp_path / "parent.json"
        parent.write_text(_json.dumps({
            "version": "4.0",
            "metadata": {"name": "parent", "base_dir": str(tmp_path)},
            "vertices": [
                {"name": "p_in", "content": "hello", "state": "data ready", "attributes": ["start"]},
                {"name": "p_box", "content": _json.dumps({"subgraph_manifest": "child/graph.json"}),
                 "attributes": ["subgraph"], "state": "todo"},
                {"name": "p_out", "content": "", "state": "todo", "attributes": ["end"]},
            ],
            "edges": [
                {"id": "e_in", "type": "code", "input_vertex": "p_in", "output_vertex": "p_box"},
                {"id": "e_out", "type": "code", "input_vertex": "p_box", "output_vertex": "p_out"},
            ],
        }))

        store = VertexStoreV4(":memory:")
        manager = SessionGraphManagerV4(store)
        executor = SSEExecutorV4(manager=manager, store=store, default_manifest=parent)
        result = await executor.execute_harness_call()
        info = _json.loads(result["function"]["arguments"])["info"]

        assert info["success"] is False, "child failure was reported as parent success"
        assert info["vertex_contents"]["p_out"] != "hello"
