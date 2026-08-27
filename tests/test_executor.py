"""Tests for framework.executor."""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import Graph, Executor, MockAgent
from framework.vertex import VertexState


# ── linear execution ─────────────────────────────────────────────
class TestLinearExecution:
    @pytest.mark.asyncio
    async def test_linear_succeeds(self, linear_config):
        g = Graph.from_dict(linear_config)
        result = await Executor(g, MockAgent(), timeout=10).run()

        assert result.success
        assert len(result.errors) == 0
        assert g.vertices["A"].state == VertexState.DONE
        assert g.vertices["B"].state == VertexState.DONE
        assert g.vertices["C"].state == VertexState.DONE

    @pytest.mark.asyncio
    async def test_edge_results_populated(self, linear_config):
        g = Graph.from_dict(linear_config)
        result = await Executor(g, MockAgent(), timeout=10).run()

        assert "e1" in result.edge_results
        assert "e2" in result.edge_results

    @pytest.mark.asyncio
    async def test_data_flows_through(self, linear_config):
        g = Graph.from_dict(linear_config)
        agent = MockAgent(response_fn=lambda d, p, m, s: f"[{d}]")
        result = await Executor(g, agent, timeout=10).run()

        # A had "hello", e1 wraps it → "[hello]", e2 wraps that → "[[hello]]"
        c_data = result.vertex_results["C"]["data"]
        assert any("[[hello]]" in str(v) for v in c_data.values())


# ── diamond execution ────────────────────────────────────────────
class TestDiamondExecution:
    @pytest.mark.asyncio
    async def test_diamond_succeeds(self, diamond_config):
        g = Graph.from_dict(diamond_config)
        result = await Executor(g, MockAgent(), timeout=10).run()
        assert result.success

    @pytest.mark.asyncio
    async def test_fan_in_vertex_gets_both_inputs(self, diamond_config):
        g = Graph.from_dict(diamond_config)
        result = await Executor(g, MockAgent(), timeout=10).run()

        d_data = result.vertex_results["D"]["data"]
        assert len(d_data) >= 2  # received from both B and C


# ── concurrency ──────────────────────────────────────────────────
class TestConcurrency:
    @pytest.mark.asyncio
    async def test_max_concurrency_1(self, diamond_config):
        """Serial execution (concurrency=1) should still succeed."""
        g = Graph.from_dict(diamond_config)
        result = await Executor(g, MockAgent(), max_concurrency=1, timeout=10).run()
        assert result.success

    @pytest.mark.asyncio
    async def test_wide_fanout(self):
        """10-way fanout from a single source."""
        config = {
            "vertices": [
                {"id": "src", "initial_data": [{"data_id": "d", "tags": [str(i)], "value": f"v{i}"} for i in range(10)]},
            ] + [{"id": f"dst{i}"} for i in range(10)],
            "edges": [
                {"id": f"e{i}", "source": "src", "destination": f"dst{i}",
                 "data_id": "d", "tags": [str(i)], "prompt": "go", "model": "m"}
                for i in range(10)
            ],
        }
        g = Graph.from_dict(config)
        result = await Executor(g, MockAgent(), max_concurrency=5, timeout=10).run()
        assert result.success
        assert len(result.edge_results) == 10


# ── timeout ──────────────────────────────────────────────────────
class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_fires(self):
        """An agent that sleeps should trigger timeout."""
        async def slow_process(data, prompt, model, settings):
            await asyncio.sleep(10)
            return data

        class SlowAgent(MockAgent):
            async def process(self, data, prompt, model, settings=None):
                return await slow_process(data, prompt, model, settings)

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "d", "value": "x"}]},
                {"id": "B"},
            ],
            "edges": [
                {"id": "e", "source": "A", "destination": "B",
                 "data_id": "d", "prompt": "", "model": "m"},
            ],
        }
        g = Graph.from_dict(config)
        result = await Executor(g, SlowAgent(), timeout=0.5).run()

        assert not result.success
        assert any("timed out" in e.lower() for e in result.errors)


# ── error handling ───────────────────────────────────────────────
class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_agent_error_recorded(self):
        def fail_agent(d, p, m, s):
            raise RuntimeError("boom")

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "d", "value": "x"}]},
                {"id": "B"},
            ],
            "edges": [
                {"id": "e", "source": "A", "destination": "B",
                 "data_id": "d", "prompt": "", "model": "m"},
            ],
        }
        g = Graph.from_dict(config)
        result = await Executor(g, MockAgent(response_fn=fail_agent), timeout=10).run()

        assert not result.success
        assert any("boom" in e for e in result.errors)
        assert g.vertices["A"].state == VertexState.ERROR


# ── result object ────────────────────────────────────────────────
class TestExecutionResult:
    @pytest.mark.asyncio
    async def test_summary_contains_info(self, linear_config):
        g = Graph.from_dict(linear_config)
        result = await Executor(g, MockAgent(), timeout=10).run()
        s = result.summary()
        assert "SUCCESS" in s
        assert "e1" in s
        assert "e2" in s

    @pytest.mark.asyncio
    async def test_execution_time_positive(self, linear_config):
        g = Graph.from_dict(linear_config)
        result = await Executor(g, MockAgent(), timeout=10).run()
        assert result.execution_time > 0
