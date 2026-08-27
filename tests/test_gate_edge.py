"""Tests for Edge and conditional dynamic routing / abort handling."""

import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import Vertex, VertexState, Edge, Graph, Executor, MockAgent
from framework.vertex import EdgeSignal


# ── Edge Condition Evaluation ──────────────────────────────
class TestEdgeCondition:
    def test_default_truthiness(self):
        gate = Edge("g1", "v1", "v2")
        assert gate.evaluate_condition("hello", {}) is True
        assert gate.evaluate_condition("", {}) is True
        assert gate.evaluate_condition(1, {}) is True
        assert gate.evaluate_condition(0, {}) is True
        assert gate.evaluate_condition([], {}) is True
        assert gate.evaluate_condition([1], {}) is True

    def test_threshold_operators(self):
        gate = Edge("g1", "v1", "v2", settings={"threshold": 80, "operator": ">="})
        assert gate.evaluate_condition(80, gate.settings) is True
        assert gate.evaluate_condition(95, gate.settings) is True
        assert gate.evaluate_condition(79, gate.settings) is False

        gate_lt = Edge("g2", "v1", "v2", settings={"threshold": 50, "operator": "<"})
        assert gate_lt.evaluate_condition(49, gate_lt.settings) is True
        assert gate_lt.evaluate_condition(50, gate_lt.settings) is False

        gate_eq = Edge("g3", "v1", "v2", settings={"threshold": "apple", "operator": "=="})
        assert gate_eq.evaluate_condition("apple", gate_eq.settings) is True
        assert gate_eq.evaluate_condition("banana", gate_eq.settings) is False

        gate_contains = Edge("g4", "v1", "v2", settings={"threshold": "draw", "operator": "contains"})
        assert gate_contains.evaluate_condition("please draw a cat", gate_contains.settings) is True
        assert gate_contains.evaluate_condition("write a poem", gate_contains.settings) is False

    def test_threshold_with_dict_field(self):
        gate = Edge("g1", "v1", "v2", settings={"field": "score", "threshold": 60, "operator": ">="})
        assert gate.evaluate_condition({"score": 75, "name": "Alice"}, gate.settings) is True
        assert gate.evaluate_condition({"score": 50, "name": "Bob"}, gate.settings) is False
        assert gate.evaluate_condition({"other": 100}, gate.settings) is False

    def test_dictionary_match(self):
        gate = Edge("g1", "v1", "v2", settings={"match": {"intent": "image", "vip": True}})
        assert gate.evaluate_condition({"intent": "image", "vip": True, "prompt": "cat"}, gate.settings) is True
        assert gate.evaluate_condition({"intent": "image", "vip": False}, gate.settings) is False
        assert gate.evaluate_condition({"intent": "text", "vip": True}, gate.settings) is False

    def test_subclass_override(self):
        class CustomGate(Edge):
            def condition(self, data, settings):
                return isinstance(data, str) and data.startswith("ALLOW")

        gate = CustomGate("g1", "v1", "v2")
        assert gate.evaluate_condition("ALLOW: test", {}) is True
        assert gate.evaluate_condition("DENY: test", {}) is False


# ── Edge Execution Unit Tests ──────────────────────────────
class TestEdgeExecution:
    @pytest.mark.asyncio
    async def test_gate_edge_passes_data(self):
        v1 = Vertex("v1", initial_data=[{"data_id": "score", "value": 90}])
        v2 = Vertex("v2")
        v2.incoming_edges = ["g1"]

        gate = Edge("g1", "v1", "v2", channel="score", settings={"threshold": 80, "operator": ">="})
        agent = MockAgent()

        result = await gate.execute(v1, v2, agent)
        assert result == 90
        assert gate.completed is True
        assert gate.aborted is False
        assert v2.state == VertexState.READY
        assert await v2.fetch_data(channel="score") == 90

    @pytest.mark.asyncio
    async def test_gate_edge_aborts_on_condition_false(self):
        v1 = Vertex("v1", initial_data=[{"data_id": "score", "value": 50}])
        v2 = Vertex("v2")
        v2.incoming_edges = ["g1"]

        gate = Edge("g1", "v1", "v2", channel="score", settings={"threshold": 80, "operator": ">="})
        agent = MockAgent()

        result = await gate.execute(v1, v2, agent)
        assert result is None
        assert gate.completed is False
        assert gate.aborted is True
        assert "not satisfied" in gate.abort_reason
        assert v2.state == VertexState.ABORTED
        assert "g1" in v2.aborted_incoming_edges


# ── Diamond Dynamic Routing & Non-blocking Join ────────────────
class TestConditionalDiamondRouting:
    @pytest.mark.asyncio
    async def test_diamond_single_active_branch_settles_join_node(self):
        """
        Topology:
                 /-- [Gate: score >= 80] --> HighBranch -- [Edge] --\\
          Source                                                      --> Sink
                 \\-- [Gate: score < 80]  --> LowBranch  -- [Edge] --/
        """
        config = {
            "vertices": [
                {
                    "id": "Source",
                    "initial_data": [{"data_id": "score", "value": 95}],
                },
                {"id": "HighBranch"},
                {"id": "LowBranch"},
                {"id": "Sink"},
            ],
            "edges": [
                {
                    "id": "g_high",
                    "type": "gate",
                    "source": "Source",
                    "destination": "HighBranch",
                    "data_id": "score",
                    "settings": {"threshold": 80, "operator": ">="},
                },
                {
                    "id": "g_low",
                    "type": "gate",
                    "source": "Source",
                    "destination": "LowBranch",
                    "data_id": "score",
                    "settings": {"threshold": 80, "operator": "<"},
                },
                {
                    "id": "e_high",
                    "source": "HighBranch",
                    "destination": "Sink",
                    "data_id": "score",
                    "prompt": "high score",
                },
                {
                    "id": "e_low",
                    "source": "LowBranch",
                    "destination": "Sink",
                    "data_id": "score",
                    "prompt": "low score",
                },
            ],
        }

        g = Graph.from_dict(config)
        agent = MockAgent(response_fn=lambda d, p, m, s: f"PROCESSED:{p}:{d}")
        executor = Executor(g, agent, timeout=5)
        result = await executor.run()

        assert result.success
        assert len(result.errors) == 0

        # State verifications
        assert g.vertices["Source"].state == VertexState.DONE
        assert g.vertices["HighBranch"].state == VertexState.DONE
        assert g.vertices["LowBranch"].state == VertexState.ABORTED
        assert g.vertices["Sink"].state == VertexState.DONE

        # Edge states
        assert g.edges["g_high"].completed is True
        assert g.edges["g_low"].aborted is True
        assert g.edges["e_low"].aborted is True
        assert g.edges["e_high"].completed is True

        # Data in Sink
        sink_data = await g.vertices["Sink"].fetch_data(channel="score")
        assert "PROCESSED:high score:95" in sink_data

    @pytest.mark.asyncio
    async def test_all_branches_aborted_cascades_to_sink(self):
        """When all gates abort, sink aborts cleanly with 0 errors and no deadlocks."""
        config = {
            "vertices": [
                {
                    "id": "Source",
                    "initial_data": [{"data_id": "val", "value": 10}],
                },
                {"id": "BranchA"},
                {"id": "BranchB"},
                {"id": "Sink"},
            ],
            "edges": [
                {
                    "id": "g_a",
                    "type": "gate",
                    "source": "Source",
                    "destination": "BranchA",
                    "data_id": "val",
                    "settings": {"threshold": 100, "operator": ">"},
                },
                {
                    "id": "g_b",
                    "type": "gate",
                    "source": "Source",
                    "destination": "BranchB",
                    "data_id": "val",
                    "settings": {"threshold": 200, "operator": ">"},
                },
                {
                    "id": "e_a",
                    "source": "BranchA",
                    "destination": "Sink",
                    "data_id": "val",
                },
                {
                    "id": "e_b",
                    "source": "BranchB",
                    "destination": "Sink",
                    "data_id": "val",
                },
            ],
        }

        g = Graph.from_dict(config)
        result = await Executor(g, MockAgent(), timeout=5).run()

        assert result.success
        assert len(result.errors) == 0
        assert g.vertices["Source"].state == VertexState.DONE
        assert g.vertices["BranchA"].state == VertexState.ABORTED
        assert g.vertices["BranchB"].state == VertexState.ABORTED
        assert g.vertices["Sink"].state == VertexState.ABORTED

    @pytest.mark.asyncio
    async def test_deep_cascading_abort(self):
        """A -> Gate(False) -> B -> C -> D -> E all cascade abort."""
        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "d", "value": "no"}]},
                {"id": "B"},
                {"id": "C"},
                {"id": "D"},
                {"id": "E"},
            ],
            "edges": [
                {"id": "g1", "type": "gate", "source": "A", "destination": "B", "data_id": "d", "settings": {"threshold": "yes", "operator": "=="}},
                {"id": "e1", "source": "B", "destination": "C", "data_id": "d"},
                {"id": "e2", "source": "C", "destination": "D", "data_id": "d"},
                {"id": "e3", "source": "D", "destination": "E", "data_id": "d"},
            ],
        }

        g = Graph.from_dict(config)
        result = await Executor(g, MockAgent(), timeout=5).run()

        assert result.success
        assert g.vertices["A"].state == VertexState.DONE
        assert g.vertices["B"].state == VertexState.ABORTED
        assert g.vertices["C"].state == VertexState.ABORTED
        assert g.vertices["D"].state == VertexState.ABORTED
        assert g.vertices["E"].state == VertexState.ABORTED
