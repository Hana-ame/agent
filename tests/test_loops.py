"""Tests for stateful loop (cyclic graph) support.

Covers:
- Graph.validate() accepting guarded back-edges
- Graph.validate() rejecting unguarded back-edges
- Simple A -> B -> A self-correction loop
- Loop iteration counting and max_iterations enforcement
- Guard-terminated loops (condition fails, loop exits cleanly)
- Multi-vertex loop: A -> B -> C -> A
- Loop with data accumulation across iterations
"""

import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import Graph, Executor, MockAgent, Vertex, Edge
from framework.vertex import VertexState


# ── Graph validation ────────────────────────────────────────────────

class TestLoopValidation:
    def test_unguarded_cycle_rejected(self):
        """A back-edge without max_iterations must still raise ValueError."""
        config = {
            "vertices": [{"id": "A"}, {"id": "B"}],
            "edges": [
                {"id": "e1", "source": "A", "destination": "B", "data_id": "x"},
                {"id": "e2", "source": "B", "destination": "A", "data_id": "x"},
            ],
        }
        with pytest.raises(ValueError, match="unguarded cycle"):
            Graph.from_dict(config)

    def test_guarded_cycle_accepted(self):
        """A back-edge with max_iterations > 0 should NOT raise."""
        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "x", "value": 0}]},
                {"id": "B"},
            ],
            "edges": [
                {"id": "fwd", "source": "A", "destination": "B", "channel": "x"},
                {"id": "back", "source": "B", "destination": "A",
                 "channel": "x", "max_iterations": 3},
            ],
        }
        g = Graph.from_dict(config)
        assert "back" in g.vertices["A"].loop_incoming_edges
        assert g.vertices["A"].loop_incoming_edges["back"] == 3

    def test_guarded_cycle_max_iterations_zero_rejected(self):
        """max_iterations=0 is treated as 'unguarded' and must be rejected."""
        config = {
            "vertices": [{"id": "A"}, {"id": "B"}],
            "edges": [
                {"id": "fwd", "source": "A", "destination": "B", "channel": "x"},
                {"id": "back", "source": "B", "destination": "A",
                 "channel": "x", "max_iterations": 0},
            ],
        }
        with pytest.raises(ValueError):
            Graph.from_dict(config)

    def test_dag_still_validated_cleanly(self):
        """A pure DAG with no back-edges should validate without touching loop metadata."""
        config = {
            "vertices": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
            "edges": [
                {"id": "ab", "source": "A", "destination": "B", "channel": "x"},
                {"id": "bc", "source": "B", "destination": "C", "channel": "x"},
            ],
        }
        g = Graph.from_dict(config)
        for v in g.vertices.values():
            assert len(v.loop_incoming_edges) == 0


# ── Two-vertex loop: A <-> B ────────────────────────────────────────

class TestTwoVertexLoop:
    @pytest.mark.asyncio
    async def test_loop_runs_correct_number_of_iterations(self):
        """
        Topology:  A --fwd--> B --back(max=3)--> A

        A starts with counter=0.
        Each iteration, the mock agent increments the value by 1.
        After 3 loop-back re-entries, A should have value 3 and
        iteration_count == 3.
        """
        call_count = {"n": 0}

        def counting_agent(data, prompt, model, settings):
            call_count["n"] += 1
            return (data or 0) + 1

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "val", "value": 0}]},
                {"id": "B"},
            ],
            "edges": [
                {"id": "fwd", "source": "A", "destination": "B",
                 "channel": "val", "prompt": "increment", "model": "m"},
                {"id": "back", "source": "B", "destination": "A",
                 "channel": "val", "max_iterations": 3},
            ],
        }

        g = Graph.from_dict(config)
        agent = MockAgent(response_fn=counting_agent)
        result = await Executor(g, agent, timeout=10).run()

        assert result.success, result.summary()
        assert g.vertices["A"].state == VertexState.DONE
        assert g.vertices["B"].state == VertexState.DONE
        # A was re-entered exactly 3 times
        assert g.vertices["A"].iteration_count == 3
        # The fwd edge (prompt-driven) fires once per A-processing:
        # 1 initial + 3 re-entries = 4 total agent calls
        assert call_count["n"] == 4

    @pytest.mark.asyncio
    async def test_loop_final_value_is_correct(self):
        """Data accumulation: after N iterations, value == N."""
        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "v", "value": 0}]},
                {"id": "B"},
            ],
            "edges": [
                {"id": "fwd", "source": "A", "destination": "B",
                 "channel": "v", "prompt": "+1", "model": "m"},
                {"id": "back", "source": "B", "destination": "A",
                 "channel": "v", "max_iterations": 5},
            ],
        }
        g = Graph.from_dict(config)
        agent = MockAgent(response_fn=lambda d, p, m, s: (d or 0) + 1)
        result = await Executor(g, agent, timeout=10).run()

        assert result.success
        assert g.vertices["A"].iteration_count == 5
        final = await g.vertices["A"].fetch_data("v")
        assert final == 5

    @pytest.mark.asyncio
    async def test_single_iteration_loop(self):
        """max_iterations=1 means exactly one re-entry."""
        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "x", "value": "start"}]},
                {"id": "B"},
            ],
            "edges": [
                {"id": "fwd", "source": "A", "destination": "B", "channel": "x"},
                {"id": "back", "source": "B", "destination": "A",
                 "channel": "x", "max_iterations": 1},
            ],
        }
        g = Graph.from_dict(config)
        result = await Executor(g, MockAgent(), timeout=5).run()
        assert result.success
        assert g.vertices["A"].iteration_count == 1


# ── Three-vertex loop: A -> B -> C -> A ─────────────────────────────

class TestThreeVertexLoop:
    @pytest.mark.asyncio
    async def test_multi_vertex_loop_all_vertices_cycle(self):
        """
        Topology:  A --ab--> B --bc--> C --ca(max=2)--> A

        A processes, B processes, C processes, then C sends back to A.
        This repeats 2 more times.  All vertices end DONE.
        """
        visits: dict = {"A": 0, "B": 0, "C": 0}

        def tracking_agent(data, prompt, model, settings):
            visits[prompt] += 1
            return (data or 0) + 1

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "n", "value": 0}]},
                {"id": "B"},
                {"id": "C"},
            ],
            "edges": [
                {"id": "ab", "source": "A", "destination": "B",
                 "channel": "n", "prompt": "B", "model": "m"},
                {"id": "bc", "source": "B", "destination": "C",
                 "channel": "n", "prompt": "C", "model": "m"},
                {"id": "ca", "source": "C", "destination": "A",
                 "channel": "n", "max_iterations": 2},
            ],
        }

        g = Graph.from_dict(config)
        agent = MockAgent(response_fn=tracking_agent)
        result = await Executor(g, agent, timeout=10).run()

        assert result.success, result.summary()
        assert g.vertices["A"].state == VertexState.DONE
        assert g.vertices["B"].state == VertexState.DONE
        assert g.vertices["C"].state == VertexState.DONE

        # A re-entered 2 times → iteration_count = 2
        assert g.vertices["A"].iteration_count == 2
        # B and C each ran once per A-processing (initial + 2 re-entries = 3 total)
        assert visits["B"] == 3
        assert visits["C"] == 3

    @pytest.mark.asyncio
    async def test_loop_with_sink_receives_final_value(self):
        """
        Topology:  Source --fwd--> Worker --back(max=3)--> Source
                   Source --out--> Sink

        After 3 loop iterations, Source fires its out edge to Sink
        on each pass.  Sink sees the final value from the last iteration.
        """
        config = {
            "vertices": [
                {"id": "Src", "initial_data": [{"data_id": "v", "value": 0}]},
                {"id": "Worker"},
                {"id": "Sink"},
            ],
            "edges": [
                {"id": "fwd",  "source": "Src",    "destination": "Worker",
                 "channel": "v", "prompt": "+1", "model": "m"},
                {"id": "back", "source": "Worker", "destination": "Src",
                 "channel": "v", "max_iterations": 3},
                {"id": "out",  "source": "Src",    "destination": "Sink",
                 "channel": "v"},
            ],
        }
        g = Graph.from_dict(config)
        agent = MockAgent(response_fn=lambda d, p, m, s: (d or 0) + 1)
        result = await Executor(g, agent, timeout=10).run()

        assert result.success, result.summary()
        # Sink receives from Src on each iteration (3 re-entries + initial = 4 firings)
        # Final value written to Sink is from last firing (value = 3, the last +1)
        sink_val = await g.vertices["Sink"].fetch_data("v")
        assert sink_val == 3  # last Worker output (0+1, 1+1, 2+1 = 3rd iteration val)


# ── Guard-terminated loop ────────────────────────────────────────────

class TestGuardTerminatedLoop:
    @pytest.mark.asyncio
    async def test_guard_terminates_loop_early(self):
        """
        Topology:  A --fwd--> B --back(max=10, guard: val<3)--> A

        The loop-back edge has a guard that aborts when val >= 3.
        Even though max_iterations=10, the loop naturally ends when
        the guard condition fails.  Graph should settle cleanly.
        """
        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "v", "value": 0}]},
                {"id": "B"},
            ],
            "edges": [
                {"id": "fwd",  "source": "A", "destination": "B",
                 "channel": "v", "prompt": "+1", "model": "m"},
                {"id": "back", "source": "B", "destination": "A",
                 "channel": "v", "max_iterations": 10,
                 "settings": {"threshold": 3, "operator": "<"}},
            ],
        }
        g = Graph.from_dict(config)
        agent = MockAgent(response_fn=lambda d, p, m, s: (d or 0) + 1)
        result = await Executor(g, agent, timeout=10).run()

        assert result.success, result.summary()
        assert g.vertices["A"].state == VertexState.DONE
        # Guard fires when val >= 3, so only 2 re-entries (val=1 < 3 → loop,
        # val=2 < 3 → loop, val=3 is NOT < 3 → guard aborts back-edge)
        assert g.vertices["A"].iteration_count == 2


# ── Iteration count in results ───────────────────────────────────────

class TestLoopResultsMetadata:
    @pytest.mark.asyncio
    async def test_iteration_count_in_execution_result(self):
        """iteration_count is captured in ExecutionResult.vertex_results."""
        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"data_id": "x", "value": 0}]},
                {"id": "B"},
            ],
            "edges": [
                {"id": "fwd",  "source": "A", "destination": "B", "channel": "x"},
                {"id": "back", "source": "B", "destination": "A",
                 "channel": "x", "max_iterations": 4},
            ],
        }
        g = Graph.from_dict(config)
        result = await Executor(g, MockAgent(), timeout=10).run()

        assert result.success
        assert result.vertex_results["A"]["iterations"] == 4
