"""Tests for Pipeline business retry policy and Executor real-time event streaming."""

import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import Graph, Executor, MockAgent, Vertex, Edge, Pipeline, GraphEvent
from framework.vertex import VertexState


# =====================================================================
# Phase 1 Tests: Business Logic Retry (Self-Correction)
# =====================================================================

class TestPipelineBusinessRetry:
    @pytest.mark.asyncio
    async def test_retry_on_post_process_keyerror_with_prompt_feedback(self):
        """
        Scenario:
        1. First 2 LLM calls return dict missing 'target_key' -> post_process raises KeyError.
        2. Pipeline intercepts KeyError, updates prompt with [SYSTEM FEEDBACK:...].
        3. On 3rd attempt, LLM returns valid dict with 'target_key'.
        4. Overall execution succeeds.
        """
        attempts = []

        def flaking_agent(data, prompt, model, settings):
            attempts.append(prompt)
            if len(attempts) < 3:
                return {"wrong_key": "bad_data"}
            return {"target_key": "valid_data"}

        class RetryEdge(Edge):
            def post_process(self, result, settings):
                return result["target_key"]

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"channel": "in", "value": "input_val"}]},
                {"id": "B"},
            ],
            "edges": [
                {
                    "id": "e_retry",
                    "source": "A",
                    "destination": "B",
                    "channel": "in",
                    "settings": {
                        "prompt": "Please extract target JSON",
                        "retry_policy": {
                            "max_retries": 3,
                            "backoff_factor": 0.01,
                            "retry_on": ["KeyError"],
                        }
                    },
                }
            ],
        }

        g = Graph.from_dict(config)
        
        # Override the edge instantiated by Graph with our subclass
        old_edge = g.edges["e_retry"]
        e_retry = RetryEdge(
            edge_id=old_edge.id, source_id=old_edge.source_id,
            destination_id=old_edge.destination_id, channel=old_edge.channel,
            settings=old_edge.settings, concurrency_type=old_edge.concurrency_type,
            max_iterations=old_edge.max_iterations
        )
        g.edges["e_retry"] = e_retry

        agent = MockAgent(response_fn=flaking_agent)
        result = await Executor(g, agent).run()

        assert result.success, result.summary()
        assert len(attempts) == 3
        # Verify self-correction prompt feedback in subsequent attempts
        assert "Please extract target JSON" in attempts[0]
        assert "[SYSTEM FEEDBACK: Your previous output produced a KeyError" in attempts[1]
        assert "[SYSTEM FEEDBACK: Your previous output produced a KeyError" in attempts[2]
        assert await g.vertices["B"].fetch_data("in") == "valid_data"

    @pytest.mark.asyncio
    async def test_retry_feedback_is_not_stacked_across_attempts(self):
        """Self-correction feedback must not accumulate into a stacked prompt.

        Scenario: post_process always raises KeyError -> every attempt retries.
        Each retry must rebuild the active prompt from the frozen base template,
        so every intercepted prompt carries exactly ONE [SYSTEM FEEDBACK] block
        (no duplicate error stack), and the edge prompt is restored afterwards.
        """
        captured = []

        def always_bad(data, prompt, model, settings):
            captured.append(prompt)
            return {"wrong_key": "bad"}

        class RetryEdge(Edge):
            def post_process(self, result, settings):
                return result["target_key"]  # always KeyError

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"channel": "in", "value": "x"}]},
                {"id": "B"},
            ],
            "edges": [
                {
                    "id": "e_retry_stack",
                    "source": "A",
                    "destination": "B",
                    "channel": "in",
                    "settings": {
                        "prompt": "extract target_key",
                        "retry_policy": {
                            "max_retries": 3,
                            "backoff_factor": 0.01,
                            "retry_on": ["KeyError"],
                        },
                    },
                }
            ],
        }

        g = Graph.from_dict(config)
        old = g.edges["e_retry_stack"]
        e_retry = RetryEdge(
            edge_id=old.id, source_id=old.source_id,
            destination_id=old.destination_id, channel=old.channel,
            settings=old.settings, concurrency_type=old.concurrency_type,
            max_iterations=old.max_iterations,
        )
        g.edges["e_retry_stack"] = e_retry

        agent = MockAgent(response_fn=always_bad)
        result = await Executor(g, agent).run()

        assert not result.success
        # 1 initial attempt + 3 retries
        assert len(captured) == 4
        # First call: pristine base prompt, no feedback
        assert captured[0] == "extract target_key"
        assert "[SYSTEM FEEDBACK:" not in captured[0]
        # Every retried prompt carries exactly ONE feedback block, never a stack.
        for i in range(1, 4):
            assert captured[i].count("[SYSTEM FEEDBACK:") == 1, (
                f"attempt {i} stacked feedback: {captured[i]!r}"
            )
            assert captured[i].startswith("extract target_key")
        # Edge prompt restored to base after execution (no state pollution).
        assert e_retry.prompt == "extract target_key"

    @pytest.mark.asyncio
    async def test_retry_exceeds_max_retries_fails_gracefully(self):
        """If exceptions continue past max_retries, it should raise and signal FAILED."""
        def always_failing_agent(data, prompt, model, settings):
            return "not_a_number"

        class FailEdge(Edge):
            def post_process(self, result, settings):
                return int(result)  # ValueError

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"channel": "x", "value": 1}]},
                {"id": "B"},
            ],
            "edges": [
                {
                    "id": "e_fail",
                    "source": "A",
                    "destination": "B",
                    "channel": "x",
                    "settings": {
                        "prompt": "output number",
                        "retry_policy": {
                            "max_retries": 2,
                            "backoff_factor": 0.01,
                            "retry_on": ["ValueError"],
                        }
                    },
                }
            ],
        }

        g = Graph.from_dict(config)
        
        old_edge = g.edges["e_fail"]
        e_fail = FailEdge(
            edge_id=old_edge.id, source_id=old_edge.source_id,
            destination_id=old_edge.destination_id, channel=old_edge.channel,
            settings=old_edge.settings, concurrency_type=old_edge.concurrency_type,
            max_iterations=old_edge.max_iterations
        )
        g.edges["e_fail"] = e_fail
        edge = e_fail

        agent = MockAgent(response_fn=always_failing_agent)
        result = await Executor(g, agent).run()

        assert not result.success
        assert edge.error is not None
        assert "invalid literal for int()" in edge.error

    @pytest.mark.asyncio
    async def test_non_matching_retry_on_fails_immediately(self):
        """If exception type is not in retry_on, do not retry."""
        attempt_count = {"n": 0}

        def bad_agent(data, prompt, model, settings):
            attempt_count["n"] += 1
            raise TypeError("unexpected type")

        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"channel": "x", "value": 1}]},
                {"id": "B"},
            ],
            "edges": [
                {
                    "id": "e_type_err",
                    "source": "A",
                    "destination": "B",
                    "channel": "x",
                    "settings": {
                        "prompt": "test",
                        "retry_policy": {
                            "max_retries": 5,
                            "backoff_factor": 0.01,
                            "retry_on": ["KeyError"],  # Only retry on KeyError
                        }
                    },
                }
            ],
        }

        g = Graph.from_dict(config)
        agent = MockAgent(response_fn=bad_agent)
        result = await Executor(g, agent).run()

        assert not result.success
        assert attempt_count["n"] == 1  # No retries executed


# =====================================================================
# Phase 2 Tests: Real-Time Event Streaming (Sidecar Observability)
# =====================================================================

class TestExecutorEventStreaming:
    @pytest.mark.asyncio
    async def test_stream_yields_all_lifecycle_events(self):
        """
        Verify that executor.stream() produces structured events in real-time
        without blocking execution.
        """
        config = {
            "vertices": [
                {"id": "start", "initial_data": [{"channel": "msg", "value": "hello"}]},
                {"id": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "start", "destination": "end", "channel": "msg", "settings": {"prompt": "echo"}}
            ],
        }
        g = Graph.from_dict(config)
        ex = Executor(g, MockAgent())

        collected_events = []
        async for event in ex.stream():
            assert isinstance(event, GraphEvent)
            assert event.timestamp.endswith("Z")
            collected_events.append(event)

        event_types = [e.event_type for e in collected_events]
        assert "workflow_started" in event_types
        assert "vertex_state_changed" in event_types
        assert "edge_started" in event_types
        assert "edge_completed" in event_types
        assert "workflow_finished" in event_types
        assert ex._result.success is True

    @pytest.mark.asyncio
    async def test_run_and_stream_produce_identical_results(self):
        """Verify backwards compatibility: run() and stream() arrive at the exact same ExecutionResult."""
        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"channel": "val", "value": 10}]},
                {"id": "B"},
            ],
            "edges": [
                {"id": "ab", "source": "A", "destination": "B", "channel": "val", "settings": {"prompt": "+5"}}
            ],
        }
        agent = MockAgent(response_fn=lambda d, p, m, s: d + 5)

        # 1. Via stream()
        g1 = Graph.from_dict(config)
        ex1 = Executor(g1, agent)
        events = [ev async for ev in ex1.stream()]
        res1 = ex1._result

        # 2. Via run()
        g2 = Graph.from_dict(config)
        ex2 = Executor(g2, agent)
        res2 = await ex2.run()

        assert res1.success == res2.success == True
        assert res1.vertex_results["B"]["data"] == res2.vertex_results["B"]["data"]
        assert len(events) >= 4
