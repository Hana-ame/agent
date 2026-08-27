"""Tests for EdgePipeline business retry policy and Executor real-time event streaming."""

import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import Graph, Executor, MockAgent, Vertex, Edge, EdgePipeline, GraphEvent
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
        2. EdgePipeline intercepts KeyError, updates prompt with [SYSTEM FEEDBACK:...].
        3. On 3rd attempt, LLM returns valid dict with 'target_key'.
        4. Overall execution succeeds.
        """
        attempts = []

        def flaking_agent(data, prompt, model, settings):
            attempts.append(prompt)
            if len(attempts) < 3:
                return {"wrong_key": "bad_data"}
            return {"target_key": "valid_data"}

        def my_post_process(result, settings):
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
                    "prompt": "Please extract target JSON",
                    "settings": {
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
        edge = g.edges["e_retry"]
        edge._pipeline._hook_provider = type("Hook", (), {"post_process": staticmethod(my_post_process)})()

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
    async def test_retry_exceeds_max_retries_fails_gracefully(self):
        """If exceptions continue past max_retries, it should raise and signal FAILED."""
        def always_failing_agent(data, prompt, model, settings):
            return "not_a_number"

        def strict_post_process(result, settings):
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
                    "prompt": "output number",
                    "settings": {
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
        edge = g.edges["e_fail"]
        edge._pipeline._hook_provider = type("Hook", (), {"post_process": staticmethod(strict_post_process)})()

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
                    "prompt": "test",
                    "settings": {
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
                {"id": "e1", "source": "start", "destination": "end", "channel": "msg", "prompt": "echo"}
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
                {"id": "ab", "source": "A", "destination": "B", "channel": "val", "prompt": "+5"}
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
