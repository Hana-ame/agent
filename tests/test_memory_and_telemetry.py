"""Unit tests for Global Memory & Telemetry Tracking (v3.0 Pillars A & B)."""

import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import (
    Graph,
    Executor,
    HttpLLMAgent,
    MemoryStore,
    TelemetryTracker,
    UsageMetrics,
    calculate_cost,
    estimate_tokens,
)


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_memory_store_basic_crud(self):
        mem = MemoryStore({"initial_key": "val0"})
        assert await mem.get("initial_key") == "val0"

        await mem.set("k1", 100)
        assert await mem.get("k1") == 100

        await mem.update({"k2": "hello", "k3": True})
        assert await mem.get("k2") == "hello"
        assert await mem.get("k3") is True

        deleted = await mem.delete("k1")
        assert deleted is True
        assert await mem.get("k1") is None

    @pytest.mark.asyncio
    async def test_memory_store_ttl_expiration(self):
        mem = MemoryStore()
        await mem.set("temp_token", "secret", ttl=0.05)
        assert await mem.get("temp_token") == "secret"

        await asyncio.sleep(0.08)
        assert await mem.get("temp_token") is None

    @pytest.mark.asyncio
    async def test_memory_scoped_view(self):
        root_mem = MemoryStore()
        user_scope = root_mem.scope("user_1")
        await user_scope.set("profile", {"name": "Alice"})

        # Scoped view can read with local key
        assert await user_scope.get("profile") == {"name": "Alice"}
        # Root view sees namespaced key
        assert await root_mem.get("user_1:profile") == {"name": "Alice"}

        all_user_data = await user_scope.get_all()
        assert all_user_data == {"profile": {"name": "Alice"}}


class TestMemoryGraphIntegration:
    @pytest.mark.asyncio
    async def test_edges_read_and_write_global_memory(self):
        """
        Verify:
        Edge1 writes output into global memory key 'session_id'.
        Edge2 (downstream or parallel) reads 'session_id' directly from memory.
        """
        config = {
            "vertices": [
                {"id": "Start", "initial_data": [{"channel": "in", "value": "init"}]},
                {"id": "Middle"},
                {"id": "End"},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "Start",
                    "destination": "Middle",
                    "channel": "in",
                    "settings": {
                        "prompt": "generate session",
                        # Write result dict field 'session_token' -> memory 'global_session'
                        "memory_write": {"session_token": "global_session"}
                    }
                },
                {
                    "id": "e2",
                    "source": "Middle",
                    "destination": "End",
                    "channel": "in",
                    "settings": {
                        "prompt": "consume session",
                        # Read 'global_session' from memory into input data
                        "memory_read": ["global_session"]
                    }
                },
            ],
        }

        def mock_llm_fn(data, prompt, model, settings):
            if "generate session" in prompt:
                return {"session_token": "SESS-XYZ-999", "status": "ok"}
            elif "consume session" in prompt:
                # Expect 'global_session' to have been injected from memory into dict data
                return f"Authenticated with {data.get('global_session')}"
            return "ok"

        memory = MemoryStore()
        g = Graph.from_dict(config)
        agent = HttpLLMAgent(mock=True, mock_handler=mock_llm_fn)
        executor = Executor(g, agents=agent, memory=memory)

        result = await executor.run()
        assert result.success is True

        # Verify global memory has written value
        assert await memory.get("global_session") == "SESS-XYZ-999"
        assert result.memory_snapshot.get("global_session") == "SESS-XYZ-999"

        # Verify End node received authenticated output
        end_data = await g.vertices["End"].fetch_data("in")
        assert end_data == "Authenticated with SESS-XYZ-999"


class TestTelemetryAndCostProfiling:
    def test_token_estimation_and_cost_calculation(self):
        text = "Hello world from vertex-edge agent!"
        tokens = estimate_tokens(text)
        assert tokens > 0

        # 1,000,000 prompt tokens with gemini-1.5-pro = $3.50
        cost = calculate_cost(prompt_tokens=1_000_000, completion_tokens=1_000_000, model="gemini-1.5-pro")
        assert cost == (3.50 + 10.50)

    def test_free_tier_model_has_zero_cost(self):
        """Free-tier models (hy3-free) must bill $0.00 — never fall back to
        the 'default' paid rates, which would inflate report figures."""
        assert calculate_cost(1_000_000, 1_000_000, "hy3-free") == 0.0
        assert calculate_cost(500_000, 200_000, "hy3-free") == 0.0

    @pytest.mark.asyncio
    async def test_executor_records_telemetry_metrics_and_events(self):
        config = {
            "vertices": [
                {"id": "A", "initial_data": [{"channel": "q", "value": "What is AI?"}]},
                {"id": "B"},
            ],
            "edges": [
                {
                    "id": "e_ask",
                    "source": "A",
                    "destination": "B",
                    "channel": "q",
                    "settings": {"prompt": "Answer thoroughly", "model": "gpt-4o"},
                }
            ],
        }

        g = Graph.from_dict(config)
        agent = HttpLLMAgent(mock=True, mock_handler=lambda d, p, m, s: "AI is artificial intelligence that simulates human cognition.")
        executor = Executor(g, agents=agent)

        events = []
        async for event in executor.stream():
            events.append(event)

        assert executor._result.success is True
        metrics = executor._result.metrics
        assert metrics is not None
        assert metrics.total_tokens > 0
        assert metrics.cost_usd > 0.0
        assert metrics.latency_ms >= 0.0

        # Check summary string includes telemetry
        summary = executor._result.summary()
        assert "Prompt Tokens:" in summary
        assert "Estimated Cost:" in summary

        # Check streaming event for edge_completed contained telemetry payload
        completed_ev = next(e for e in events if e.event_type == "edge_completed")
        assert "telemetry" in completed_ev.payload
        assert completed_ev.payload["telemetry"]["total_tokens"] > 0
