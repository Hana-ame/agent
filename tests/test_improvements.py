"""Tests for newly implemented improvements:
- Graph.to_dict() and Graph.to_json() serialization
- GraphBuilder fluent API
- BaseStateStore interface
- Per-edge timeout control
- SchemaMismatchError specific exception
"""

import asyncio
import json
import pytest
import tempfile
import os

from framework import (
    Graph,
    Executor,
    HttpLLMAgent,
    GraphBuilder,
    BaseStateStore,
    SQLiteStateStore,
    SchemaMismatchError,
    SchemaRegistry,
)
from pydantic import BaseModel


class TestGraphSerialization:
    def test_graph_to_dict_and_to_json(self):
        g = (
            GraphBuilder("test_pipeline", "Test pipeline description")
            .vertex("A", initial_data=[{"channel": "msg", "value": "start"}])
            .vertex("B", settings={"key": "val"})
            .edge("A", "B", edge_id="e1", prompt="Process", model="gemini-pro", max_iterations=2)
            .build()
        )

        d = g.to_dict()
        assert d["metadata"]["name"] == "test_pipeline"
        assert len(d["vertices"]) == 2
        assert len(d["edges"]) == 1
        assert d["edges"][0]["max_iterations"] == 2

        # Test to_json string
        json_str = g.to_json()
        parsed = json.loads(json_str)
        assert parsed["metadata"]["name"] == "test_pipeline"

        # Test roundtrip from_dict
        g2 = Graph.from_dict(d)
        assert len(g2.vertices) == 2
        assert len(g2.edges) == 1
        assert "e1" in g2.edges

        # Test to_json file writing
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tf_path = tf.name
        try:
            g.to_json(tf_path)
            g3 = Graph.from_json(tf_path)
            assert len(g3.vertices) == 2
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)


class TestGraphBuilder:
    def test_fluent_builder_dag(self):
        builder = GraphBuilder("fluent_dag")
        g = (
            builder
            .vertex("start", initial_data=[{"channel": "default", "value": "init"}])
            .vertex("mid")
            .vertex("end")
            .edge("start", "mid", prompt="Step 1")
            .edge("mid", "end", prompt="Step 2")
            .build()
        )
        assert len(g.vertices) == 3
        assert len(g.edges) == 2
        assert g.metadata["name"] == "fluent_dag"

    @pytest.mark.asyncio
    async def test_fluent_builder_execution(self):
        g = (
            GraphBuilder()
            .vertex("A", initial_data=[{"channel": "default", "value": "hello"}])
            .vertex("B")
            .edge("A", "B", prompt="echo", model="mock")
            .build()
        )
        executor = Executor(g, agents=HttpLLMAgent(mock=True))
        result = await executor.run()
        assert result.success is True
        data = await g.vertices["B"].fetch_data("default")
        assert "[mock] hello" in str(data)


class TestBaseStateStore:
    def test_sqlite_inherits_base_store(self):
        assert issubclass(SQLiteStateStore, BaseStateStore)
        store = SQLiteStateStore(":memory:")
        assert isinstance(store, BaseStateStore)


class TestEdgeTimeout:
    @pytest.mark.asyncio
    async def test_edge_timeout_triggers_failure(self):
        """When an edge has a timeout configured, exceeding it should fail the edge."""
        async def slow_fn(data, prompt, model, settings):
            await asyncio.sleep(0.5)
            return "too slow"

        g = (
            GraphBuilder()
            .vertex("A", initial_data=[{"channel": "default", "value": "data"}])
            .vertex("B")
            .edge("A", "B", prompt="slow", settings={"timeout": 0.05})
            .build()
        )

        agent = HttpLLMAgent(mock=True, mock_handler=slow_fn)
        executor = Executor(g, agents=agent)
        result = await executor.run()

        # Edge should fail due to TimeoutError
        assert result.success is False
        assert g.edges["e_A_B"].error is not None
        assert "TimeoutError" in str(result.errors) or "timed out" in str(result.errors).lower() or g.vertices["B"].state.value == "error"


class TestSchemaMismatchErrorDirect:
    def test_schema_mismatch_raises_specific_error(self):
        class SchemaA(BaseModel):
            x: int

        class SchemaB(BaseModel):
            y: str

        SchemaRegistry.register("SchemaA", SchemaA)
        SchemaRegistry.register("SchemaB", SchemaB)

        config = {
            "vertices": [
                {"id": "V1"},
                {"id": "V2", "settings": {"input_schema": "SchemaB"}},
            ],
            "edges": [
                {"id": "E1", "source": "V1", "destination": "V2", "settings": {"output_schema": "SchemaA"}},
            ]
        }

        with pytest.raises(SchemaMismatchError) as excinfo:
            Graph.from_dict(config)

        assert "Schema Mismatch on edge 'E1'" in str(excinfo.value)


class TestExecutorHooks:
    @pytest.mark.asyncio
    async def test_executor_hooks_lifecycle(self):
        from framework import ExecutorHooks

        events = []

        class TrackingHooks(ExecutorHooks):
            async def on_workflow_start(self, graph):
                events.append("started")

            async def on_vertex_state_changed(self, vertex, state):
                events.append(f"vertex:{vertex.id}:{state.value}")

            async def on_edge_started(self, edge):
                events.append(f"edge_start:{edge.id}")

            async def on_edge_completed(self, edge, result):
                events.append(f"edge_done:{edge.id}")

            async def on_workflow_finish(self, result):
                events.append("finished")

        g = (
            GraphBuilder()
            .vertex("N1", initial_data=[{"channel": "default", "value": "data"}])
            .vertex("N2")
            .edge("N1", "N2", edge_id="e_step", prompt="process")
            .build()
        )

        hooks = TrackingHooks()
        executor = Executor(g, agents=HttpLLMAgent(mock=True), hooks=hooks)
        result = await executor.run()

        assert result.success is True
        # Allow async hook tasks to settle
        await asyncio.sleep(0.05)

        assert "started" in events
        assert "finished" in events
        assert any("edge_done:e_step" in e for e in events)


class TestSQLiteStoreLifecycle:
    def test_store_context_manager_and_close(self):
        store = SQLiteStateStore(":memory:")
        store.create_run("test_run")
        assert store.get_run("test_run") is not None
        store.close()
        with pytest.raises(RuntimeError, match="closed"):
            store.get_run("test_run")

        with SQLiteStateStore(":memory:") as s:
            s.create_run("ctx_run")
            assert s.get_run("ctx_run") is not None
        assert s._closed is True


class TestBaseAgentStreaming:
    @pytest.mark.asyncio
    async def test_base_agent_stream_process(self):
        agent = HttpLLMAgent(mock=True)
        chunks = []
        async for chunk in agent.stream_process("hello world", "prompt", "test-model"):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert "[test-model] hello world" in full_text


class TestErrorHierarchy:
    def test_error_inheritance(self):
        from framework import (
            FrameworkError,
            ExecutionError,
            GuardAbortError,
            AbortPipeline,
            HookError,
            ComputeError,
            SubgraphError,
            DataRejectedError,
        )

        assert issubclass(ExecutionError, FrameworkError)
        assert issubclass(GuardAbortError, ExecutionError)
        assert issubclass(AbortPipeline, GuardAbortError)
        assert issubclass(HookError, ExecutionError)
        assert issubclass(ComputeError, ExecutionError)
        assert issubclass(SubgraphError, ExecutionError)
        assert issubclass(DataRejectedError, FrameworkError)

        err = AbortPipeline("condition failed")
        assert isinstance(err, GuardAbortError)
        assert isinstance(err, ExecutionError)
        assert isinstance(err, FrameworkError)
