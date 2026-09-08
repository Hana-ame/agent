"""Live integration tests for SenseNovaEdgeV4 executing real remote inference.

Validates:
1. Two-sided handshake and real LLM generation against SenseNova 6.8 Flash Lite.
2. Structured JSON response parsing and markdown fence stripping with JSON attribute.
3. Live SSE streaming execution via run_stream().
4. End-to-end multi-edge graph pipeline execution with ExecutorV4.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator
import pytest

from framework.edge_v4 import CodeEdgeV4
from framework.executor_v4 import ExecutorV4
from framework.graph_v4 import GraphV4
from framework.sensenova_edge_v4 import SenseNovaEdgeV4
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)


@pytest.fixture
def live_store() -> VertexStoreV4:
    """Provide an in-memory SQLite store for live test sessions."""
    store = VertexStoreV4(":memory:")
    yield store
    store.close()


@pytest.mark.live
@pytest.mark.asyncio
async def test_sensenova_live_handshake_and_generation(live_store: VertexStoreV4):
    """Execute live SenseNova inference and verify handshake, output, and staging."""
    session_id = "test_live_generation"
    live_store.save_vertex(
        session_id=session_id,
        name="prompt_node",
        content="What is 2 + 2? Answer only with the single number 4.",
        state=VertexStateV4.DATA_READY.value,
    )
    live_store.save_vertex(
        session_id=session_id,
        name="output_node",
        content="",
        state=VertexStateV4.TODO.value,
    )

    edge = SenseNovaEdgeV4(
        edge_id="e_live_calc",
        input_vertex="prompt_node",
        output_vertex="output_node",
        settings={"temperature": 0.1},
    )

    try:
        res = await edge.run(session_id, live_store)
        assert res.success is True, f"Inference failed with reason: {res.reason}"
        assert res.output is not None
        assert "4" in res.output

        # Verify destination vertex updated in SQLite store
        out_v = live_store.get_vertex(session_id, "output_node")
        assert out_v is not None
        assert out_v.state == VertexStateV4.DATA_READY.value
        assert out_v.content == res.output
        assert out_v.processed_count == 1

        # Verify staged observability prompt
        staged = live_store.get_staged(session_id, vertex_name="output_node")
        staged_keys = {item.key for item in staged}
        assert "rendered_prompt" in staged_keys

        # Verify usage metadata reported
        assert "usage" in res.metadata
        assert res.metadata["usage"].get("calls", 0) >= 1
    finally:
        await edge.close_agent()


@pytest.mark.live
@pytest.mark.asyncio
async def test_sensenova_live_json_validation(live_store: VertexStoreV4):
    """Execute live SenseNova inference with downstream JSON attribute enforcement."""
    session_id = "test_live_json"
    live_store.save_vertex(
        session_id=session_id,
        name="json_prompt",
        content='Output ONLY a valid JSON object without any commentary: {"city": "Tokyo", "status": "ok"}',
        state=VertexStateV4.DATA_READY.value,
    )
    live_store.save_vertex(
        session_id=session_id,
        name="json_result",
        content="",
        attributes=[VertexAttributeV4.JSON.value],
        state=VertexStateV4.TODO.value,
    )

    edge = SenseNovaEdgeV4(
        edge_id="e_live_json",
        input_vertex="json_prompt",
        output_vertex="json_result",
        settings={"temperature": 0.1},
    )

    try:
        res = await edge.run(session_id, live_store)
        assert res.success is True, f"JSON inference failed: {res.reason}"

        # Parse output as JSON to confirm clean extraction
        parsed = json.loads(res.output)
        assert isinstance(parsed, dict)
        assert parsed.get("status") == "ok"

        # Verify store state
        out_v = live_store.get_vertex(session_id, "json_result")
        assert out_v is not None
        assert out_v.state == VertexStateV4.DATA_READY.value
    finally:
        await edge.close_agent()


@pytest.mark.live
@pytest.mark.asyncio
async def test_sensenova_live_streaming(live_store: VertexStoreV4):
    """Execute live SenseNova SSE streaming and verify real-time chunks and final state."""
    session_id = "test_live_stream"
    live_store.save_vertex(
        session_id=session_id,
        name="stream_in",
        content="Count sequentially: 1, 2, 3.",
        state=VertexStateV4.DATA_READY.value,
    )
    live_store.save_vertex(
        session_id=session_id,
        name="stream_out",
        content="",
        state=VertexStateV4.TODO.value,
    )

    edge = SenseNovaEdgeV4(
        edge_id="e_live_stream",
        input_vertex="stream_in",
        output_vertex="stream_out",
    )

    try:
        chunks = []
        async for chunk in edge.run_stream(session_id, live_store):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_content = "".join(chunks)
        assert len(full_content.strip()) > 0

        # Verify store updated to data ready
        out_v = live_store.get_vertex(session_id, "stream_out")
        assert out_v is not None
        assert out_v.state == VertexStateV4.DATA_READY.value
        assert out_v.content == full_content
    finally:
        await edge.close_agent()


@pytest.mark.live
@pytest.mark.asyncio
async def test_sensenova_live_in_graph_pipeline(live_store: VertexStoreV4):
    """Execute a multi-tier GraphV4 pipeline with live SenseNova edge and CodeEdge."""
    session_id = "test_live_pipeline"

    graph = GraphV4(session_id=session_id, name="live_sn_pipeline")
    v1 = VertexRecordV4(0, session_id, "user_query", "Respond with the single word: ANTIGRAVITY", [], VertexStateV4.DATA_READY.value)
    v2 = VertexRecordV4(0, session_id, "llm_reply", "", [], VertexStateV4.TODO.value)
    v3 = VertexRecordV4(0, session_id, "final_echo", "", [], VertexStateV4.TODO.value)

    graph.add_vertex(v1)
    graph.add_vertex(v2)
    graph.add_vertex(v3)

    live_store.save_vertex(session_id, v1.name, content=v1.content, state=v1.state)
    live_store.save_vertex(session_id, v2.name, content=v2.content, state=v2.state)
    live_store.save_vertex(session_id, v3.name, content=v3.content, state=v3.state)

    sn_edge = SenseNovaEdgeV4(
        edge_id="e1_sn",
        input_vertex="user_query",
        output_vertex="llm_reply",
        settings={"temperature": 0.1},
    )
    code_edge = CodeEdgeV4(
        edge_id="e2_format",
        input_vertex="llm_reply",
        output_vertex="final_echo",
    )

    graph.add_edge(sn_edge)
    graph.add_edge(code_edge)
    graph.validate(strict_dag=False)

    try:
        executor = ExecutorV4(graph=graph, store=live_store)
        exec_res = await executor.run()

        assert exec_res.success is True
        assert "e1_sn" in exec_res.completed_edges
        assert "e2_format" in exec_res.completed_edges

        # Check final output vertex
        v3_final = live_store.get_vertex(session_id, "final_echo")
        assert v3_final is not None
        assert v3_final.state == VertexStateV4.DATA_READY.value
        assert "ANTIGRAVITY" in v3_final.content.upper()
    finally:
        await sn_edge.close_agent()


if __name__ == "__main__":
    async def run_standalone():
        store = VertexStoreV4(":memory:")
        print("Running test_sensenova_live_handshake_and_generation...")
        await test_sensenova_live_handshake_and_generation(store)
        print("PASS")
        print("Running test_sensenova_live_json_validation...")
        await test_sensenova_live_json_validation(store)
        print("PASS")
        print("Running test_sensenova_live_streaming...")
        await test_sensenova_live_streaming(store)
        print("PASS")
        print("Running test_sensenova_live_in_graph_pipeline...")
        await test_sensenova_live_in_graph_pipeline(store)
        print("PASS")
        store.close()

    asyncio.run(run_standalone())
