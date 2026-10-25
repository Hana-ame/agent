"""Tests for V4 WorkflowExecutorV4 and server SSE streaming adapter separation."""

import json
import pytest
from pathlib import Path

from framework.server.manager import SessionGraphManagerV4
from framework.server.sse import ToolCallEcho, format_result_as_harness_echo, stream_workflow_as_sse
from framework.sse_executor_v4 import SSEExecutorV4
from framework.vertex_v4 import VertexStoreV4
from framework.workflow_executor_v4 import WorkflowEventV4, WorkflowExecutorV4, WorkflowResultV4


@pytest.mark.asyncio
async def test_workflow_executor_clean_domain_execution():
    """Verify WorkflowExecutorV4 produces pure domain objects without SSE strings."""
    demo_dir = Path(__file__).resolve().parent.parent / "examples" / "subgraph_v4"
    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    session_id = "test_clean_workflow_sess"

    graph = manager.get_or_create_graph(session_id)
    graph.add_vertex(demo_dir / "parent_in.json")
    graph.add_vertex(demo_dir / "parent_subgraph.json")
    graph.add_vertex(demo_dir / "parent_out.json")
    graph.add_edge(demo_dir / "e_start_to_subgraph.json")
    graph.add_edge(demo_dir / "e_subgraph_to_output.json")

    executor = WorkflowExecutorV4(manager=manager, store=store)

    # 1. Non-streaming execution produces WorkflowResultV4
    result: WorkflowResultV4 = await executor.run(session_id=session_id)
    assert isinstance(result, WorkflowResultV4)
    assert result.success is True
    assert "e_in_to_subgraph" in result.completed_edges
    assert "bridge_subgraph_enrichment_box" in result.completed_edges
    assert "e_subgraph_to_out" in result.completed_edges

    out_content = json.loads(result.vertex_contents["final_output"])
    assert out_content["processed_text"] == "Hello Antigravity Nested Subgraph Processing"
    assert out_content["word_count"] == 5

    # 2. Convert to harness echo in server layer
    echo_dict = format_result_as_harness_echo(result)
    assert echo_dict["type"] == "function"
    assert echo_dict["function"]["name"] == "echo"
    parsed_args = json.loads(echo_dict["function"]["arguments"])
    assert parsed_args["info"]["success"] is True


@pytest.mark.asyncio
async def test_workflow_executor_stream_and_sse_adapter():
    """Verify WorkflowExecutorV4 event stream and server SSE adapter formatting."""
    demo_dir = Path(__file__).resolve().parent.parent / "examples" / "subgraph_v4"
    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    session_id = "test_clean_stream_sess"

    graph = manager.get_or_create_graph(session_id)
    graph.add_vertex(demo_dir / "parent_in.json")
    graph.add_vertex(demo_dir / "parent_subgraph.json")
    graph.add_vertex(demo_dir / "parent_out.json")
    graph.add_edge(demo_dir / "e_start_to_subgraph.json")
    graph.add_edge(demo_dir / "e_subgraph_to_output.json")

    executor = WorkflowExecutorV4(manager=manager, store=store)

    # 1. Direct stream yields pure WorkflowEventV4 instances
    events = []
    async for ev in executor.stream(session_id=session_id):
        assert isinstance(ev, WorkflowEventV4)
        events.append(ev)

    event_types = [e.event_type for e in events]
    assert "workflow_started" in event_types
    assert "workflow_finished" in event_types

    # 2. Server SSE adapter translates stream to SSE chunks
    sse_chunks = []
    stream_gen = executor.stream(session_id=session_id)
    async for chunk in stream_workflow_as_sse(stream_gen):
        assert chunk.startswith("data: ")
        sse_chunks.append(chunk)

    assert sse_chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_sse_executor_v4_backward_compatibility():
    """Verify SSEExecutorV4 façade preserves 100% backward compatibility."""
    demo_dir = Path(__file__).resolve().parent.parent / "examples" / "subgraph_v4"
    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    session_id = "test_facade_sess"

    graph = manager.get_or_create_graph(session_id)
    graph.add_vertex(demo_dir / "parent_in.json")
    graph.add_vertex(demo_dir / "parent_subgraph.json")
    graph.add_vertex(demo_dir / "parent_out.json")
    graph.add_edge(demo_dir / "e_start_to_subgraph.json")
    graph.add_edge(demo_dir / "e_subgraph_to_output.json")

    facade = SSEExecutorV4(manager=manager, store=store)

    # 1. execute_harness_call backward compatibility
    res = await facade.execute_harness_call(session_id=session_id)
    info = json.loads(res["function"]["arguments"])["info"]
    assert info["success"] is True
    assert "bridge_subgraph_enrichment_box" in info["completed_edges"]

    # 2. execute_and_stream backward compatibility
    chunks = []
    async for chunk in facade.execute_and_stream(session_id=session_id):
        chunks.append(chunk)
    assert chunks[-1] == "data: [DONE]\n\n"
