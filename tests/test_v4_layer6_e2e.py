"""Layer 6 E2E Tests: System E2E Tests & Distributed Worker Interface."""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any

import pytest

from framework.worker_queue_v4 import (
    EdgeTaskPayload,
    EdgeTaskResult,
    InMemoryWorkerQueueV4,
)
from framework.executor_v4 import ExecutorV4
from framework.graph_v4 import GraphV4
from framework.edge_v4 import CodeEdgeV4, ReflexiveEdgeV4
from framework.vertex_v4 import (
    VertexRecordV4,
    VertexStateV4,
    VertexAttributeV4,
    VertexStoreV4,
)
from framework.server_v4 import SessionGraphManagerV4
from framework.sse_executor_v4 import SSEExecutorV4
from framework.sensenova_edge_v4 import SenseNovaEdgeV4


@pytest.mark.asyncio
async def test_worker_queue_interface():
    """Test the InMemoryWorkerQueueV4."""
    queue = InMemoryWorkerQueueV4()
    assert await queue.health_check() is True

    # Submit and complete
    task = EdgeTaskPayload(
        edge_id="e1",
        input_content="hello",
    )
    task_id = await queue.submit_edge(task)
    assert task_id

    # Poll before complete
    result = await queue.poll_result(task_id)
    assert result is None

    # Complete task manually
    queue.complete_task(
        task_id,
        EdgeTaskResult(task_id=task_id, edge_id="e1", success=True, output="world")
    )

    # Poll after complete
    result = await queue.poll_result(task_id)
    assert result is not None
    assert result.success is True
    assert result.output == "world"

    # Wait result
    result2 = await queue.wait_result(task_id, timeout=1.0)
    assert result2.output == "world"

    # Task cancellation
    task2 = EdgeTaskPayload(edge_id="e2")
    task_id2 = await queue.submit_edge(task2)
    canceled = await queue.cancel_task(task_id2)
    assert canceled is True
    
    result3 = await queue.wait_result(task_id2)
    assert result3.success is False
    assert result3.error == "Task cancelled"


@pytest.mark.asyncio
async def test_full_pipeline_execution(tmp_path: Path):
    """Test building a multi-step graph with code edges and executing it end-to-end."""
    store = VertexStoreV4(":memory:")
    session_id = "e2e_sess_1"
    graph = GraphV4(session_id)

    # Add vertices
    graph.add_vertex(VertexRecordV4(0, session_id, "v_in", "start data", [VertexAttributeV4.START.value], VertexStateV4.DATA_READY.value))
    graph.add_vertex(VertexRecordV4(0, session_id, "v_mid", "", [], VertexStateV4.TODO.value))
    graph.add_vertex(VertexRecordV4(0, session_id, "v_out", "", [VertexAttributeV4.END.value], VertexStateV4.TODO.value))

    # Add edges
    def step1(content, settings, staging):
        return f"{content} -> step1"

    def step2(content, settings, staging):
        return f"{content} -> step2"

    graph.add_edge(CodeEdgeV4("e1", "v_in", "v_mid", script=step1))
    graph.add_edge(CodeEdgeV4("e2", "v_mid", "v_out", script=step2))
    
    graph.validate()

    executor = ExecutorV4(graph=graph, store=store)
    result = await executor.run()

    assert result.success is True
    assert "e1" in result.completed_edges
    assert "e2" in result.completed_edges

    v_out = store.get_vertex(session_id, "v_out")
    assert v_out.state == VertexStateV4.DATA_READY.value
    assert v_out.content == "start data -> step1 -> step2"


@pytest.mark.asyncio
async def test_deep_recursive_subgraph_execution(tmp_path: Path):
    """Test 3-level nested subgraphs."""
    # Similar to test_v4_system.py, but confirming deep recursion again
    # 1. Level 3 subgraph
    l3_dir = tmp_path / "l3_graph"
    l3_dir.mkdir(parents=True)
    (l3_dir / "l3_in.json").write_text(json.dumps({"name": "l3_start", "content": "", "attributes": ["start"], "state": "idle"}))
    (l3_dir / "l3_out.json").write_text(json.dumps({"name": "l3_end", "content": "", "attributes": ["end"], "state": "todo"}))
    l3_script = l3_dir / "l3_trans.py"
    l3_script.write_text("def process(data, settings=None, staging=None):\n    return f'{data}->L3'\n")
    (l3_dir / "l3_edge.json").write_text(json.dumps({
        "id": "e_l3", "type": "code", "input_vertex": "l3_start", "output_vertex": "l3_end", "script": f"{l3_script}:process"
    }))
    l3_manifest = l3_dir / "graph.json"
    l3_manifest.write_text(json.dumps({
        "version": "4.0", "session_id": "l3_sess",
        "vertices": ["l3_in.json", "l3_out.json"], "edges": ["l3_edge.json"]
    }))

    # 2. Level 2 subgraph
    l2_dir = tmp_path / "l2_graph"
    l2_dir.mkdir(parents=True)
    (l2_dir / "l2_in.json").write_text(json.dumps({"name": "l2_start", "content": "", "attributes": ["start"], "state": "idle"}))
    (l2_dir / "l2_box.json").write_text(json.dumps({
        "name": "l2_subbox", "content": json.dumps({"subgraph_manifest": str(l3_manifest)}),
        "attributes": ["subgraph"], "state": "todo"
    }))
    (l2_dir / "l2_out.json").write_text(json.dumps({"name": "l2_end", "content": "", "attributes": ["end"], "state": "todo"}))
    (l2_dir / "e_l2_in.json").write_text(json.dumps({"id": "e_l2_in", "type": "code", "input_vertex": "l2_start", "output_vertex": "l2_subbox"}))
    (l2_dir / "e_l2_out.json").write_text(json.dumps({"id": "e_l2_out", "type": "code", "input_vertex": "l2_subbox", "output_vertex": "l2_end"}))
    l2_manifest = l2_dir / "graph.json"
    l2_manifest.write_text(json.dumps({
        "version": "4.0", "session_id": "l2_sess",
        "vertices": ["l2_in.json", "l2_box.json", "l2_out.json"], "edges": ["e_l2_in.json", "e_l2_out.json"]
    }))

    # 3. Level 1 subgraph (Root)
    parent_sess = "parent_recursive_sess"
    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    parent_graph = manager.get_or_create_graph(parent_sess)

    parent_graph.add_vertex(VertexRecordV4(0, parent_sess, "root", "TOP", [VertexAttributeV4.START.value], VertexStateV4.DATA_READY.value))
    parent_graph.add_vertex(VertexRecordV4(0, parent_sess, "l1_box", json.dumps({"subgraph_manifest": str(l2_manifest)}), [VertexAttributeV4.SUBGRAPH.value], VertexStateV4.TODO.value))
    parent_graph.add_vertex(VertexRecordV4(0, parent_sess, "sink", "", [VertexAttributeV4.END.value], VertexStateV4.TODO.value))
    parent_graph.add_edge(CodeEdgeV4("e_to_l1", "root", "l1_box"))
    parent_graph.add_edge(CodeEdgeV4("e_from_l1", "l1_box", "sink"))

    executor = SSEExecutorV4(manager=manager, store=store)
    res = await executor.execute_harness_call(session_id=parent_sess)
    info = json.loads(res["function"]["arguments"])["info"]
    assert info["success"] is True

    sink_v = store.get_vertex(parent_sess, "sink")
    assert sink_v.content == "TOP->L3"
    assert sink_v.state == VertexStateV4.DATA_READY.value


@pytest.mark.asyncio
async def test_sse_streaming_e2e(tmp_path: Path):
    """Test SSEExecutorV4.execute_and_stream produces correct event sequence."""
    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    session_id = "sse_sess"
    graph = manager.get_or_create_graph(session_id)

    graph.add_vertex(VertexRecordV4(0, session_id, "in_v", "start", [VertexAttributeV4.START.value], VertexStateV4.DATA_READY.value))
    graph.add_vertex(VertexRecordV4(0, session_id, "out_v", "", [VertexAttributeV4.END.value], VertexStateV4.TODO.value))
    graph.add_edge(CodeEdgeV4("e1", "in_v", "out_v", script=lambda c, s, st: "finished"))

    executor = SSEExecutorV4(manager=manager, store=store)
    
    events = []
    async for chunk in executor.execute_and_stream(session_id=session_id):
        if chunk == "data: [DONE]\n\n":
            continue
        # Extract the JSON payload
        data_str = chunk.replace("data: ", "").strip()
        if data_str:
            data = json.loads(data_str)
            tool_calls = data["choices"][0]["delta"].get("tool_calls", [])
            if tool_calls:
                args = json.loads(tool_calls[0]["function"]["arguments"])
                events.append(args["info"]["event"])
                
    assert "workflow_started" in events
    assert "edge_started" in events
    assert "edge_completed" in events
    assert "workflow_finished" in events


@pytest.mark.asyncio
async def test_reflexive_error_recovery_pipeline():
    """Test an edge fails, reflexive recovers it, and pipeline completes."""
    store = VertexStoreV4(":memory:")
    session_id = "recovery_sess"
    graph = GraphV4(session_id)

    graph.add_vertex(VertexRecordV4(0, session_id, "start_v", "init", [VertexAttributeV4.START.value], VertexStateV4.DATA_READY.value))
    graph.add_vertex(VertexRecordV4(0, session_id, "mid_v", "", [], VertexStateV4.TODO.value))
    graph.add_vertex(VertexRecordV4(0, session_id, "end_v", "", [VertexAttributeV4.END.value], VertexStateV4.TODO.value))

    attempt = 0

    def flaky_code(c, s, st):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise ValueError("Fail on first attempt")
        return f"{c} -> recovered"

    graph.add_edge(CodeEdgeV4("e1", "start_v", "mid_v", script=flaky_code))
    graph.add_edge(ReflexiveEdgeV4("e_refl", "mid_v", trigger_state=VertexStateV4.REJECT.value, max_retries=3))
    graph.add_edge(CodeEdgeV4("e2", "mid_v", "end_v", script=lambda c, s, st: f"{c} -> done"))
    
    graph.validate()

    executor = ExecutorV4(graph=graph, store=store)
    result = await executor.run()

    assert result.success is True
    assert attempt == 2
    assert "e_refl" in result.completed_edges

    end_v = store.get_vertex(session_id, "end_v")
    assert end_v.state == VertexStateV4.DATA_READY.value
    assert end_v.content == "init -> recovered -> done"


@pytest.mark.asyncio
async def test_sensenova_live_pipeline_e2e():
    """Non-mock live test executing a multi-stage graph with real SenseNova 6.8 Flash Lite inference.
    
    Dataflow:
    prompt_node (DATA_READY) 
      --> [SenseNovaEdgeV4] (real LLM call, outputs JSON)
      --> model_node (JSON validated)
      --> [CodeEdgeV4] (parses result and computes double)
      --> final_node (END, DATA_READY)
    """
    store = VertexStoreV4(":memory:")
    session_id = "sensenova_live_e2e_sess"
    graph = GraphV4(session_id)

    # 1. Prompt input vertex
    prompt_content = 'Calculate 15 + 25. Output ONLY a valid JSON object in the form {"result": 40} without any Markdown fences or explanations.'
    graph.add_vertex(VertexRecordV4(
        id=0,
        session_id=session_id,
        name="prompt_node",
        content=prompt_content,
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value,
    ))

    # 2. Intermediate LLM response vertex (with JSON attribute enforcement)
    graph.add_vertex(VertexRecordV4(
        id=0,
        session_id=session_id,
        name="model_node",
        content="",
        attributes=[VertexAttributeV4.JSON.value],
        state=VertexStateV4.TODO.value,
    ))

    # 3. Final sink vertex
    graph.add_vertex(VertexRecordV4(
        id=0,
        session_id=session_id,
        name="final_node",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value,
    ))

    # Real SenseNova LLM edge
    llm_edge = SenseNovaEdgeV4(
        edge_id="e_live_llm",
        input_vertex="prompt_node",
        output_vertex="model_node",
        settings={"temperature": 0.1},
    )

    # Downstream code parsing edge
    def parse_and_double(content: str, settings: dict, staging: dict) -> str:
        data = json.loads(content)
        val = int(data.get("result", 0))
        return json.dumps({"original": val, "doubled": val * 2})

    code_edge = CodeEdgeV4(
        edge_id="e_code_process",
        input_vertex="model_node",
        output_vertex="final_node",
        script=parse_and_double,
    )

    graph.add_edge(llm_edge)
    graph.add_edge(code_edge)
    graph.validate()

    try:
        executor = ExecutorV4(graph=graph, store=store, max_concurrency=2)
        res = await executor.run()

        assert res.success is True
        assert "e_live_llm" in res.completed_edges
        assert "e_code_process" in res.completed_edges

        # Verify intermediate model_node
        mid_v = store.get_vertex(session_id, "model_node")
        assert mid_v.state == VertexStateV4.DATA_READY.value
        mid_data = json.loads(mid_v.content)
        assert mid_data.get("result") == 40

        # Verify final_node transformed content
        final_v = store.get_vertex(session_id, "final_node")
        assert final_v.state == VertexStateV4.DATA_READY.value
        final_data = json.loads(final_v.content)
        assert final_data["original"] == 40
        assert final_data["doubled"] == 80
    finally:
        await llm_edge.close_agent()
        store.close()


@pytest.mark.asyncio
async def test_sensenova_live_sse_streaming_e2e():
    """Non-mock live test executing a workflow with SenseNovaEdgeV4 via SSEExecutorV4 streaming."""
    store = VertexStoreV4(":memory:")
    manager = SessionGraphManagerV4(store)
    session_id = "sensenova_sse_live_sess"
    graph = manager.get_or_create_graph(session_id)

    graph.add_vertex(VertexRecordV4(
        id=0,
        session_id=session_id,
        name="user_query",
        content="Echo the word: PHOENIX",
        attributes=[VertexAttributeV4.START.value],
        state=VertexStateV4.DATA_READY.value,
    ))
    graph.add_vertex(VertexRecordV4(
        id=0,
        session_id=session_id,
        name="ai_response",
        content="",
        attributes=[VertexAttributeV4.END.value],
        state=VertexStateV4.TODO.value,
    ))

    llm_edge = SenseNovaEdgeV4(
        edge_id="e_stream_llm",
        input_vertex="user_query",
        output_vertex="ai_response",
        settings={"temperature": 0.1},
    )
    graph.add_edge(llm_edge)
    graph.validate()

    executor = SSEExecutorV4(manager=manager, store=store)

    events = []
    try:
        async for chunk in executor.execute_and_stream(session_id=session_id):
            if chunk == "data: [DONE]\n\n":
                continue
            data_str = chunk.replace("data: ", "").strip()
            if data_str:
                data = json.loads(data_str)
                tool_calls = data["choices"][0]["delta"].get("tool_calls", [])
                if tool_calls:
                    args = json.loads(tool_calls[0]["function"]["arguments"])
                    events.append(args["info"]["event"])

        assert "workflow_started" in events
        assert "edge_completed" in events
        assert "workflow_finished" in events

        res_v = store.get_vertex(session_id, "ai_response")
        assert res_v.state == VertexStateV4.DATA_READY.value
        assert "PHOENIX" in res_v.content.upper()
    finally:
        await llm_edge.close_agent()
        store.close()

