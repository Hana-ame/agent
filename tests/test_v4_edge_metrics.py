"""Tests for Phase 4: Edge-level performance benchmarking and metrics database."""

import asyncio
import json
import pytest
from httpx import ASGITransport, AsyncClient

from framework.edge_v4 import CodeEdgeV4, ToolEdgeV4
from framework.executor_v4 import ExecutorV4
from framework.graph_v4 import GraphV4
from framework.http_executor_v4 import HttpHarnessExecutorV4
from framework.server_v4 import create_v4_server
from framework.vertex_v4 import (
    EdgeMetricRecordV4,
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)


@pytest.mark.asyncio
async def test_vertex_store_record_and_list_metrics():
    """Verify record_edge_metric and list_edge_metrics in VertexStoreV4."""
    store = VertexStoreV4(":memory:")
    try:
        # Record successful metric
        m1 = store.record_edge_metric(
            session_id="sess_test_1",
            edge_id="edge_1",
            edge_type="code",
            input_vertex="node_a",
            output_vertex="node_b",
            execution_time_ms=45.2,
            prompt_tokens=100,
            completion_tokens=25,
            cost_usd=0.0005,
            success=True,
            metadata={"test_tag": "run1"},
        )
        assert isinstance(m1, EdgeMetricRecordV4)
        assert m1.id > 0
        assert m1.session_id == "sess_test_1"
        assert m1.edge_id == "edge_1"
        assert m1.execution_time_ms == 45.2
        assert m1.prompt_tokens == 100
        assert m1.completion_tokens == 25
        assert m1.total_tokens == 125  # auto-computed if 0
        assert m1.cost_usd == 0.0005
        assert m1.success is True
        assert m1.error is None
        assert m1.metadata.get("test_tag") == "run1"
        assert m1.created_at != ""

        # Record failed metric
        m2 = store.record_edge_metric(
            session_id="sess_test_1",
            edge_id="edge_2",
            edge_type="tool",
            input_vertex="node_b",
            output_vertex="node_c",
            execution_time_ms=120.0,
            prompt_tokens=50,
            completion_tokens=0,
            total_tokens=50,
            cost_usd=0.0001,
            success=False,
            error="Command timed out",
        )
        assert m2.success is False
        assert m2.error == "Command timed out"

        # Record metric in another session
        m3 = store.record_edge_metric(
            session_id="sess_test_2",
            edge_id="edge_x",
            edge_type="llm",
            input_vertex="v1",
            output_vertex="v2",
            execution_time_ms=80.0,
            prompt_tokens=300,
            completion_tokens=150,
            success=True,
        )
        assert m3.session_id == "sess_test_2"

        # Filter by session_id
        sess1_metrics = store.list_edge_metrics(session_id="sess_test_1")
        assert len(sess1_metrics) == 2
        assert [m.edge_id for m in sess1_metrics] == ["edge_1", "edge_2"]

        # Filter by edge_id
        edge2_metrics = store.list_edge_metrics(edge_id="edge_2")
        assert len(edge2_metrics) == 1
        assert edge2_metrics[0].edge_id == "edge_2"

        # Limit filter
        limited = store.list_edge_metrics(limit=1)
        assert len(limited) == 1
    finally:
        store.close()


@pytest.mark.asyncio
async def test_vertex_store_metrics_summary():
    """Verify get_edge_metrics_summary aggregations per session and globally."""
    store = VertexStoreV4(":memory:")
    try:
        # Session 1: 2 executions (1 success, 1 failure)
        store.record_edge_metric(
            session_id="sess_1",
            edge_id="e1",
            edge_type="code",
            input_vertex="A",
            output_vertex="B",
            execution_time_ms=10.0,
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.001,
            success=True,
        )
        store.record_edge_metric(
            session_id="sess_1",
            edge_id="e1",
            edge_type="code",
            input_vertex="A",
            output_vertex="B",
            execution_time_ms=30.0,
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=0.002,
            success=False,
            error="Exception",
        )

        # Session 2: 1 execution
        store.record_edge_metric(
            session_id="sess_2",
            edge_id="e2",
            edge_type="tool",
            input_vertex="B",
            output_vertex="C",
            execution_time_ms=60.0,
            prompt_tokens=50,
            completion_tokens=50,
            cost_usd=0.0005,
            success=True,
        )

        # Session 1 summary
        summ_1 = store.get_edge_metrics_summary("sess_1")
        assert summ_1["session_id"] == "sess_1"
        assert summ_1["total_executions"] == 2
        assert summ_1["successful_executions"] == 1
        assert summ_1["failed_executions"] == 1
        assert summ_1["error_rate"] == 0.5
        assert summ_1["total_execution_time_ms"] == 40.0
        assert summ_1["avg_execution_time_ms"] == 20.0
        assert summ_1["total_prompt_tokens"] == 300
        assert summ_1["total_completion_tokens"] == 150
        assert summ_1["total_tokens"] == 450
        assert summ_1["total_cost_usd"] == 0.003
        assert "e1" in summ_1["by_edge"]
        assert summ_1["by_edge"]["e1"]["executions"] == 2
        assert summ_1["by_edge"]["e1"]["failed_executions"] == 1

        # Global summary
        global_summ = store.get_edge_metrics_summary(session_id=None)
        assert global_summ["total_executions"] == 3
        assert global_summ["successful_executions"] == 2
        assert global_summ["failed_executions"] == 1
        assert round(global_summ["error_rate"], 2) == 0.33
        assert global_summ["total_execution_time_ms"] == 100.0
        assert global_summ["total_tokens"] == 550
        assert "e1" in global_summ["by_edge"]
        assert "e2" in global_summ["by_edge"]

        # DB stats should include total_edge_metrics
        stats = store.get_db_stats()
        assert stats["total_edge_metrics"] == 3
        assert "sess_1" in store.list_sessions()
        assert "sess_2" in store.list_sessions()

        # clear_session purges metrics for that session
        store.clear_session("sess_1")
        assert len(store.list_edge_metrics("sess_1")) == 0
        assert len(store.list_edge_metrics("sess_2")) == 1
    finally:
        store.close()


@pytest.mark.asyncio
async def test_executor_v4_records_edge_metrics():
    """Verify ExecutorV4 automatically records metrics for executed edges."""
    store = VertexStoreV4(":memory:")
    session_id = "sess_exec_metrics"
    try:
        # Setup graph with start -> code edge -> end
        store.save_vertex(session_id, "start", content="hello", attributes=[VertexAttributeV4.START])
        store.save_vertex(session_id, "end", content="", attributes=[VertexAttributeV4.END])
        store.update_vertex_state(session_id, "start", VertexStateV4.DATA_READY.value)
        store.update_vertex_state(session_id, "end", VertexStateV4.TODO.value)

        graph = GraphV4(session_id=session_id)
        graph.add_vertex("start", attributes=[VertexAttributeV4.START])
        graph.add_vertex("end", attributes=[VertexAttributeV4.END])

        def transform(text: str) -> str:
            return f"{text}_transformed"

        edge = CodeEdgeV4("edge_trans", "start", "end", script=transform)
        graph.add_edge(edge)

        executor = ExecutorV4(graph=graph, store=store)
        res = await executor.run()

        assert res.success is True
        assert "end" in res.vertex_contents
        assert res.vertex_contents["end"] == "hello_transformed"

        # Check metrics recorded in SQLite
        metrics = store.list_edge_metrics(session_id=session_id)
        assert len(metrics) == 1
        m = metrics[0]
        assert m.edge_id == "edge_trans"
        assert m.edge_type == "code"
        assert m.input_vertex == "start"
        assert m.output_vertex == "end"
        assert m.execution_time_ms > 0.0
        assert m.success is True
        assert m.error is None

        # Check ExecutionResultV4 has metrics_summary populated
        assert res.metrics_summary is not None
        assert res.metrics_summary["total_executions"] == 1
        assert res.metrics_summary["successful_executions"] == 1
    finally:
        store.close()


@pytest.mark.asyncio
async def test_http_harness_executor_records_tool_and_code_metrics():
    """Verify HttpHarnessExecutorV4 records metrics for tool edges and code edges."""
    store = VertexStoreV4(":memory:")
    session_id = "sess_harness_metrics"
    try:
        store.save_vertex(session_id, "start", content="main.py", attributes=[VertexAttributeV4.START])
        store.save_vertex(session_id, "sandbox_out", content="")
        store.save_vertex(session_id, "end", content="", attributes=[VertexAttributeV4.END])
        store.update_vertex_state(session_id, "start", VertexStateV4.DATA_READY.value)
        store.update_vertex_state(session_id, "sandbox_out", VertexStateV4.TODO.value)
        store.update_vertex_state(session_id, "end", VertexStateV4.TODO.value)

        graph = GraphV4(session_id=session_id)
        graph.add_vertex("start", attributes=[VertexAttributeV4.START])
        graph.add_vertex("sandbox_out")
        graph.add_vertex("end", attributes=[VertexAttributeV4.END])

        tool_edge = ToolEdgeV4(
            edge_id="tool_ls",
            input_vertex="start",
            output_vertex="sandbox_out",
            tool_name="bash",
            arguments_template='{"command": "cat {input}"}',
        )
        graph.add_edge(tool_edge)

        code_edge = CodeEdgeV4(
            edge_id="code_summarize",
            input_vertex="sandbox_out",
            output_vertex="end",
            script=lambda s: f"Summary: {s.upper()}",
        )
        graph.add_edge(code_edge)

        http_exec = HttpHarnessExecutorV4(store=store)

        # Hop 1: Turn 1 triggers ToolEdgeV4 -> returns tool_call
        resp1 = await http_exec.step(
            session_id=session_id,
            graph=graph,
            messages=[{"role": "user", "content": "Analyze main.py"}],
        )
        assert resp1["choices"][0]["finish_reason"] == "tool_calls"
        tool_call = resp1["choices"][0]["message"]["tool_calls"][0]
        call_id = tool_call["id"]

        # Simulate delay
        await asyncio.sleep(0.01)

        # Hop 2: Turn 2 passes tool execution result back -> settles ToolEdgeV4, runs CodeEdgeV4 to terminal
        resp2 = await http_exec.step(
            session_id=session_id,
            graph=graph,
            messages=[
                {"role": "user", "content": "Analyze main.py"},
                resp1["choices"][0]["message"],
                {"role": "tool", "tool_call_id": call_id, "content": "print('hello world')"},
            ],
        )
        assert resp2["choices"][0]["finish_reason"] == "stop"
        assert "Summary: PRINT('HELLO WORLD')" in resp2["choices"][0]["message"]["content"]

        # Check recorded metrics
        metrics = store.list_edge_metrics(session_id=session_id)
        assert len(metrics) == 2
        edge_ids = {m.edge_id for m in metrics}
        assert "tool_ls" in edge_ids
        assert "code_summarize" in edge_ids

        tool_metric = next(m for m in metrics if m.edge_id == "tool_ls")
        assert tool_metric.edge_type == "tool"
        assert tool_metric.execution_time_ms >= 5.0  # Measured roundtrip from tool call issuance
        assert tool_metric.success is True

        code_metric = next(m for m in metrics if m.edge_id == "code_summarize")
        assert code_metric.edge_type == "code"
        assert code_metric.execution_time_ms > 0.0
        assert code_metric.success is True

        # Check summary
        summary = store.get_edge_metrics_summary(session_id=session_id)
        assert summary["total_executions"] == 2
        assert summary["successful_executions"] == 2
        assert summary["failed_executions"] == 0
    finally:
        store.close()


@pytest.mark.asyncio
async def test_server_metrics_endpoints():
    """Verify GET /api/sessions/{id}/metrics, GET /api/metrics/summary, and GET /api/db/stats."""
    store = VertexStoreV4(":memory:")
    app = create_v4_server(store)

    session_id = "sess_api_bench"
    # Seed metrics
    store.record_edge_metric(
        session_id=session_id,
        edge_id="edge_bench_1",
        edge_type="code",
        input_vertex="A",
        output_vertex="B",
        execution_time_ms=12.5,
        prompt_tokens=150,
        completion_tokens=50,
        total_tokens=200,
        cost_usd=0.0015,
        success=True,
    )
    store.record_edge_metric(
        session_id=session_id,
        edge_id="edge_bench_2",
        edge_type="llm",
        input_vertex="B",
        output_vertex="C",
        execution_time_ms=45.0,
        prompt_tokens=300,
        completion_tokens=100,
        total_tokens=400,
        cost_usd=0.0030,
        success=True,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. GET /api/sessions/{session_id}/metrics
        resp = await client.get(f"/api/sessions/{session_id}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["summary"]["total_executions"] == 2
        assert data["summary"]["total_tokens"] == 600
        assert len(data["metrics"]) == 2

        # 2. Filter by edge_id
        resp_filtered = await client.get(f"/api/sessions/{session_id}/metrics?edge_id=edge_bench_1")
        assert resp_filtered.status_code == 200
        filtered_data = resp_filtered.json()
        assert len(filtered_data["metrics"]) == 1
        assert filtered_data["metrics"][0]["edge_id"] == "edge_bench_1"

        # 3. GET /api/metrics/summary (global)
        resp_global = await client.get("/api/metrics/summary")
        assert resp_global.status_code == 200
        global_data = resp_global.json()
        assert global_data["total_executions"] >= 2
        assert global_data["total_tokens"] >= 600

        # 4. GET /api/db/stats includes total_edge_metrics
        resp_stats = await client.get("/api/db/stats")
        assert resp_stats.status_code == 200
        stats_data = resp_stats.json()
        assert stats_data["total_edge_metrics"] >= 2

        # 5. Verify OpenAI completion returns non-zero usage when metrics exist
        req_body = {
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "session_id": session_id,
        }
        resp_chat = await client.post("/v1/chat/completions", json=req_body)
        assert resp_chat.status_code == 200
        chat_data = resp_chat.json()
        assert "usage" in chat_data
        assert chat_data["usage"]["total_tokens"] >= 600
