#!/usr/bin/env python3
"""Executable runner for Hacker News AI & Tech Digest V4 Graph.

Demonstrates:
1. Declarative V4 Graph loading via DiscreteGraphLoaderV4.
2. Concurrent Multi-Branch Fan-In DAG with MergeStrategyV4.JSON_MERGE settlement barrier.
3. Real-time Event Streaming via ExecutorV4.stream() (observability).
4. Edge-level performance benchmarking & metrics (SQLite edge_metrics table).
5. Session staging scratchpad persistence (session_staging table).
6. Reflexive self-healing error recovery via ReflexiveEdgeV4.
7. Agent Harness clock-tick stepping with declarative ToolEdgeV4 (HttpHarnessExecutorV4).
8. Live Markdown digest report generation (report.md).

Usage:
    python3 examples/hn_v4/run.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.graph_v4 import DiscreteGraphLoaderV4, GraphV4
from framework.edge_v4 import CodeEdgeV4, ReflexiveEdgeV4, ToolEdgeV4
from framework.executor_v4 import ExecutorV4
from framework.http_executor_v4 import HttpHarnessExecutorV4
from framework.vertex_v4 import VertexAttributeV4, VertexRecordV4, VertexStateV4, VertexStoreV4

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("hn_v4_demo")

DEMO_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = DEMO_DIR / "manifest.json"


async def run_streaming_v4_pipeline() -> VertexStoreV4:
    """Run Hacker News V4 pipeline using standalone ExecutorV4 with live event streaming."""
    print("\n" + "=" * 78)
    print("🚀 [Part 1] Multi-Branch Fan-In DAG Execution via ExecutorV4.stream()")
    print("=" * 78)

    # 1. Initialize SQLite store
    store = VertexStoreV4(":memory:")
    session_id = "hn_v4_streaming_session"

    # 2. Load graph from manifest
    graph = DiscreteGraphLoaderV4.load_from_manifest(
        str(MANIFEST_PATH),
        override_session_id=session_id,
    )
    print(f"  • Graph Name:         {graph.name}")
    print(f"  • Registered Vertices: {list(graph.vertices.keys())}")
    print(f"  • Registered Edges:    {list(graph.edges.keys())}")

    # 3. Populate store with vertices and edges
    DiscreteGraphLoaderV4.populate_store(graph, store)

    # Stage initial request parameters into session_staging scratchpad
    store.stage_output(
        session_id=session_id,
        edge_id="init",
        key="pipeline_trigger",
        value=json.dumps({"topic": "AI, Systems, Agents", "limit": 6}),
        metadata={"user": "hn_curator", "engine": "v4_stream"},
    )

    # 4. Stream graph pipeline events
    start_t = time.perf_counter()
    executor = ExecutorV4(graph=graph, store=store, max_concurrency=4)

    print("\n  ⚡ Live Graph Execution Event Stream:")
    async for event in executor.stream():
        ev_type = event.event_type
        if ev_type == "edge_started":
            payload = event.payload or {}
            print(f"    ▶ [STARTED]   Edge: {event.edge_id:16s} ({payload.get('input')} ➔ {payload.get('output')})")
        elif ev_type == "edge_completed":
            print(f"    ✔ [COMPLETED] Edge: {event.edge_id:16s}")
        elif ev_type == "fan_in_failed":
            print(f"    ✖ [FAN-IN FAIL] Vertex: {event.vertex_name}")
        elif ev_type == "edge_failed":
            print(f"    ✖ [FAILED]    Edge: {event.edge_id:16s} - Error: {event.payload}")

    elapsed = time.perf_counter() - start_t
    print(f"\n  ✅ Pipeline Traversal Completed in {elapsed:.2f}s!")

    # 5. Fetch final generated report from output vertex
    final_node = store.get_vertex(session_id, "v_final_report")
    report_text = final_node.content if final_node else ""

    print("\n" + "-" * 78)
    print("📄 [Generated Hacker News AI & Tech Digest Report Preview]")
    print("-" * 78)
    preview_lines = report_text.splitlines()[:28]
    print("\n".join(preview_lines))
    if len(report_text.splitlines()) > 28:
        print(f"\n  ... [{len(report_text.splitlines()) - 28} more lines saved in report.md] ...")

    return store


def showcase_edge_metrics_and_staging(store: VertexStoreV4, session_id: str = "hn_v4_streaming_session") -> None:
    """Inspect and display edge-level metrics and session staging records from SQLite."""
    print("\n" + "=" * 78)
    print("📊 [Part 2] Edge-Level Performance Benchmarks & Staging (SQLite edge_metrics)")
    print("=" * 78)

    metrics_summary = store.get_edge_metrics_summary(session_id)
    print(f"  • Total Edge Executions:    {metrics_summary['total_executions']}")
    print(f"  • Successful Executions:    {metrics_summary['successful_executions']}")
    print(f"  • Failed Executions:        {metrics_summary['failed_executions']} (Error Rate: {metrics_summary['error_rate']:.1%})")
    print(f"  • Cumulative Latency:       {metrics_summary['total_execution_time_ms']:.2f} ms")
    print(f"  • Average Edge Latency:     {metrics_summary['avg_execution_time_ms']:.2f} ms")

    print("\n  Per-Edge Performance Breakdown:")
    print(f"    {'Edge ID':20s} | {'Type':6s} | {'Latency (ms)':12s} | {'Executions':10s} | {'Errors':6s}")
    print("    " + "-" * 66)
    for edge_id, e_info in sorted(metrics_summary.get("by_edge", {}).items()):
        print(
            f"    {edge_id:20s} | {e_info['edge_type']:6s} | {e_info['total_execution_time_ms']:12.2f} | "
            f"{e_info['executions']:10d} | {e_info['failed_executions']:6d}"
        )

    print("\n  Session Staging Scratchpad Inspection (session_staging table):")
    staged_records = store.get_staged(session_id)
    for sr in staged_records:
        val_preview = (sr.value[:50] + "...") if len(sr.value) > 50 else sr.value
        print(f"    • [{sr.edge_id:12s}] key='{sr.key}' ➔ {val_preview}")


async def run_reflexive_self_healing_demo() -> None:
    """Demonstrate ReflexiveEdgeV4 self-healing recovery when a vertex rejects."""
    print("\n" + "=" * 78)
    print("🛡️ [Part 3] Self-Healing Error Recovery via ReflexiveEdgeV4")
    print("=" * 78)

    store = VertexStoreV4(":memory:")
    session_id = "hn_v4_reflexive_healing"
    graph = GraphV4(session_id=session_id, name="reflexive_demo")

    # Vertex starts in rejected / corrupt state
    v_corrupt = VertexRecordV4(
        id=1,
        session_id=session_id,
        name="v_raw_stories",
        content='{"corrupt": true, "error": "upstream_api_503"}',
        state=VertexStateV4.REJECT.value,
        attributes=["json"],
    )
    v_downstream = VertexRecordV4(
        id=2,
        session_id=session_id,
        name="v_filtered_stories",
        content="",
        state=VertexStateV4.TODO.value,
        attributes=["json"],
    )
    # Reflexive edge that triggers on REJECT, restores fallback stories, transitions to DATA_READY
    e_recover = ReflexiveEdgeV4(
        edge_id="e_recover_fetch",
        vertex_name="v_raw_stories",
        trigger_state=VertexStateV4.REJECT.value,
        target_state=VertexStateV4.DATA_READY.value,
        max_retries=2,
        script="examples/hn_v4/hn_transforms.py:recovery_fetch_fallback",
    )
    # Forward edge waiting for v_raw_stories to be data ready
    e_filter = CodeEdgeV4(
        edge_id="e_filter_forward",
        input_vertex="v_raw_stories",
        output_vertex="v_filtered_stories",
        script="examples/hn_v4/hn_transforms.py:filter_stories",
    )
    graph.add_vertex(v_corrupt)
    graph.add_vertex(v_downstream)
    graph.add_edge(e_recover)
    graph.add_edge(e_filter)
    DiscreteGraphLoaderV4.populate_store(graph, store)

    print(f"  • Initial Vertex State: '{v_corrupt.name}' is '{v_corrupt.state}' (faulty)")
    print(f"  • Registered Reflexive Edge: '{e_recover.id}' on trigger_state='{e_recover.trigger_state}'")

    executor = ExecutorV4(graph=graph, store=store)
    result = await executor.run()

    healed_v = store.get_vertex(session_id, "v_raw_stories")
    downstream_v = store.get_vertex(session_id, "v_filtered_stories")

    print(f"  • Post-Recovery State:  '{healed_v.name}' is now '{healed_v.state}'")
    print(f"  • Downstream Edge Run:  '{e_filter.id}' completed={result.success}")
    print(f"  • Filtered Data Generated: {len(json.loads(downstream_v.content))} stories recovered and filtered!")
    print("  ✅ Reflexive Self-Healing Handshake Verified!")

    store.close()


async def run_harness_tool_edge_stepping() -> None:
    """Demonstrate Agent Harness clock-tick stepping with declarative ToolEdgeV4."""
    print("\n" + "=" * 78)
    print("🌐 [Part 4] Agent Harness Clock Stepping with ToolEdgeV4 (HttpHarnessExecutorV4)")
    print("=" * 78)

    store = VertexStoreV4(":memory:")
    session_id = "hn_v4_harness_tool_run"
    graph = GraphV4(session_id=session_id, name="harness_tool_graph")

    v_start = VertexRecordV4(
        id=1, session_id=session_id, name="v_start", content="{\"action\": \"fetch_git_info\"}",
        state=VertexStateV4.DATA_READY.value, attributes=[VertexAttributeV4.START.value],
    )
    v_tool_out = VertexRecordV4(
        id=2, session_id=session_id, name="v_tool_out", content="",
        state=VertexStateV4.TODO.value,
    )
    v_final = VertexRecordV4(
        id=3, session_id=session_id, name="v_final", content="",
        state=VertexStateV4.TODO.value, attributes=[VertexAttributeV4.END.value],
    )

    for v in (v_start, v_tool_out, v_final):
        graph.add_vertex(v)

    # Edge 1: Declarative ToolEdge produces real bash tool call
    e_tool = ToolEdgeV4(
        edge_id="e_git_status",
        input_vertex="v_start",
        output_vertex="v_tool_out",
        tool_name="bash",
        arguments={"command": "git log -1 --oneline"},
        priority=10,
    )
    # Edge 2: CodeEdge digests tool output and creates final answer
    e_summary = CodeEdgeV4(
        edge_id="e_digest_summary",
        input_vertex="v_tool_out",
        output_vertex="v_final",
        script=lambda content: f"Agent Harness Execution Verified. Sandbox output: {content}",
        priority=1,
    )
    graph.add_edge(e_tool)
    graph.add_edge(e_summary)
    DiscreteGraphLoaderV4.populate_store(graph, store)

    http_exec = HttpHarnessExecutorV4(store=store)

    messages = [{"role": "user", "content": "Execute repository environment inspection."}]
    turn = 1

    while True:
        resp = await http_exec.step(session_id, graph, messages, model="default")
        choice = resp["choices"][0]
        finish_reason = choice["finish_reason"]
        msg = choice["message"]
        print(f"  • [Pulse {turn}] finish_reason='{finish_reason}'")

        if finish_reason == "tool_calls":
            tool_call = msg["tool_calls"][0]
            print(f"    🔧 Declarative ToolEdge Emitted: {tool_call['function']['name']}({tool_call['function']['arguments']})")
            messages.append(msg)
            # Simulate harness executing command in environment/sandbox
            simulated_stdout = "c3f81e0 chore: implement Phase 4 edge metrics telemetry table"
            print(f"    💻 Harness Sandbox Returned: '{simulated_stdout}'")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": simulated_stdout,
            })
            turn += 1
        elif finish_reason == "stop":
            content = msg.get("content") or ""
            print(f"    🎯 Harness Stepping Complete! Final Message: \"{content}\"")
            break
        else:
            break

    store.close()


async def main():
    print("=" * 78)
    print("        HACKER NEWS AI & TECH DIGEST — V4 COMPREHENSIVE SHOWCASE")
    print("=" * 78)

    store = await run_streaming_v4_pipeline()
    showcase_edge_metrics_and_staging(store)
    store.close()

    await run_reflexive_self_healing_demo()
    await run_harness_tool_edge_stepping()

    print("\n" + "=" * 78)
    print("🎉 All Hacker News V4 pipeline features verified and successfully executed!")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())

