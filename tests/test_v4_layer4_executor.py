import asyncio
import json
import logging
import pytest
import time
from framework.graph_v4 import GraphV4
from framework.executor_v4 import ExecutorV4
from framework.vertex_v4 import VertexStoreV4, VertexStateV4, VertexAttributeV4
from framework.edge_v4 import EdgeV4, CodeEdgeV4

logger = logging.getLogger(__name__)

# A simple mock agent for LLM tests if needed
class MockAgent:
    def chat(self, prompt, **kwargs):
        return f"Mock reply to: {prompt}"

@pytest.mark.asyncio
async def test_semaphore_boundary_enforcement():
    """Verify max_concurrency limits parallel execution."""
    store = VertexStoreV4(":memory:")
    graph = GraphV4("sess_concurrent")
    
    # We will create 50 edges that all can run immediately.
    # To do this, we create 50 input vertices already in DATA_READY,
    # and 50 output vertices in TODO.
    graph.add_vertex("start_v", state=VertexStateV4.DATA_READY.value)
    for i in range(20):
        graph.add_vertex(f"in_{i}", state=VertexStateV4.DATA_READY.value, content=f"{i}")
        graph.add_vertex(f"out_{i}", state=VertexStateV4.TODO.value)
        
        async def process_fn(content, settings, staging=None):
            tracker = settings['tracker']
            tracker['current'] += 1
            if tracker['current'] > tracker['max_observed']:
                tracker['max_observed'] = tracker['current']
            await asyncio.sleep(0.05)
            tracker['current'] -= 1
            return content
        
        graph.add_edge(CodeEdgeV4(
            edge_id=f"edge_{i}",
            input_vertex=f"in_{i}",
            output_vertex=f"out_{i}",
            script=process_fn,
            settings={"priority": 0}
        ))
        
    tracker = {"current": 0, "max_observed": 0}
    for e in graph.edges.values():
        e.settings["tracker"] = tracker
        
    executor = ExecutorV4(graph, store, max_concurrency=4)
    res = await executor.run()
    
    assert res.success is True
    assert tracker["max_observed"] <= 4
    assert len(res.completed_edges) == 20

@pytest.mark.asyncio
async def test_priority_and_tier_preservation():
    """Verify urgency, priority, and tiers are respected in scheduling."""
    store = VertexStoreV4(":memory:")
    graph = GraphV4("sess_priority")
    
    # Tier 0 nodes
    graph.add_vertex("t0_in", state=VertexStateV4.DATA_READY.value)
    
    # Target nodes with different urgencies
    graph.add_vertex("t0_out_urgent", state=VertexStateV4.TODO_URGENT.value)
    graph.add_vertex("t0_out_normal", state=VertexStateV4.TODO.value)
    
    execution_order = []
    
    def track_order(content, settings, staging):
        execution_order.append(settings["label"])
        return "done"

    e1 = CodeEdgeV4("edge_normal", "t0_in", "t0_out_normal", script=track_order, settings={"label": "normal", "priority": 10})
    e2 = CodeEdgeV4("edge_urgent", "t0_in", "t0_out_urgent", script=track_order, settings={"label": "urgent", "priority": 0})
    
    graph.add_edge(e1)
    graph.add_edge(e2)
    
    # Urgent has lower priority but higher urgency state (TODO_URGENT). Urgency > Priority > Tier.
    # Therefore, urgent should run first.
    executor = ExecutorV4(graph, store, max_concurrency=1)
    await executor.run()
    
    assert execution_order == ["urgent", "normal"]

@pytest.mark.asyncio
async def test_fan_in_accumulation():
    """Verify JSON_MERGE and LIST_APPEND strategies under concurrent execution."""
    store = VertexStoreV4(":memory:")
    graph = GraphV4("sess_fan_in")
    
    graph.add_vertex("in_1", state=VertexStateV4.DATA_READY.value, content='{"a": 1}')
    graph.add_vertex("in_2", state=VertexStateV4.DATA_READY.value, content='{"b": 2}')
    graph.add_vertex("out_merge", state=VertexStateV4.TODO.value, content='{"base": 0}')
    
    e1 = CodeEdgeV4("edge_merge_1", "in_1", "out_merge", settings={"merge_strategy": "json_merge"})
    e2 = CodeEdgeV4("edge_merge_2", "in_2", "out_merge", settings={"merge_strategy": "json_merge"})
    
    graph.add_edge(e1)
    graph.add_edge(e2)
    
    executor = ExecutorV4(graph, store, max_concurrency=2)
    res = await executor.run()
    
    assert res.success is True
    
    out_v = store.get_vertex("sess_fan_in", "out_merge")
    out_dict = json.loads(out_v.content)
    assert out_dict["base"] == 0
    assert out_dict["a"] == 1
    assert out_dict["b"] == 2
    assert out_v.state == VertexStateV4.DATA_READY.value
    
    # List append
    graph2 = GraphV4("sess_list")
    graph2.add_vertex("l_in_1", state=VertexStateV4.DATA_READY.value, content='"item1"')
    graph2.add_vertex("l_in_2", state=VertexStateV4.DATA_READY.value, content='"item2"')
    graph2.add_vertex("l_out", state=VertexStateV4.TODO.value, content='[]')
    
    e3 = CodeEdgeV4("edge_list_1", "l_in_1", "l_out", settings={"merge_strategy": "list_append"})
    e4 = CodeEdgeV4("edge_list_2", "l_in_2", "l_out", settings={"merge_strategy": "list_append"})
    
    graph2.add_edge(e3)
    graph2.add_edge(e4)
    
    exec2 = ExecutorV4(graph2, store, max_concurrency=2)
    await exec2.run()
    
    l_out = store.get_vertex("sess_list", "l_out")
    l_list = json.loads(l_out.content)
    assert "item1" in l_list
    assert "item2" in l_list
    assert l_out.state == VertexStateV4.DATA_READY.value

@pytest.mark.asyncio
async def test_task_exception_shielding():
    """Verify that exceptions in edges are caught and do not crash the executor."""
    store = VertexStoreV4(":memory:")
    graph = GraphV4("sess_err")
    
    graph.add_vertex("in1", state=VertexStateV4.DATA_READY.value)
    graph.add_vertex("out1", state=VertexStateV4.TODO.value)
    
    def fail_func(c, s, st):
        raise ValueError("Intentional crash")
        
    e1 = CodeEdgeV4("edge_fail", "in1", "out1", script=fail_func)
    graph.add_edge(e1)
    
    executor = ExecutorV4(graph, store)
    res = await executor.run()
    
    # Should complete without throwing, but success is False since target vertex became REJECT
    assert res.success is False
    assert len(res.errors) > 0
    assert "Intentional crash" in str(res.errors)
    
    v = store.get_vertex("sess_err", "out1")
    assert v.state == VertexStateV4.REJECT.value

@pytest.mark.asyncio
async def test_event_driven_scheduling():
    """Verify that the executor uses event wait and doesn't poll aggressively."""
    store = VertexStoreV4(":memory:")
    graph = GraphV4("sess_event")
    
    graph.add_vertex("v_in", state=VertexStateV4.DATA_READY.value)
    graph.add_vertex("v_out", state=VertexStateV4.TODO.value)
    
    async def slow_func(c, s, st):
        await asyncio.sleep(0.5)
        return "slow"
        
    e1 = CodeEdgeV4("edge_slow", "v_in", "v_out", script=slow_func)
    graph.add_edge(e1)
    
    # Set a huge scan interval. If it polling-based, this would take 10 seconds after task finishes.
    # With event-driven, it should finish immediately after task completes.
    executor = ExecutorV4(graph, store, scan_interval=10.0)
    
    start = time.time()
    res = await executor.run()
    duration = time.time() - start
    
    assert res.success is True
    # If events work properly, it shouldn't hit the 10-second timeout wait
    assert duration < 5.0 


@pytest.mark.asyncio
async def test_executor_graph_and_store_cache_coherence():
    """Verify that in-memory graph.vertices and SQLite store never diverge during execution (Issue 5 fix)."""
    store = VertexStoreV4(":memory:")
    graph = GraphV4("sess_coherence")

    graph.add_vertex("v_source", state=VertexStateV4.DATA_READY.value, content="initial_data")
    graph.add_vertex("v_target", state=VertexStateV4.TODO.value, content="")

    def transform_fn(content, settings, staging=None):
        return f"transformed_{content}"

    e1 = CodeEdgeV4("edge_trans", "v_source", "v_target", script=transform_fn)
    graph.add_edge(e1)

    executor = ExecutorV4(graph, store)

    # Before run: in-memory vertex is empty and todo
    assert graph.get_vertex("v_target").content == ""
    assert graph.get_vertex("v_target").state == VertexStateV4.TODO.value

    # Run workflow
    res = await executor.run()
    assert res.success is True

    # Check store
    db_v = store.get_vertex("sess_coherence", "v_target")
    assert db_v is not None
    assert db_v.state == VertexStateV4.DATA_READY.value
    assert db_v.content == "transformed_initial_data"

    # Check in-memory graph: MUST be in 100% coherence with store (NO dual-source divergence!)
    mem_v = executor.graph.get_vertex("v_target")
    assert mem_v is not None
    assert mem_v.state == VertexStateV4.DATA_READY.value
    assert mem_v.content == "transformed_initial_data"
    assert mem_v.state == db_v.state
    assert mem_v.content == db_v.content

