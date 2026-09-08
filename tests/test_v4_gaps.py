import asyncio
import pytest

from framework.edge_v4 import CodeEdgeV4, LLMEdgeV4
from framework.executor_v4 import ExecutorV4
from framework.graph_v4 import GraphV4
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

@pytest.fixture
def mem_store() -> VertexStoreV4:
    """Fixture providing an in-memory SQLite store."""
    store = VertexStoreV4(":memory:")
    yield store
    store.close()

class MockAgent:
    def __init__(self, response="mocked response", should_fail=False):
        self.response = response
        self.should_fail = should_fail

    def chat(self, prompt, **kwargs):
        if self.should_fail:
            raise RuntimeError("Agent failure")
        return self.response

@pytest.mark.asyncio
async def test_llm_edge_with_mock_agent(mem_store: VertexStoreV4):
    session_id = "test_llm_edge"
    mem_store.save_vertex(session_id, "in_v", "input data", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(session_id, "out_v", "", state=VertexStateV4.TODO)

    mock_agent = MockAgent(response="processed: input data")
    
    edge = LLMEdgeV4(
        edge_id="llm_edge_1",
        input_vertex="in_v",
        output_vertex="out_v",
        prompt_template="Process: {input}",
        settings={}
    )

    res = await edge.run(session_id, mem_store, agent=mock_agent)
    
    assert res.success is True
    assert res.output == "processed: input data"

    out_v = mem_store.get_vertex(session_id, "out_v")
    assert out_v.content == "processed: input data"
    assert out_v.state == VertexStateV4.DATA_READY.value
    
    staged = mem_store.get_latest_staged_for_vertex(session_id, "out_v", key="rendered_prompt")
    assert staged is not None
    assert staged.value == "Process: input data"

@pytest.mark.asyncio
async def test_llm_edge_failure_stages_error(mem_store: VertexStoreV4):
    session_id = "test_llm_edge_fail"
    mem_store.save_vertex(session_id, "in_v", "input data", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(session_id, "out_v", "", state=VertexStateV4.TODO)

    mock_agent = MockAgent(should_fail=True)
    
    edge = LLMEdgeV4(
        edge_id="llm_edge_1",
        input_vertex="in_v",
        output_vertex="out_v",
        prompt_template="Process: {input}",
        settings={}
    )

    res = await edge.run(session_id, mem_store, agent=mock_agent)
    
    assert res.success is False
    assert "Agent failure" in (res.error or "")

    out_v = mem_store.get_vertex(session_id, "out_v")
    assert out_v.state == VertexStateV4.REJECT.value
    
    staged = mem_store.get_latest_staged_for_vertex(session_id, "out_v", key="error_feedback")
    assert staged is not None
    assert "Agent failure" in staged.value

@pytest.mark.asyncio
async def test_priority_scheduling_urgent_before_normal(mem_store: VertexStoreV4):
    session_id = "test_priority"
    graph = GraphV4(session_id)
    
    graph.add_vertex(VertexRecordV4(0, session_id, "A", "start data", [VertexAttributeV4.START.value], VertexStateV4.DATA_READY.value))
    graph.add_vertex(VertexRecordV4(0, session_id, "B", "", [], VertexStateV4.TODO_URGENT.value))
    graph.add_vertex(VertexRecordV4(0, session_id, "C", "", [], VertexStateV4.TODO.value))
    
    execution_order = []
    
    def make_fn(name: str):
        def fn(content, settings, staging):
            execution_order.append(name)
            return f"{content}_{name}"
        return fn
        
    e_ab = CodeEdgeV4("e_ab", "A", "B", script=make_fn("B"))
    e_ac = CodeEdgeV4("e_ac", "A", "C", script=make_fn("C"))
    
    graph.add_edge(e_ab)
    graph.add_edge(e_ac)
    graph.validate()
    
    executor = ExecutorV4(graph=graph, store=mem_store, max_concurrency=1)
    await executor.run()
    
    assert execution_order == ["B", "C"]

@pytest.mark.asyncio
async def test_concurrency_semaphore_enforcement(mem_store: VertexStoreV4):
    session_id = "test_concurrency"
    graph = GraphV4(session_id)
    
    graph.add_vertex(VertexRecordV4(0, session_id, "start", "init", [VertexAttributeV4.START.value], VertexStateV4.DATA_READY.value))
    for i in range(3):
        graph.add_vertex(VertexRecordV4(0, session_id, f"out_{i}", "", [], VertexStateV4.TODO.value))
    
    current_concurrent = 0
    max_concurrent_observed = 0
    lock = asyncio.Lock()
    
    async def async_process(content, settings, staging):
        nonlocal current_concurrent, max_concurrent_observed
        async with lock:
            current_concurrent += 1
            max_concurrent_observed = max(max_concurrent_observed, current_concurrent)
        
        await asyncio.sleep(0.1)
        
        async with lock:
            current_concurrent -= 1
        return "done"
        
    for i in range(3):
        graph.add_edge(CodeEdgeV4(f"e_{i}", "start", f"out_{i}", script=async_process))
        
    graph.validate()
    
    executor = ExecutorV4(graph=graph, store=mem_store, max_concurrency=2)
    await executor.run()
    
    assert max_concurrent_observed <= 2
    assert max_concurrent_observed > 0

@pytest.mark.asyncio
async def test_pruning_state_edge_skip(mem_store: VertexStoreV4):
    session_id = "test_pruning"
    mem_store.save_vertex(session_id, "in_v", "data", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(session_id, "out_v", "", state=VertexStateV4.PRUNING)
    
    edge = CodeEdgeV4("edge_skip", "in_v", "out_v", script=lambda c, s, st: "changed")
    res = await edge.run(session_id, mem_store)
    
    assert res.success is False
    assert res.skipped is True
    assert "required 'todo'" in (res.reason or "")

@pytest.mark.asyncio
async def test_forbidden_state_edge_skip(mem_store: VertexStoreV4):
    session_id = "test_forbidden"
    mem_store.save_vertex(session_id, "in_v", "data", state=VertexStateV4.DATA_READY)
    mem_store.save_vertex(session_id, "out_v", "", state=VertexStateV4.FORBIDDEN)
    
    edge = CodeEdgeV4("edge_skip", "in_v", "out_v", script=lambda c, s, st: "changed")
    res = await edge.run(session_id, mem_store)
    
    assert res.success is False
    assert res.skipped is True
    assert "required 'todo'" in (res.reason or "")

def test_state_validation_rejects_invalid(mem_store: VertexStoreV4):
    with pytest.raises(ValueError, match="Invalid state"):
        mem_store.save_vertex("session", "v1", "", state="running")

def test_attribute_validation_rejects_invalid(mem_store: VertexStoreV4):
    with pytest.raises(ValueError, match="Invalid attribute"):
        mem_store.save_vertex("session", "v1", "", attributes=["garbage_tag"])
