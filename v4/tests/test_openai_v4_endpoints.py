import json
import pytest
import httpx
from httpx import ASGITransport

from framework.server_v4 import create_v4_server, SessionGraphManagerV4
from framework.vertex_v4 import VertexStoreV4, VertexRecordV4, VertexStateV4, VertexAttributeV4
from framework.graph_v4 import GraphV4
from framework.edge_v4 import CodeEdgeV4


@pytest.fixture
def store():
    return VertexStoreV4(":memory:")


@pytest.fixture
def manager(store):
    return SessionGraphManagerV4(store=store)


@pytest.fixture
def app(store, manager):
    return create_v4_server(store_or_db=store, manager=manager)


def setup_simple_graph(store, manager, session_id):
    graph = GraphV4(session_id=session_id, name="test")
    src = VertexRecordV4(id=0, session_id=session_id, name="src", content="", 
                         state=VertexStateV4.TODO.value, attributes=[VertexAttributeV4.START.value])
    sink = VertexRecordV4(id=0, session_id=session_id, name="sink", content="",
                          state=VertexStateV4.TODO.value, attributes=[VertexAttributeV4.END.value])
    graph.add_vertex(src)
    graph.add_vertex(sink)
    
    def process_func(content: str, **kwargs) -> str:
        return f"processed: {content}"
        
    edge = CodeEdgeV4(edge_id="e1", input_vertex="src", output_vertex="sink", 
                      script=process_func)
    graph.add_edge(edge)
    
    manager._graphs[session_id] = graph
    store.save_vertex(session_id=session_id, name="src", content="",
                      attributes=[VertexAttributeV4.START.value], state=VertexStateV4.TODO.value)
    store.save_vertex(session_id=session_id, name="sink", content="",
                      attributes=[VertexAttributeV4.END.value], state=VertexStateV4.TODO.value)
    store.save_edge(session_id=session_id, edge_id="e1", edge_type="code",
                    input_vertex="src", output_vertex="sink")
    
    return graph

@pytest.mark.asyncio
async def test_v1_models_endpoint(app):
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert any(model["id"] == "default" for model in data["data"])

@pytest.mark.asyncio
async def test_v1_models_with_active_session(app, manager):
    # Simulate an active session
    manager.get_or_create_graph("test_session_123")
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert any(model["id"] == "test_session_123" for model in data["data"])

@pytest.mark.asyncio
async def test_chat_completions_empty_messages(app):
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 422

@pytest.mark.asyncio
async def test_chat_completions_full_mode(app, store, manager):
    session_id = "test_full_mode"
    setup_simple_graph(store, manager, session_id)
    
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "vea_execution_mode": "full",
        "session_id": session_id
    }
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        choice = data["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == "processed: hello"

@pytest.mark.asyncio
async def test_chat_completions_per_tier_mode(app, store, manager):
    session_id = "test_tier_mode"
    setup_simple_graph(store, manager, session_id)
    
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "vea_execution_mode": "per_tier",
        "session_id": session_id
    }
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp1 = await client.post("/v1/chat/completions", json=payload)
        assert resp1.status_code == 200
        data1 = resp1.json()
        choice1 = data1["choices"][0]
        
        if choice1["finish_reason"] == "tool_calls":
            tool_calls = choice1["message"]["tool_calls"]
            assert len(tool_calls) == 1
            assert tool_calls[0]["function"]["name"] == "advance_graph"
            
            resp2 = await client.post("/v1/chat/completions", json={
                "model": "default",
                "messages": [{"role": "user", "content": "hello"}],
                "vea_execution_mode": "per_tier",
                "session_id": session_id
            })
            data2 = resp2.json()
            choice2 = data2["choices"][0]
            assert choice2["finish_reason"] == "stop"
            assert choice2["message"]["content"] == "processed: hello"
        else:
            assert choice1["finish_reason"] == "stop"
            assert choice1["message"]["content"] == "processed: hello"

@pytest.mark.asyncio
async def test_chat_completions_session_persistence(app, store, manager):
    session_id = "test_persistence"
    setup_simple_graph(store, manager, session_id)
    
    payload1 = {
        "messages": [{"role": "user", "content": "hello"}],
        "session_id": session_id
    }
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/v1/chat/completions", json=payload1)
        
        v = store.get_vertex(session_id, "sink")
        assert v.content == "processed: hello"
        
        assert session_id in manager._graphs

@pytest.mark.asyncio
async def test_chat_completions_session_from_header(app, store, manager):
    session_id = "test_header"
    setup_simple_graph(store, manager, session_id)
    
    payload = {
        "messages": [{"role": "user", "content": "hello"}]
    }
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=payload, headers={"x-session-id": session_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "processed: hello"

@pytest.mark.asyncio
async def test_chat_completions_streaming(app, store, manager):
    session_id = "test_stream"
    setup_simple_graph(store, manager, session_id)
    
    payload = {
        "messages": [{"role": "user", "content": "stream"}],
        "stream": True,
        "session_id": session_id
    }
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            chunks = []
            async for chunk in response.aiter_lines():
                if chunk and chunk.startswith("data: ") and chunk != "data: [DONE]":
                    data = json.loads(chunk[6:])
                    chunks.append(data)
                    
            assert len(chunks) > 0
            has_stop = any(c["choices"][0].get("finish_reason") == "stop" for c in chunks)
            assert has_stop

@pytest.mark.asyncio
async def test_graph_template_registry(app, store, manager):
    app.state.register_graph_template(
        "my_test_model", 
        config={
            "vertices": [
                {"name": "src", "type": "input", "attributes": ["start"]},
                {"name": "sink", "type": "output", "attributes": ["end"]}
            ],
            "edges": [
                {"id": "e1", "type": "code", "input_vertex": "src", "output_vertex": "sink", "script": "lambda x: 'templated: ' + x"}
            ]
        }
    )
    
    payload = {
        "model": "my_test_model",
        "messages": [{"role": "user", "content": "hello"}],
        "vea_execution_mode": "full",
        "session_id": "test_registry"
    }
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "hello"
