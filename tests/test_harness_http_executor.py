import pytest
import httpx
from httpx import ASGITransport

from framework.graph_v4 import GraphV4
from framework.vertex_v4 import VertexRecordV4, VertexStateV4, VertexAttributeV4, VertexStoreV4
from framework.edge_v4 import ToolEdgeV4, CodeEdgeV4
from framework.server_v4 import create_v4_server, SessionGraphManagerV4


@pytest.fixture
def store():
    return VertexStoreV4(":memory:")


@pytest.fixture
def manager(store):
    return SessionGraphManagerV4(store=store)


@pytest.fixture
def app(store, manager):
    return create_v4_server(store_or_db=store, manager=manager)


@pytest.mark.asyncio
async def test_harness_multi_tool_edge_lifecycle(app, store, manager):
    """Test full multi-turn harness interaction:
    1. Turn 1: Harness requests task -> gets ToolEdge 1 (ls)
    2. Turn 2: Harness returns ls output -> gets ToolEdge 2 (git status), content shows Edge 1 results
    3. Turn 3: Harness returns git status output -> graph completes, gets final answer with stop
    """
    session_id = "sess_harness_multi_test"
    graph = GraphV4(session_id=session_id, name="harness_graph")

    # Vertices: start -> v_files -> v_git -> final
    v_start = VertexRecordV4(
        id=0, session_id=session_id, name="start", content="",
        state=VertexStateV4.TODO.value, attributes=[VertexAttributeV4.START.value]
    )
    v_files = VertexRecordV4(
        id=0, session_id=session_id, name="files", content="",
        state=VertexStateV4.TODO.value
    )
    v_git = VertexRecordV4(
        id=0, session_id=session_id, name="git_status", content="",
        state=VertexStateV4.TODO.value
    )
    v_final = VertexRecordV4(
        id=0, session_id=session_id, name="final", content="",
        state=VertexStateV4.TODO.value, attributes=[VertexAttributeV4.END.value]
    )

    for v in (v_start, v_files, v_git, v_final):
        graph.add_vertex(v)
        store.save_vertex(
            session_id=session_id, name=v.name, content=v.content,
            attributes=v.attributes, state=v.state
        )

    # Edge 1: ToolEdge for 'ls'
    e_ls = ToolEdgeV4(
        edge_id="e_ls",
        input_vertex="start",
        output_vertex="files",
        tool_name="bash",
        arguments={"command": "ls -la"},
        priority=10,
    )
    # Edge 2: ToolEdge for 'git status'
    e_git = ToolEdgeV4(
        edge_id="e_git",
        input_vertex="files",
        output_vertex="git_status",
        tool_name="bash",
        arguments={"command": "git status"},
        priority=5,
    )
    # Edge 3: CodeEdge to summarize and write final answer
    e_summary = CodeEdgeV4(
        edge_id="e_summary",
        input_vertex="git_status",
        output_vertex="final",
        script=lambda content: f"Analysis complete: {content}",
        priority=1,
    )

    for e in (e_ls, e_git, e_summary):
        graph.add_edge(e)
        store.save_edge(
            session_id=session_id, edge_id=e.id, edge_type=e.type,
            input_vertex=e.input_vertex, output_vertex=e.output_vertex,
            settings=e.settings,
        )

    manager._graphs[session_id] = graph

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # -------------------------------------------------------------
        # Turn 1: Harness sends user prompt
        # -------------------------------------------------------------
        resp1 = await client.post("/v1/chat/completions", json={
            "session_id": session_id,
            "messages": [{"role": "user", "content": "Please diagnose repository"}],
        })
        assert resp1.status_code == 200
        data1 = resp1.json()
        choice1 = data1["choices"][0]
        assert choice1["finish_reason"] == "tool_calls"
        assert len(choice1["message"]["tool_calls"]) == 1

        tc1 = choice1["message"]["tool_calls"][0]
        assert tc1["function"]["name"] == "bash"
        assert "ls -la" in tc1["function"]["arguments"]

        # -------------------------------------------------------------
        # Turn 2: Harness executes ls -la in sandbox and returns tool output
        # -------------------------------------------------------------
        resp2 = await client.post("/v1/chat/completions", json={
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "Please diagnose repository"},
                choice1["message"],
                {
                    "role": "tool",
                    "tool_call_id": tc1["id"],
                    "content": "README.md src/ tests/ pyproject.toml"
                }
            ],
        })
        assert resp2.status_code == 200
        data2 = resp2.json()
        choice2 = data2["choices"][0]

        # Observability: message.content reports Edge 1 completion
        assert "Edge Completed: e_ls" in choice2["message"]["content"]
        assert "README.md" in choice2["message"]["content"]

        # Next tool call is Edge 2 (git status)
        assert choice2["finish_reason"] == "tool_calls"
        assert len(choice2["message"]["tool_calls"]) == 1
        tc2 = choice2["message"]["tool_calls"][0]
        assert tc2["function"]["name"] == "bash"
        assert "git status" in tc2["function"]["arguments"]

        # -------------------------------------------------------------
        # Turn 3: Harness executes git status and returns tool output
        # -------------------------------------------------------------
        resp3 = await client.post("/v1/chat/completions", json={
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "Please diagnose repository"},
                choice1["message"],
                {"role": "tool", "tool_call_id": tc1["id"], "content": "README.md src/ tests/ pyproject.toml"},
                choice2["message"],
                {
                    "role": "tool",
                    "tool_call_id": tc2["id"],
                    "content": "On branch main. Working tree clean."
                }
            ],
        })
        assert resp3.status_code == 200
        data3 = resp3.json()
        choice3 = data3["choices"][0]

        # Finish reason is stop — entire graph is complete!
        assert choice3["finish_reason"] == "stop"
        assert choice3["message"]["content"] is not None
        assert "Analysis complete: On branch main. Working tree clean." in choice3["message"]["content"]
        assert "Edge Completed: e_git" in choice3["message"]["content"]


@pytest.mark.asyncio
async def test_zero_config_session_binding(app, store, manager):
    """Test that standard OpenAI SDK client calls without custom session_id
    automatically bind to the same session via conversation fingerprinting.
    """
    root_prompt = "Unique task prompt for automated fingerprint binding"
    from framework.http_executor_v4 import HttpHarnessExecutorV4
    expected_session_id = HttpHarnessExecutorV4.derive_session_id([{"role": "user", "content": root_prompt}])

    # Setup simple graph under derived session id
    graph = GraphV4(session_id=expected_session_id, name="fingerprint_graph")
    src = VertexRecordV4(
        id=0, session_id=expected_session_id, name="src", content="",
        state=VertexStateV4.TODO.value, attributes=[VertexAttributeV4.START.value]
    )
    sink = VertexRecordV4(
        id=0, session_id=expected_session_id, name="sink", content="",
        state=VertexStateV4.TODO.value, attributes=[VertexAttributeV4.END.value]
    )
    graph.add_vertex(src)
    graph.add_vertex(sink)
    edge = ToolEdgeV4(
        edge_id="e_tool", input_vertex="src", output_vertex="sink",
        tool_name="bash", arguments={"command": "whoami"}
    )
    graph.add_edge(edge)

    manager._graphs[expected_session_id] = graph
    store.save_vertex(expected_session_id, "src", "", attributes=[VertexAttributeV4.START.value], state=VertexStateV4.TODO.value)
    store.save_vertex(expected_session_id, "sink", "", attributes=[VertexAttributeV4.END.value], state=VertexStateV4.TODO.value)
    store.save_edge(expected_session_id, "e_tool", "tool", "src", "sink")

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Turn 1: Client does NOT provide session_id or X-Session-ID header
        resp1 = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": root_prompt}],
        })
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["system_fingerprint"] == f"vea-v4-{expected_session_id}"
        tc = data1["choices"][0]["message"]["tool_calls"][0]

        # Turn 2: Client returns tool result with same conversation root
        resp2 = await client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "user", "content": root_prompt},
                data1["choices"][0]["message"],
                {"role": "tool", "tool_call_id": tc["id"], "content": "ubuntu"}
            ]
        })
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["choices"][0]["finish_reason"] == "stop"
        assert "ubuntu" in data2["choices"][0]["message"]["content"]
