import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from framework.server_v4 import create_v4_server
from framework.vertex_v4 import VertexStoreV4


@pytest.fixture
def store():
    return VertexStoreV4(":memory:")


@pytest.fixture
def app_and_client(store, tmp_path):
    # Set the manifest_base_dir to the tmp_path for path validation
    app = create_v4_server(store)
    app.state.manifest_base_dir = tmp_path
    
    # Also we can use client to call it
    client = TestClient(app)
    return app, client, tmp_path


def test_path_traversal_sse_execute(app_and_client):
    app, client, tmp_path = app_and_client
    
    payloads = [
        "../../etc/passwd.json",
        "/tmp/evil.json",
        f"{tmp_path}/../outside.json",
        "../"
    ]
    
    for path in payloads:
        # manifest_path must end with .json or it will fail with a different error
        if not path.endswith('.json'):
            continue
            
        res = client.post("/api/sse/execute", json={
            "session_id": "test_session",
            "manifest_path": path
        })
        assert res.status_code == 400
        data = res.json()
        assert "detail" in data
        assert isinstance(data["detail"], dict)
        assert data["detail"].get("error_code") == "PATH_TRAVERSAL_REJECTED"


def test_path_traversal_graph_dump(app_and_client):
    app, client, tmp_path = app_and_client
    
    payloads = [
        "../../etc/passwd.json",
        "/tmp/evil.json",
        f"{tmp_path}/../outside.json",
    ]
    
    for path in payloads:
        res = client.post("/api/sessions/test_session/graph/dump", json={
            "path": path
        })
        assert res.status_code == 400
        data = res.json()
        assert "detail" in data
        assert isinstance(data["detail"], dict)
        assert data["detail"].get("error_code") == "PATH_TRAVERSAL_REJECTED"


def test_json_extension_enforcement(app_and_client):
    app, client, tmp_path = app_and_client
    
    # Should fail because it doesn't end in .json
    valid_path_wrong_ext = str(tmp_path / "valid_manifest.txt")
    
    res = client.post("/api/sse/execute", json={
        "session_id": "test_session",
        "manifest_path": valid_path_wrong_ext
    })
    assert res.status_code == 400
    assert "must end in .json" in res.json()["detail"].lower()

    # Same for dump endpoint
    res = client.post("/api/sessions/test_session/graph/dump", json={
        "path": valid_path_wrong_ext
    })
    assert res.status_code == 400
    assert "must end in .json" in res.json()["detail"].lower()


def test_pydantic_validation_vertex_states(app_and_client):
    app, client, _ = app_and_client
    
    # Invalid state
    res = client.post("/api/sessions/test_session/graph/vertices", json={
        "name": "v1",
        "state": "INVALID_STATE"
    })
    assert res.status_code == 422 # FastAPI validation error
    data = res.json()
    assert any("Invalid state" in err.get("msg", "") or "Input should be" in err.get("msg", "") for err in data["detail"])

    # Missing required field `name`
    res = client.post("/api/sessions/test_session/graph/vertices", json={
        "state": "idle"
    })
    assert res.status_code == 422
    data = res.json()
    assert any(err["loc"][-1] == "name" for err in data["detail"])

    # Invalid attributes
    res = client.post("/api/sessions/test_session/graph/vertices", json={
        "name": "v1",
        "attributes": ["INVALID_ATTR"]
    })
    assert res.status_code == 422


def test_pydantic_validation_edge_types(app_and_client):
    app, client, _ = app_and_client
    
    # Missing input_vertex and output_vertex
    res = client.post("/api/sessions/test_session/graph/edges", json={
        "id": "e1",
        "type": "code"
    })
    assert res.status_code == 422
    data = res.json()
    assert any(err["loc"][-1] == "input_vertex" for err in data["detail"])
    
    # Invalid edge type handled in route? Actually Pydantic allows any string for `type` but the route rejects it
    # Pydantic schema doesn't validate type, so it passes Pydantic but route raises ValueError
    with pytest.raises(ValueError, match="Unsupported edge type: INVALID_TYPE"):
        client.post("/api/sessions/test_session/graph/edges", json={
            "id": "e1",
            "input_vertex": "v1",
            "output_vertex": "v2",
            "type": "INVALID_TYPE"
        })

