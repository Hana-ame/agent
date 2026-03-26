from fastapi.testclient import TestClient
from fastapi_todo import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Todo API"}

def test_create_todo():
    response = client.post("/todos", json={"title": "Test Todo", "description": "Test"})
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Test Todo"
    assert data["id"] is not None

def test_list_todos():
    response = client.get("/todos")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_todo():
    # 先创建
    create_resp = client.post("/todos", json={"title": "Get Test"})
    todo_id = create_resp.json()["id"]
    response = client.get(f"/todos/{todo_id}")
    assert response.status_code == 200
    assert response.json()["title"] == "Get Test"

def test_update_todo():
    create_resp = client.post("/todos", json={"title": "Update Test"})
    todo_id = create_resp.json()["id"]
    response = client.put(f"/todos/{todo_id}", json={"title": "Updated", "description": "New"})
    assert response.status_code == 200
    assert response.json()["title"] == "Updated"

def test_delete_todo():
    create_resp = client.post("/todos", json={"title": "Delete Test"})
    todo_id = create_resp.json()["id"]
    response = client.delete(f"/todos/{todo_id}")
    assert response.status_code == 204
    # 确认已删除
    get_resp = client.get(f"/todos/{todo_id}")
    assert get_resp.status_code == 404
