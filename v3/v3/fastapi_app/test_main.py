from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)

def test_create_user():
    response = client.post("/users", json={"username": "testuser", "password": "testpass"})
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "testuser"
    assert "id" in data

def test_login():
    # Create user first
    client.post("/users", json={"username": "testuser2", "password": "testpass"})
    response = client.post("/token", data={"username": "testuser2", "password": "testpass"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    return response.json()["access_token"]

def test_create_todo():
    token = test_login()
    response = client.post(
        "/todos",
        json={"title": "Test Todo", "description": "Test"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Todo"
    assert data["completed"] == False
