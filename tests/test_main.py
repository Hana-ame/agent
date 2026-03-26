import pytest
from app.main import app

def test_root_endpoint():
    response = app.test_client().get('/')
    assert response.status_code == 200
    assert response.json == {"message": "Hello World"}