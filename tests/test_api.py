from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "app" in data
    assert "version" in data


def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) >= 1
    # Verify model structure
    first = data["models"][0]
    assert "id" in first
    assert "display_name" in first
    assert "parameters" in first


def test_system_status_endpoint():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "ram_used_gb" in data
    assert "ram_total_gb" in data


def test_chat_completions_invalid_payload():
    # Empty messages list should fail validation or be handled
    response = client.post("/api/chat/completions", json={})
    assert response.status_code == 422  # Unprocessable Entity
