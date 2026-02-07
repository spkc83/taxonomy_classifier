import pytest
from fastapi.testclient import TestClient

from taxonomy_framework.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_version(self):
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.1.0"

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert set(data.keys()) == {"status", "version"}


class TestHealthEndpointV1:
    def test_health_v1_returns_200(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_v1_returns_healthy_status(self):
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"
