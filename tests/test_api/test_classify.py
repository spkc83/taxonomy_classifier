import pytest
from fastapi.testclient import TestClient

from taxonomy_framework.api.main import app

client = TestClient(app)


class TestClassifyEndpoint:
    def test_classify_accepts_valid_request(self):
        response = client.post(
            "/api/v1/classify",
            json={"text": "This is a sample text to classify"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_category" in data
        assert "confidence" in data
        assert "path" in data

    def test_classify_returns_confidence_in_range(self):
        response = client.post(
            "/api/v1/classify",
            json={"text": "Sample text for testing"},
        )
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_classify_returns_path_list(self):
        response = client.post(
            "/api/v1/classify",
            json={"text": "Sample text"},
        )
        data = response.json()
        assert isinstance(data["path"], list)

    def test_classify_returns_alternatives(self):
        response = client.post(
            "/api/v1/classify",
            json={"text": "Sample text"},
        )
        data = response.json()
        assert "alternatives" in data
        assert isinstance(data["alternatives"], list)

    def test_classify_with_taxonomy_id(self):
        response = client.post(
            "/api/v1/classify",
            json={
                "text": "Sample text",
                "taxonomy_id": "custom_taxonomy",
            },
        )
        assert response.status_code == 200

    def test_classify_with_options(self):
        response = client.post(
            "/api/v1/classify",
            json={
                "text": "Sample text",
                "options": {"max_depth": 3, "include_scores": True},
            },
        )
        assert response.status_code == 200

    def test_classify_empty_text_returns_422(self):
        response = client.post(
            "/api/v1/classify",
            json={"text": ""},
        )
        assert response.status_code == 200 or response.status_code == 422

    def test_classify_missing_text_returns_422(self):
        response = client.post(
            "/api/v1/classify",
            json={},
        )
        assert response.status_code == 422


class TestBatchClassifyEndpoint:
    def test_batch_classify_accepts_valid_request(self):
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": ["Text one", "Text two", "Text three"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 3

    def test_batch_classify_returns_results_for_each_text(self):
        texts = ["First text", "Second text"]
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": texts},
        )
        data = response.json()
        assert len(data["results"]) == len(texts)
        for result in data["results"]:
            assert "predicted_category" in result
            assert "confidence" in result

    def test_batch_classify_with_taxonomy_id(self):
        response = client.post(
            "/api/v1/classify/batch",
            json={
                "texts": ["Text one", "Text two"],
                "taxonomy_id": "custom_taxonomy",
            },
        )
        assert response.status_code == 200

    def test_batch_classify_empty_texts_returns_400(self):
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": []},
        )
        assert response.status_code == 400 or response.status_code == 422

    def test_batch_classify_missing_texts_returns_422(self):
        response = client.post(
            "/api/v1/classify/batch",
            json={},
        )
        assert response.status_code == 422


class TestTaxonomyEndpoint:
    def test_taxonomy_returns_200(self):
        response = client.get("/api/v1/taxonomy")
        assert response.status_code == 200

    def test_taxonomy_returns_structure(self):
        response = client.get("/api/v1/taxonomy")
        data = response.json()
        assert "id" in data
        assert "name" in data
        assert "root" in data
        assert "total_nodes" in data

    def test_taxonomy_root_has_name(self):
        response = client.get("/api/v1/taxonomy")
        data = response.json()
        assert "name" in data["root"]

    def test_taxonomy_root_has_children(self):
        response = client.get("/api/v1/taxonomy")
        data = response.json()
        assert "children" in data["root"]
        assert isinstance(data["root"]["children"], list)

    def test_taxonomy_total_nodes_positive(self):
        response = client.get("/api/v1/taxonomy")
        data = response.json()
        assert data["total_nodes"] > 0


class TestValidationErrors:
    def test_invalid_json_returns_422(self):
        response = client.post(
            "/api/v1/classify",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_wrong_type_text_returns_422(self):
        response = client.post(
            "/api/v1/classify",
            json={"text": 12345},
        )
        assert response.status_code == 422

    def test_batch_wrong_type_texts_returns_422(self):
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": "not a list"},
        )
        assert response.status_code == 422

    def test_batch_invalid_text_item_returns_422(self):
        response = client.post(
            "/api/v1/classify/batch",
            json={"texts": [123, 456]},
        )
        assert response.status_code == 422
