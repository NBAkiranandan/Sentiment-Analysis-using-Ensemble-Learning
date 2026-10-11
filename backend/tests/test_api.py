import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "models_loaded" in data
    assert "models_available" in data
    assert "Logistic Regression" in data["models_available"]
    assert "Random Forest" in data["models_available"]
    assert "AdaBoost" in data["models_available"]

def test_dataset_stats_endpoint():
    response = client.get("/api/dataset/stats")
    assert response.status_code == 200
    assert "total_samples" in response.json()

def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 3

def test_methodology_endpoint():
    response = client.get("/api/methodology")
    assert response.status_code == 200
    data = response.json()
    assert "steps" in data
    assert len(data["steps"]) >= 8
