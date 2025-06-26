from pathlib import Path
import os
import sys
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(Path(__file__).resolve().parents[1], "")))

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "AI Lawyer Agent API"
    assert data["status"] == "active"
    assert "components" in data
