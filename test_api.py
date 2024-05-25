from fastapi.testclient import TestClient
from API.main import app
import requests

client = TestClient(app)

def test_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "<h1>Welcome to the API</h1>" in response.text

def test_classify_success():
    test_text = "I am happy"
    response = requests.post(f"http://127.0.0.1:8000/classify?text={test_text}")
    assert response.status_code == 200
    assert "text" in response.json()
    assert "predict" in response.json()
    assert response.json()["text"] == test_text
    assert response.json()["predict"] == "joy"

def test_classify_no_text():
    response = requests.post(f"http://127.0.0.1:8000/classify?text=")
    assert response.status_code == 200
    assert response.json() == {"text": "error", "predict": "error"}
