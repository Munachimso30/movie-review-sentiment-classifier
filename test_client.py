from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_info_endpoint():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "predict_json" in data
    assert data["predict_json"] == "/api/predict"


def test_api_predict_positive():
    payload = {"review_text": "This movie was amazing, I really loved it!"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "probability" in data
    assert data["label"] in ["positive", "negative"]
    assert 0.0 <= data["probability"] <= 1.0


def test_api_predict_empty_text():
    payload = {"review_text": "   "}
    response = client.post("/api/predict", json=payload)
    # empty after strip -> 400 from HTTPException
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "review_text must not be empty."
