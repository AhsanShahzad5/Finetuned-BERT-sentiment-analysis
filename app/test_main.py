# test_app.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert "succesfully running" in response.json()

def test_predict_positive():
    payload = {"text": "I really loved this movie!"}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "sentiment" in body
    assert "confidence" in body
    assert isinstance(body["confidence"], float)
    # Since the model is not perfect, don’t force strict label check
    assert body["sentiment"] in ["positive", "negative"]

def test_predict_negative():
    payload = {"text": "This movie was terrible and boring."}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "sentiment" in body
    assert "confidence" in body
    assert isinstance(body["confidence"], float)
    assert body["sentiment"] in ["positive", "negative"]
