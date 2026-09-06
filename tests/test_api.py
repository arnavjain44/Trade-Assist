from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Intraday Trading Dashboard" in response.text or "<html" in response.text

def test_trace_endpoint():
    response = client.get("/api/v1/trace")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

def test_indicators_endpoint():
    response = client.get("/api/v1/indicators/RELIANCE.NS")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "RELIANCE.NS"
    assert "indicators" in data
    assert len(data["indicators"]) > 0

def test_recommendations_endpoint():
    payload = {
        "investment_amount": 5000.0,
        "tickers": ["RELIANCE.NS", "TCS.NS"],
        "provider": "auto"
    }
    response = client.post("/api/v1/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["investment_amount"] == 5000.0
    assert "recommendations" in data
    assert "charts" in data

def test_provider_override_detection():
    from app.agent.llm_client import detect_provider_override

    # Standard default
    provider, order = detect_provider_override("explain paytm 5 ema chart")
    assert provider is None
    assert order == ["gemini", "groq", "openrouter"]

    # Explicit override for Groq
    provider, order = detect_provider_override("explain paytm 5 ema chart via groq")
    assert provider == "groq"
    assert order == ["groq", "gemini", "openrouter"]

    # Explicit override for Gemini
    provider, order = detect_provider_override("use gemini for reliance analysis")
    assert provider == "gemini"
    assert order == ["gemini", "groq", "openrouter"]

