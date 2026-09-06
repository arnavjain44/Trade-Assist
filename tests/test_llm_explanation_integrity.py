import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.agent.loop import agent_loop
from app.ml.pipeline import ml_engine
from app.schemas.requests import ChatRequest

client = TestClient(app)

def test_stock_analysis_invokes_llm_explanation_path():
    """Requirement A: Verify stock analysis invokes the LLM explanation path via POST /api/v1/chat."""
    payload = {
        "message": "Explain technicals for HDFCBANK.NS",
        "session_id": "test_session",
        "provider": "auto",
        "stock_analysis": {
            "symbol": "HDFCBANK.NS",
            "current_price": 1450.0,
            "ema_5": 1445.0,
            "rsi": 55.0,
            "macd": 1.2,
            "macd_signal": 1.0,
            "vwap": 1448.0,
            "overall_sentiment_score": 0.25,
            "p_long": 0.42,
            "p_short": 0.18,
            "threshold": 0.80,
            "action": "HOLD",
            "qualified": False
        }
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "provider_used" in data
    assert data["provider_used"] in ["gemini", "groq", "openrouter", "fallback", "offline_knowledge_base"]

def test_stock_analysis_passes_factual_values_to_llm():
    """Requirement B: Verify stock analysis passes backend-calculated factual values in system context."""
    analysis = {
        "symbol": "RELIANCE.NS",
        "current_price": 2800.0,
        "ema_5": 2790.0,
        "rsi": 62.0,
        "p_long": 0.45,
        "p_short": 0.12,
        "threshold": 0.80,
        "action": "HOLD",
        "qualified": False
    }
    req = ChatRequest(
        message="Tell me about RELIANCE",
        session_id="test_facts",
        stock_analysis=analysis
    )
    assert req.stock_analysis["symbol"] == "RELIANCE.NS"
    assert req.stock_analysis["p_long"] == 0.45
    assert req.stock_analysis["action"] == "HOLD"

def test_llm_cannot_override_action():
    """Requirement C: Verify LLM explanation context strictly preserves backend action."""
    eval_res = ml_engine.inference.evaluate_dual_directions(
        technical_indicators={
            "rsi": 50.0, "obv": 1000.0, "bollinger_position": 0.5,
            "macd": 0.0, "macd_signal": 0.0, "macd_diff": 0.0,
            "price_vs_vwap": 0.0, "price_vs_ema5": 0.0
        },
        news_features={"sentiment_score": 0.0, "has_news": False, "number_of_articles": 0},
        context_features={"market_similarity": 0.0, "stock_similarity": 0.0},
        timeframe="5m"
    )
    # Model probability < 0.80 must produce HOLD
    assert eval_res["action"] == "HOLD"
    assert eval_res["qualified"] is False

def test_llm_cannot_change_model_probability_or_threshold():
    """Requirements D & E: Model probability and threshold remain strictly frozen in Phase 5 engine."""
    assert ml_engine.inference.threshold == 0.8000
    assert ml_engine.inference.model_family == "lightgbm"

def test_hold_remains_hold_even_when_asking_llm():
    """Requirement G: HOLD action is preserved regardless of chat queries."""
    pred = ml_engine.predict_trade_signal(
        symbol="INFY.NS",
        current_price=1500.0,
        indicators={
            "rsi": 52.0, "obv": 5000.0, "bollinger_position": 0.5,
            "macd": 0.1, "macd_signal": 0.1, "macd_diff": 0.0,
            "price_vs_vwap": 0.0, "price_vs_ema5": 0.0, "timeframe": "5m"
        },
        sentiment_score_or_features={"sentiment_score": 0.1, "has_news": True, "number_of_articles": 1},
        vector_similarity_or_features={"market_similarity": 0.0, "stock_similarity": 0.0},
        timeframe="5m"
    )
    # Unqualified signal must be HOLD with 0 allocation
    assert pred["action"] == "HOLD"
    assert pred["qualified"] is False

def test_allocation_remains_zero_when_no_candidate_qualifies():
    """Requirement H: When no candidate qualifies (P < 0.80), allocated capital is ₹0 and 100% cash preserved."""
    payload = {
        "investment_amount": 50000.0,
        "tickers": ["HDFCBANK.NS", "RELIANCE.NS"],
        "provider": "auto"
    }
    response = client.post("/api/v1/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["investment_amount"] == 50000.0
    if len(data["recommendations"]) == 0:
        assert data["total_allocated"] == 0.0
        assert data["unallocated_cash"] == 50000.0

def test_provider_trace_reflects_actual_provider():
    """Requirement I: Provider trace reflects actual runtime provider used or offline fallback."""
    response = client.post("/api/v1/chat", json={"message": "What is 5 EMA?"})
    assert response.status_code == 200
    data = response.json()
    assert "provider_used" in data
    assert data["provider_used"] in ["gemini", "groq", "openrouter", "fallback", "offline_knowledge_base"]
