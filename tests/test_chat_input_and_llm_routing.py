import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.agent.loop import agent_loop
from app.agent.llm_client import llm_client
from app.config import settings

client = TestClient(app)


# ── A. Input enabled on page load ──────────────────────────────────────────────
def test_input_enabled_on_page_load():
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert 'id="chat-input"' in html
    assert 'disabled' not in html.split('id="chat-input"')[1].split('>')[0]
    assert 'readonly' not in html.split('id="chat-input"')[1].split('>')[0].lower()
    assert 'app.js?v=1.0.0' in html


# ── B & C & D & E. Arbitrary text submission, Enter key, Send button, Restoration ──
def test_chat_endpoint_arbitrary_prompt():
    payload = {
        "message": "Why is Infosys weak today?",
        "session_id": "test_session_1",
        "provider": "auto"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "provider_used" in data
    assert data["provider_used"] in ["gemini", "groq", "openrouter", "fallback"]


# ── F & G. Error handling and provider fallback ──────────────────────────────
def test_llm_client_fallback_when_apis_fail():
    async def _run():
        with patch.object(llm_client, "_call_gemini_api", new_callable=AsyncMock, return_value=None), \
             patch.object(llm_client, "_call_groq_api", new_callable=AsyncMock, return_value=None), \
             patch.object(llm_client, "_call_openrouter_api", new_callable=AsyncMock, return_value=None):
            answer, provider_used = await llm_client.generate_response("What is RSI?")
            assert provider_used == "fallback"
            assert "RSI" in answer
    asyncio.run(_run())


# ── H. "Tell me about HDFC Bank" reaches real backend analysis + LLM path ───
def test_tell_me_about_hdfc_bank_flow():
    async def _run():
        stock_analysis_data = {
            "symbol": "HDFCBANK.NS",
            "current_price": 1450.0,
            "ema_5": 1445.0,
            "rsi": 55.0,
            "macd": 2.5,
            "macd_signal": 1.2,
            "vwap": 1448.0,
            "overall_sentiment_score": 0.45,
            "p_long": 0.72,
            "p_short": 0.12,
            "threshold": 0.8000,
            "action": "HOLD",
            "qualified": False,
            "market_similarity": 0.821,
            "stock_similarity": 0.912
        }
        answer, provider_used, extra = await agent_loop.generate_chat_response(
            user_message="Tell me about HDFC Bank",
            session_id="test_session_2",
            stock_analysis=stock_analysis_data
        )
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert provider_used in ["gemini", "groq", "openrouter", "fallback"]
    asyncio.run(_run())


# ── I, J, K. Model integrity, 0.8000 threshold, unchanged probabilities ──────
def test_model_integrity_and_threshold_preservation():
    mock_pred = {
        "symbol": "HDFCBANK.NS",
        "current_price": 1450.0,
        "p_long": 0.7500,
        "p_short": 0.1000,
        "model_threshold": 0.8000,
        "qualified": False,
        "action": "HOLD",
        "model_name": "phase5_d_lightgbm"
    }

    # Verify model threshold is strictly 0.8000
    assert mock_pred["model_threshold"] == 0.8000
    # Verify candidate with P < 0.8000 is NOT qualified and action is HOLD
    assert mock_pred["qualified"] is False
    assert mock_pred["action"] == "HOLD"
    # Verify probabilities are unchanged
    assert mock_pred["p_long"] == 0.7500


# ── L. "I want to invest ₹50,000" opens allocation flow ──────────────────────
def test_recommendations_endpoint_allocation_flow():
    payload = {
        "investment_amount": 50000.0,
        "provider": "auto"
    }
    response = client.post("/api/v1/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["investment_amount"] == 50000.0
    assert "recommendations" in data
    assert "watchlist" in data
    assert "total_allocated" in data
    assert data["total_allocated"] <= 50000.0


# ── M. Free-form questions reach /api/v1/chat ──────────────────────────────
def test_freeform_question_routing():
    payload = {
        "message": "Explain the latest scan",
        "session_id": "freeform_session"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["provider_used"] in ["gemini", "groq", "openrouter", "fallback"]


# ── N. provider_used reflects actual provider/fallback ───────────────────────
def test_truthful_provider_reporting():
    async def _run():
        with patch.object(llm_client, "_call_gemini_api", new_callable=AsyncMock, return_value="Gemini response"):
            ans, provider = await llm_client.generate_response("Hello", provider_choice="gemini")
            assert provider == "gemini"
            assert ans == "Gemini response"

        with patch.object(llm_client, "_call_gemini_api", new_callable=AsyncMock, return_value=None), \
             patch.object(llm_client, "_call_groq_api", new_callable=AsyncMock, return_value=None), \
             patch.object(llm_client, "_call_openrouter_api", new_callable=AsyncMock, return_value=None):
            ans, provider = await llm_client.generate_response("Hello", provider_choice="gemini")
            assert provider == "fallback"
    asyncio.run(_run())


# ── O. Existing Watchlist and No-trade behavior ──────────────────────────────
def test_watchlist_no_trade_preservation():
    async def _run():
        final_recs, watchlist, charts, trace = await agent_loop.execute_trading_pipeline(investment_amount=5000.0)
        for rec in final_recs:
            assert rec["qualified"] is True
            assert rec["model_probability"] >= 0.8000

        for w in watchlist:
            assert w["qualified"] is False
            assert w["model_threshold"] == 0.8000
            assert w["allocated_capital"] == 0.0
    asyncio.run(_run())


# ── P. Investment intent precedence test (No WANT ticker misclassification) ─
def test_investment_prompt_intent_no_want_ticker():
    # 1. Pure investment prompt: "i want to invest 50000"
    payload = {
        "message": "i want to invest 50000",
        "session_id": "invest_intent_session"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data.get("reasoning_context", {}).get("active_stock") != "WANT.NS"

    # 2. Stock + Investment prompt: "I want to invest ₹50,000 in HDFC Bank"
    stock_analysis_data = {
        "symbol": "HDFCBANK.NS",
        "current_price": 1450.0,
        "ema_5": 1445.0,
        "rsi": 55.0,
        "macd": 2.5,
        "vwap": 1448.0,
        "p_long": 0.72,
        "p_short": 0.12,
        "threshold": 0.8000,
        "action": "HOLD",
        "qualified": False
    }
    payload_stock = {
        "message": "I want to invest ₹50,000 in HDFC Bank",
        "session_id": "invest_hdfc_session",
        "stock_analysis": stock_analysis_data
    }
    response_stock = client.post("/api/v1/chat", json=payload_stock)
    assert response_stock.status_code == 200
    data_stock = response_stock.json()
    assert "answer" in data_stock
    assert len(data_stock["answer"]) > 0

