# AI Intraday Trading Web App — REST API Backend

This is the **FastAPI REST API Backend** for the AI Intraday Trading System. It replaces Streamlit with a clean, decoupled REST architecture targeting **NSE & BSE Indian Stock Equities**.

---

## Features & Architecture

1. **Strictly NSE & BSE Indian Equities**:
   - Stock universe tailored for Indian markets (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, etc.).
2. **6 Curated Technical Indicators**:
   - 5 EMA (Trend)
   - RSI 9-period (Intraday Momentum)
   - OBV (Volume Accumulation)
   - Bollinger Bands (Volatility)
   - MACD 12/26/9 (Signal Crossover)
   - VWAP (Volume Weighted Average Price)
3. **News Sentiment Scoring**:
   - Aggregated headline sentiment scoring (FinBERT / VADER).
4. **Vector & Graph DB Intelligence**:
   - ChromaDB persistent vector collections (`whole_market_fingerprints` & `per_stock_fingerprints`).
   - Neo4j graph database queries for historical pattern match relationships.
5. **Machine Learning Engine**:
   - Trained Scikit-Learn classifiers (Logistic Regression + Random Forest + XGBoost) predicting direction (BUY/SELL/HOLD), confidence score %, and target price.
6. **Deterministic Constraint Enforcement**:
   - **Hard Rule**: Mandatory `same_day` intraday exit.
   - **Position Sizing**: Confidence-weighted allocation capped at **~40% max per stock** to prevent overconcentration.
7. **Free Tier LLM Provider Pool**:
   - Supports **Gemini** (primary free option), **Groq** (ultra-fast inference), and **OpenRouter**, allowing the frontend to select the model preference per request.

---

## How to Run the Server

### 1. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
Provide your free API keys for Gemini, Groq, or OpenRouter if available.

### 3. Launch FastAPI Application
```powershell
python -m uvicorn app.main:app --reload --port 8000
```

---

## Interactive API Documentation
Open your browser to:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/recommendations` | Core endpoint: Accepts capital input (`amount`), tickers, and provider choice. Returns trade picks, capital splits, 6 indicator charts data, and AI rationale. |
| `GET` | `/api/v1/indicators/{symbol}` | Returns raw 6 technical indicator series + news sentiment for frontend charts. |
| `POST` | `/api/v1/chat` | Interactive follow-up Q&A endpoint with agent state memory (remembers picked and rejected stocks). |
| `POST` | `/api/v1/backtest` | Evaluates ML model prediction accuracy against historical stock data. |
| `GET` | `/api/v1/trace` | Operational telemetry (token usage, execution latency, provider status). |

---

## Connecting Your Custom Frontend

You can connect any custom frontend (React, Next.js, Vue, Svelte, Tailwind, or Mobile App) to this backend by sending standard JSON requests to `http://127.0.0.1:8000/api/v1/...`.

### Example Request (`POST /api/v1/recommendations`):
```json
{
  "investment_amount": 5000.0,
  "tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"],
  "provider": "auto",
  "max_stock_count": 5
}
```

### Example Response:
```json
{
  "investment_amount": 5000.0,
  "total_allocated": 4850.0,
  "unallocated_cash": 150.0,
  "recommendations": [
    {
      "symbol": "RELIANCE.NS",
      "action": "BUY",
      "confidence": 88.5,
      "current_price": 2750.0,
      "target_price": 2805.0,
      "stop_loss": 2722.5,
      "hold_until": "same_day",
      "allocation_pct": 0.40,
      "allocated_capital": 2000.0,
      "shares_to_trade": 0,
      "rationale": "Technical indicators for RELIANCE.NS show an overall BULLISH bias..."
    }
  ],
  "charts": { ... },
  "trace": { ... }
}
```
