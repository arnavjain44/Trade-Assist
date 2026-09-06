# Trade-Assist — NSE Intraday Stock Recommendation Engine

A full-stack intraday trading assistant for NSE (National Stock Exchange) equities. Trade-Assist combines real-time 5-minute market data, FinBERT news sentiment, historical vector context, and a validated LightGBM machine learning model to generate high-confidence BUY/SELL/HOLD recommendations — with strict capital allocation constraints and mandatory same-day exits.

No paid broker APIs. No synthetic data. No hardcoded signals.

---

## How It Works

```
Live 5-minute market data (yfinance)
        ↓
14 technical + news + context features
        ↓
LightGBM model  →  P(LONG) and P(SHORT)
        ↓
Threshold: P ≥ 0.8000
        ↓
BUY / SELL / HOLD
        ↓
Deterministic capital allocation (max 40% per stock, integer shares)
        ↓
Same-day intraday exit enforced
```

The model evaluates every stock in both the LONG and SHORT directions independently. A recommendation is only generated when the model''s output probability clears the 0.8000 threshold. If nothing qualifies, the system returns HOLD across the board and preserves 100% of your capital.

---

## Architecture

```
frontend/          ← Vanilla JS + Chart.js UI (served statically)
app/
  api/             ← FastAPI REST endpoints
  agent/           ← LLM orchestration loop (Gemini / Groq / OpenRouter)
  core/
    data_fetcher.py   ← yfinance 5-minute intraday data with IST timezone
    indicators.py     ← EMA-5, RSI, OBV, Bollinger Bands, MACD, VWAP
    sentiment.py      ← FinBERT news sentiment (causal temporal cutoff)
    constraints.py    ← Capital allocation, position sizing, exit rules
  ml/
    phase5_feature_builder.py  ← Canonical 14-feature vector construction
    phase5_inference.py        ← Production LightGBM inference engine
    pipeline.py                ← Live prediction pipeline (no synthetic data)
    models.py                  ← Research model class (TradeSignalClassifier)
    context_store.py           ← ChromaDB vector store management
  db/
    vector_store.py  ← Chroma similarity queries (prior-day, causal)
    graph_store.py   ← Neo4j knowledge graph (optional)
data/
  models/
    phase5_d_lightgbm.joblib  ← Frozen, serialized LightGBM artifact
  processed/
    phase5_features.parquet   ← Historical 5m candles + features (48 symbols)
scripts/
  export_phase5_model.py   ← Exports and validates the model artifact
  populate_chroma.py       ← Populates the vector store from historical data
tests/                     ← 144 unit and integration tests
chroma_db/                 ← Local ChromaDB persistent store (git-ignored)
```

---

## Features

### Market Data
- **Real-time 5-minute intraday OHLCV** from yfinance for 48 NSE NIFTY 50 equities.
- Localized to **Asia/Kolkata (IST)** — no timezone confusion.
- In-flight candles (i.e., candles not yet closed) are automatically dropped before any inference.

### Technical Indicators (6)

| Indicator | Period | Purpose |
| :--- | :---: | :--- |
| EMA | 5 | Short-term trend direction |
| RSI | 14 | Intraday momentum extremes |
| OBV | — | Volume accumulation/distribution |
| Bollinger Bands | 20, 2σ | Volatility squeeze and position |
| MACD | 12/26/9 | Trend crossover and momentum |
| VWAP | Session | Intraday institutional price anchor |

### News Sentiment
- Headlines fetched from yfinance news per ticker.
- Scored using **ProsusAI/FinBERT** (local inference, no API cost).
- **Strict temporal cutoff**: Only articles published *before* the decision timestamp are used. Zero future-data leakage.
- Graceful missing-news handling: uses training-set imputed median when no eligible articles exist.

### Historical Context (ChromaDB)
- Two ChromaDB collections: `whole_market_daily_fingerprints` and `per_stock_daily_fingerprints`.
- Each day''s market state is encoded as a **9-dimensional scale-independent indicator vector**.
- Live queries retrieve the most similar *prior day* using cosine similarity — strict temporal filter enforces `prior_day < today`. No lookahead.
- Context similarities feed directly into the ML model as features.

### Machine Learning Model
- **Algorithm**: LightGBM Classifier (`LGBMClassifier`)
- **Input**: 14 features — 8 technical indicators, 3 news features (`sentiment_score`, `has_news`, `number_of_articles`), 2 context similarities (`market_similarity`, `stock_similarity`), 1 direction flag.
- **Preprocessing**: `SimpleImputer(strategy=''median'')` fitted strictly on training data. Imputer median saved and serialized with the model — never refit at runtime.
- **Threshold**: P ≥ 0.8000 (hard requirement for BUY or SELL signal)
- **Evaluation**: Both LONG and SHORT directions scored independently per stock per candle.
- **Artifact**: `data/models/phase5_d_lightgbm.joblib` (347 KB, self-contained).

### Capital Allocation
- **Max position**: 40% of total capital per stock.
- **Integer shares**: Always floor-rounded — never fractional.
- **Same-day exit**: All positions carry `hold_until: "same_day"` — no overnight exposure.
- **Zero forced trades**: If the model qualifies nothing, allocated capital = ₹0 and full cash is preserved.
- **Target**: +2.2% from entry (LONG) / −2.2% (SHORT).
- **Stop loss**: −0.9% from entry (LONG) / +0.9% (SHORT).

---

## Model Validation Summary

The model was trained and validated on **59 trading days** (~2.7 calendar months) of real NSE intraday 5-minute history across **48 NIFTY 50 symbols**, using a strict chronological split:

- **Train**: Days 1–42
- **Validation**: Days 43–51 (threshold and weight selection)
- **Test**: Days 52–59 (locked, untouched during training)

**Out-of-sample results at P ≥ 0.8000** (after 5 bps round-trip friction):

| Metric | Value |
| :--- | :--- |
| Test trades | 48 |
| Win rate | 68.75% |
| Net avg return per trade | +0.27% |
| Net Profit Factor | 2.34 |
| Max Drawdown | 6.05% |

> **Honest Limitation**: These results come from a 59-day window. Multi-year robustness has not been established. The model is selective by design — on quiet market days it will return HOLD for all candidates. Past performance does not guarantee future results.

---

## Supported Stocks (48 NSE NIFTY 50 Universe)

`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `BHARTIARTL.NS`, `SBIN.NS`, `AXISBANK.NS`, `ITC.NS`, `WIPRO.NS`, `BAJFINANCE.NS`, `MARUTI.NS`, `LT.NS`, `HCLTECH.NS`, `SUNPHARMA.NS`, `TITAN.NS`, `ULTRACEMCO.NS`, `ASIANPAINT.NS`, `KOTAKBANK.NS`, `TATASTEEL.NS`, `INDUSINDBK.NS`, `NTPC.NS`, `POWERGRID.NS`, `COALINDIA.NS`, `ONGC.NS`, `HDFCLIFE.NS`, `SBILIFE.NS`, `BAJAJ-AUTO.NS`, `M&M.NS`, `HEROMOTOCO.NS`, `EICHERMOT.NS`, `BPCL.NS`, `IOC.NS`, `DIVISLAB.NS`, `DRREDDY.NS`, `CIPLA.NS`, `APOLLOHOSP.NS`, `BRITANNIA.NS`, `NESTLEIND.NS`, `HINDUNILVR.NS`, `GRASIM.NS`, `JSWSTEEL.NS`, `HINDALCO.NS`, `ADANIENT.NS`, `ADANIPORTS.NS`, `BEL.NS`, `HAL.NS`, `TRENT.NS`, `ZOMATO.NS`

---

## REST API

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/recommendations` | Main endpoint. Accepts investment amount, optional tickers/sector, LLM provider. Returns BUY/SELL/HOLD recommendations with allocation, charts, and rationale. |
| `GET` | `/api/v1/indicators/{symbol}` | Returns 5-minute indicator time-series for a symbol (for charting). |
| `POST` | `/api/v1/chat` | Natural language Q&A about stocks with session memory. |
| `POST` | `/api/v1/backtest` | Runs the strategy against historical candles for a given symbol. |
| `POST` | `/api/v1/peers` | Returns sector peers by Chroma vector similarity. |
| `GET` | `/api/v1/trace` | Pipeline telemetry: latency, model used, provider, tool call count. |

### Example Request
```http
POST /api/v1/recommendations
Content-Type: application/json

{
  "investment_amount": 50000,
  "tickers": ["RELIANCE.NS", "TCS.NS"],
  "provider": "auto"
}
```

### Example Response
```json
{
  "investment_amount": 50000.0,
  "total_allocated": 0.0,
  "unallocated_cash": 50000.0,
  "recommendations": [],
  "trace": {
    "model_name": "phase5_d_lightgbm",
    "execution_time_seconds": 1.24,
    "provider_used": "gemini"
  }
}
```

> When no stock meets the P ≥ 0.8000 threshold, `recommendations` is an empty list and your full capital is preserved as `unallocated_cash`.

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/arnavjain44/Trade-Assist.git
cd Trade-Assist
pip install -r requirements.txt
```

### 2. Environment Variables (All Optional)

Copy the example file:
```bash
cp .env.example .env
```

The application runs with zero environment variables. The following are optional enhancements:

```env
# Optional: Free LLM providers for AI rationale generation
GEMINI_API_KEY=""      # Google AI Studio — free tier
GROQ_API_KEY=""        # Groq — ultra-fast free inference
OPENROUTER_API_KEY=""  # OpenRouter — multi-model access

# Optional: Neo4j for graph pattern context
NEO4J_URI=""
NEO4J_USER=""
NEO4J_PASSWORD=""
```

### 3. Populate the Historical Context Store (One-Time)

The ChromaDB vector store is populated from the included historical data file:

```bash
python scripts/populate_chroma.py
```

This encodes 59 days × 48 symbols of 5-minute market states into `./chroma_db/`. The directory is git-ignored; re-run this script on a fresh checkout.

### 4. Start the Server

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser to use the frontend, or visit `http://localhost:8000/docs` for the interactive Swagger API docs.

### 5. Run Tests

```bash
python -m pytest -q
```

Expected: **144 passed**.

---

## Repository Structure — Key Files

| File | Description |
| :--- | :--- |
| `app/ml/pipeline.py` | Live prediction pipeline — connects market data to LightGBM inference |
| `app/ml/phase5_inference.py` | Production inference class — loads artifact, enforces 5m, evaluates dual directions |
| `app/ml/phase5_feature_builder.py` | Constructs the exact 14-feature input vector |
| `app/core/data_fetcher.py` | Async 5-minute yfinance fetcher with IST timezone and incomplete-candle guard |
| `app/core/sentiment.py` | FinBERT sentiment with causal temporal cutoff |
| `app/db/vector_store.py` | Chroma prior-day similarity queries |
| `app/core/constraints.py` | Deterministic capital allocation and position sizing |
| `data/models/phase5_d_lightgbm.joblib` | Frozen model artifact (LightGBM + imputer, threshold=0.8000) |
| `scripts/populate_chroma.py` | One-time Chroma DB population from historical parquet |
| `scripts/export_phase5_model.py` | Exports and validates the model artifact against training data |

---

## Cost

**₹0 / $0.** The entire stack runs locally with no paid dependencies:

- Market data: `yfinance` (free)
- Sentiment: `ProsusAI/finbert` via HuggingFace (local inference)
- Vector store: `ChromaDB` (local SQLite)
- ML inference: `LightGBM` (CPU, local)
- LLM rationale: optional free tiers (Gemini, Groq, OpenRouter)

---

## What This System Does NOT Do

- **Does not place orders** — recommendations only, no broker integration.
- **Does not guarantee returns** — past results do not predict future performance.
- **Does not claim 90%+ accuracy** — validated precision is ~68% at the P ≥ 0.80 threshold.
- **Does not use multi-year data** — validated on ~59 trading days of available history.
- **Does not trade overnight** — all positions are enforced as same-day intraday.

---

## License

MIT
