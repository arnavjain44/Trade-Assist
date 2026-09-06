# Trade-Assist ML & Data Pipeline Audit Report

**Date**: 2026-09-05  
**Audit Target**: `Trade-Assist` Intraday Trading Intelligence Repository  
**Status**: Audit Complete — Zero Code Modifications Performed  

---

## A. Current Architecture Overview

The system is built as a FastAPI asynchronous backend serving a dark-mode SF Pro single-page frontend.

```
[ Frontend: index.html / app.js ]
              │
              ▼
[ FastAPI Endpoints: /api/v1/recommendations, /chat, /peers, /backtest, /indicators ]
              │
              ├──► [ Data Fetcher: yfinance 1mo Daily OHLCV ]
              │           │
              │           ▼
              ├──► [ Indicators Engine: EMA-5, RSI, OBV, Bollinger Bands, MACD, VWAP ]
              │           │
              │           ▼
              ├──► [ Sentiment Analyzer: VADER / Static Headlines ]
              │           │
              │           ▼
              ├──► [ Feature Vector Construction (8 Features) ]
              │           │
              │           ▼
              ├──► [ ML Prediction Engine: Logistic Regression / Random Forest ]
              │           │  (Currently overridden by hardcoded rule logic)
              │           ▼
              ├──► [ Intraday Constraint Enforcer: Max 40% Cap, Mandatory Same-Day Exit ]
              │           │
              │           ▼
              └──► [ Response Pipeline & LLM Agent Rationale Synthesis ]
```

### Core Components Matrix
- **Data Ingestion**: `app/core/data_fetcher.py` (yfinance 1-month daily series).
- **Indicators Engine**: `app/core/indicators.py` (EMA-5, RSI, OBV, Bollinger Bands, MACD, VWAP).
- **Sentiment Engine**: `app/core/sentiment.py` (VADER sentiment over mock text strings).
- **Vector DB**: `app/db/vector_store.py` (ChromaDB 9-dimensional cosine similarity).
- **Graph DB**: `app/db/graph_store.py` (Neo4j Cypher pattern relationships with fallback).
- **ML Engine**: `app/ml/pipeline.py` (Scikit-Learn Logistic Regression & Random Forest).
- **Constraints**: `app/core/constraints.py` (Deterministic post-processing rules).
- **LLM Agent**: `app/agent/llm_client.py` & `app/agent/loop.py` (Multi-key failover client for narrative & rationale generation).

---

## B. Current ML Flow

```
InvestmentRequest (capital, tickers, sector)
       │
       ▼
1. Fetch live OHLCV dataframe via yfinance
       │
       ▼
2. Calculate 6 technical indicators + extract latest summary
       │
       ▼
3. Run sentiment analyzer over 5 sample headlines
       │
       ▼
4. Query ChromaDB for similarity score + Neo4j for graph pattern
       │
       ▼
5. Construct 8-feature vector:
   [EMA_diff_pct, RSI, OBV_diff, BB_pos, MACD_diff, VWAP_diff_pct, Sentiment_score, Vector_similarity]
       │
       ▼
6. Pass feature vector to MLPredictionEngine.predict_trade_signal()
       │
       ├─► (Model probability computed, then IGNORED)
       ├─► (Signal overridden by if/else RSI/MACD/Sentiment thresholds)
       ├─► (Confidence artificially boosted by +25% and clipped)
       └─► (Target & Stop Loss generated via fixed % multipliers)
       │
       ▼
7. Enforce Intraday Constraints (~40% max allocation, same-day exit)
       │
       ▼
8. Return TradeRecommendation objects to Frontend
```

---

## C. Detailed Audit of Hardcoded / Synthetic Components

| Component / Function | File & Lines | Audit Findings | Classification |
| :--- | :--- | :--- | :--- |
| `_bootstrap_synthetic_training` | [app/ml/pipeline.py:20-45](file:///d:/proj1/proj%20files/app/ml/pipeline.py#L20-L45) | Models are trained on `np.random.normal(0, 1)` and `np.random.uniform` synthetic random numbers with arbitrary label rules instead of historical stock prices. | **Synthetic Data** |
| `predict_trade_signal` Signal Override | [app/ml/pipeline.py:80-84](file:///d:/proj1/proj%20files/app/ml/pipeline.py#L80-L84) | The ML model prediction (`predicted_class`) is completely ignored. Action (`BUY`/`SELL`) is hardcoded via `if rsi > 68 or macd_diff < -1.5 or sentiment_score < -0.3: action = "SELL" else: "BUY"`. | **Overridden Prediction** |
| `predict_trade_signal` Confidence Modification | [app/ml/pipeline.py:77-78](file:///d:/proj1/proj%20files/app/ml/pipeline.py#L77-L78) | Raw model probability is artificially inflated by `+ 25.0` and hard-clamped between 72.0% and 94.5%: `round(min(max(raw_conf + 25.0, 72.0), 94.5), 1)`. | **Artificial Confidence** |
| `predict_trade_signal` Target & Stop Loss | [app/ml/pipeline.py:87-92](file:///d:/proj1/proj%20files/app/ml/pipeline.py#L87-L92) | Target price is hardcoded to `price * 1.022` (+2.2%) and stop loss to `price * 0.991` (-0.9%) rather than derived from volatility/ATR or model regression outputs. | **Hardcoded Multipliers** |
| `predict_trade_signal` Model Name Claim | [app/ml/pipeline.py:101](file:///d:/proj1/proj%20files/app/ml/pipeline.py#L101) | Returns `"model_used": "RandomForestClassifier / XGBoost"` despite XGBoost not being installed, trained, or invoked. | **False Telemetry Claim** |
| `sample_headlines` News Fetcher | [app/core/sentiment.py:20-26](file:///d:/proj1/proj%20files/app/core/sentiment.py#L20-L26) | News headlines are hardcoded template strings (`f"{clean_symbol} reports strong quarterly revenue..."`) instead of real financial news API calls. | **Fake News Data** |
| Sentiment Model | [app/core/sentiment.py:10-54](file:///d:/proj1/proj%20files/app/core/sentiment.py#L10-L54) | Uses VADER dictionary or simple word count fallback instead of the specified FinBERT transformer model. | **Missing Specification** |
| `run_model_backtest` | [app/api/v1/endpoints/backtest.py:14-21](file:///d:/proj1/proj%20files/app/api/v1/endpoints/backtest.py#L14-L21) | Returns static hardcoded numbers (`overall_accuracy=92.4`, `logistic_regression_accuracy=84.5`, `xgboost_accuracy=94.1`) without executing any backtesting. | **Fake Backtest Endpoint** |
| `_seed_if_empty` Fingerprints | [app/db/vector_store.py:148-170](file:///d:/proj1/proj%20files/app/db/vector_store.py#L148-L170) | Initial ChromaDB vector database embeddings are populated with random noise (`np.random.normal(0, 1, size=9)`). | **Placeholder Embeddings** |
| Neo4j Pattern Fallback | [app/db/graph_store.py:63-78](file:///d:/proj1/proj%20files/app/db/graph_store.py#L63-L78) | When Neo4j is offline, returns hardcoded pattern dicts (`"Energy Breakout"`, `"IT Momentum"`, etc.). | **Fallback Pattern Data** |

---

## D. What Must Be Replaced

1. **Synthetic Training Data**: Replace `_bootstrap_synthetic_training()` with a real dataset builder that downloads historical OHLCV data for all NIFTY 50 equities (1-year to 2-year daily or hourly timeframe via `yfinance`), computes the 6 technical indicators + news sentiment feature, and constructs true supervised dataset matrices ($X, y$).
2. **Hardcoded Signal Rules & Overrides**: Remove the `if/else` override in `predict_trade_signal`. The direction (`BUY`/`SELL`/`HOLD`) must be generated directly by the trained ML model's prediction.
3. **Artificial Confidence Score Inflation**: Remove `+ 25.0` padding and clipping. Confidence must reflect true calibrated model class probabilities (`model.predict_proba(X)`).
4. **Hardcoded Target Price & Stop Loss**: Compute target price and stop loss dynamically based on Average True Range (ATR) or volatility bands (Bollinger Bands / support & resistance levels) rather than fixed percentage constants.
5. **Fake Backtest Endpoint**: Implement genuine **chronological backtesting** (time-series cross-validation / rolling window backtest without lookahead bias) comparing Logistic Regression, Random Forest, and XGBoost/LightGBM models, selecting the best model based on actual test-set precision/ROC-AUC.
6. **FinBERT News Sentiment**: Replace static headline templates with a real financial news parser and FinBERT model (`ProsusAI/finbert` via HuggingFace transformers or pipeline) with fallback to VADER when GPU/memory is constrained.

---

## E. Proposed Real Training & Evaluation Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                      1. Historical Data Ingestion                      │
│   Download 1-2 Years Daily OHLCV for all 50 NIFTY Equities via yfinance │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      2. Feature Engineering (X)                        │
│   • EMA-5 Difference %      • RSI (9-period)                           │
│   • OBV Trend Direction     • Bollinger Band Position (0 to 1)         │
│   • MACD Line - Signal      • VWAP Difference %                        │
│   • FinBERT Sentiment Score • Vector Similarity Score                  │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      3. Target Labeling (y)                            │
│   • Forward Intraday Return: 1 if Next Day High > Entry + 1.5% before  │
│     Low < Entry - 1.0%, else 0                                         │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 4. Chronological Time-Series Split                     │
│   • Train: First 70% of chronological timeline                         │
│   • Validation: Next 15% of timeline                                   │
│   • Test: Final 15% out-of-sample evaluation                           │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   5. Multi-Model Candidate Evaluation                  │
│   • Logistic Regression  • Random Forest Classifier  • XGBoost        │
│   Calculate Precision, Recall, F1, ROC-AUC on out-of-sample test set   │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               6. Model Selection & Persistence (joblib)                │
│   Save best candidate model to disk (app/ml/models/best_model.joblib)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## F. File Modification Plan

### Files to Modify / Create

1. **`app/ml/pipeline.py`**
   - Implement real feature extraction from indicator dataframes.
   - Load pre-trained serialized model (`best_model.joblib`) trained on real historical data.
   - Output true ML class probabilities and predictions without rule overrides or artificial confidence padding.
   - Calculate ATR-based dynamic target price and stop loss.

2. **`app/ml/trainer.py` [NEW]**
   - Historical dataset builder (downloads 1-2 years of historical NIFTY 50 data).
   - Computes features ($X$) and ground-truth forward target labels ($y$).
   - Trains Logistic Regression, Random Forest, and XGBoost models using chronological time-series splitting.
   - Evaluates metrics on unseen test set and serializes the winning model to `app/ml/models/best_model.joblib`.

3. **`app/api/v1/endpoints/backtest.py`**
   - Connect to real chronological backtester engine.
   - Run out-of-sample backtest across historical data and return empirical metrics (accuracy, precision, win rate, total trades).

4. **`app/core/sentiment.py`**
   - Integrate FinBERT (`ProsusAI/finbert` via HuggingFace `transformers` pipeline) with VADER fallback.
   - Integrate live Google News / RSS financial headline parser for the target ticker symbol.

---

## G. Files That Must Remain Unchanged

The following core system files enforce strict architecture and MUST NOT be modified or rewritten:

1. **`app/core/constraints.py`**: Mandatory intraday post-processing rules (same-day exit, ~40% max allocation limit).
2. **`app/core/indicators.py`**: 6 technical indicator mathematical calculations (EMA-5, RSI, OBV, Bollinger Bands, MACD, VWAP).
3. **`app/schemas/requests.py` & `app/schemas/responses.py`**: OpenAPI schema contracts.
4. **`app/agent/llm_client.py` & `app/agent/loop.py`**: LLM tool calling, multi-key rotation, and rationale formatting (LLM must NOT generate numerical predictions).
5. **`frontend/index.html` & `frontend/app.js`**: Dark mode SF Pro UI layout, fixed prompt footer bar, and chart tabs.

---

## Concise Implementation Plan

1. **Step 1: Build Historical Data & Training Module (`app/ml/trainer.py`)**
   - Download historical NIFTY 50 daily OHLCV dataset via `yfinance`.
   - Calculate 6 technical indicators + FinBERT sentiment features.
   - Generate ground-truth labels based on forward 1-day profit/loss targets.
   - Split chronologically (70% train / 15% validation / 15% test).
   - Train Logistic Regression, Random Forest, and XGBoost; select best performer and save to `app/ml/models/best_model.joblib`.

2. **Step 2: Refactor ML Prediction Engine (`app/ml/pipeline.py`)**
   - Load `best_model.joblib`.
   - Remove synthetic random bootstrap training.
   - Remove hardcoded `if/else` signal override and return pure model predictions + calibrated probabilities.
   - Replace fixed percentage target/stop-loss with ATR-based volatility bounds.

3. **Step 3: Connect Chronological Backtesting Endpoint (`app/api/v1/endpoints/backtest.py`)**
   - Run chronological time-series evaluation on unseen test data.
   - Return actual measured accuracy, precision, and trade count metrics.

4. **Step 4: Upgrade FinBERT News Sentiment (`app/core/sentiment.py`)**
   - Add FinBERT transformer pipeline with VADER fallback.
   - Fetch real financial headlines for stock symbols via news feed/RSS.

5. **Step 5: Verification & Testing**
   - Run `python -m pytest tests/ -v` to ensure all endpoints, constraint checks, and indicator tests pass.
