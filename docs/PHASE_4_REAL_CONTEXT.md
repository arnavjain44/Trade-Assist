# Phase 4 — Real News Sentiment & Real Historical Context Documentation

This document describes the design, implementation, data quality audit, and verification protocol for **Phase 4** of Trade-Assist.

Phase 4 replaces remaining placeholder information sources with **real Hugging Face FinBERT news sentiment** and **real ChromaDB / Neo4j historical market context**, producing a multi-source feature layer ready for future ML experimentation without altering or retraining Phase 3 ML baselines.

---

## 1. Architecture Overview

Phase 4 introduces a leakage-free multi-source feature ingestion and joining architecture:

```
[Raw 5m Candles (48 Symbols, ~60 Days)] ───► [Technical Features (9 Causal Phase 3 Features)]
                                                              │
[Real yfinance News / Local Cache] ───► [FinBERT (ProsusAI)] ─┼──► [Phase4FeatureJoiner] ──► data/processed/phase4_features.parquet
                                                              │
[Real Indicators] ───► [Chroma 2-Collection Vector Store] ─────┤
                       - whole_market_daily_fingerprints      │
                       - per_stock_daily_fingerprints         │
                                                              │
[Real Patterns] ───► [Neo4j Graph (Idempotent MERGE)] ────────┘
```

---

## 2. News Provider Abstraction

- **Interface**: Abstract `NewsProvider` class (`app/ml/news_processor.py`).
- **Real Implementation**: `YFinanceNewsProvider` fetches real news headlines, summaries, publication timestamps, providers, and URLs directly via `yfinance`.
- **Reproducible Local Provider**: `LocalNewsCacheProvider` persists raw fetched articles to `data/processed/raw_news_cache.json` for offline reproducibility.
- **Data Integrity**: **Zero synthetic, generated, or rule-invented news headlines**.

---

## 3. FinBERT Sentiment Engine

- **Hugging Face Model**: `ProsusAI/finbert` (sequence classification pipeline).
- **Probabilities Preserved**:
  - `positive_probability`
  - `negative_probability`
  - `neutral_probability`
- **Deterministic Sentiment Score Formula**:
  $$\text{sentiment\_score} = \text{positive\_probability} - \text{negative\_probability}$$
- **Model Caching**: Cached locally in `data/processed/finbert_sentiment_cache.json` to prevent unnecessary re-downloads and re-inference.

---

## 4. News Timestamp Policy & Leakage Prevention

- **Timezone Standardization**: All news publication timestamps are converted to `Asia/Kolkata` (IST).
- **Strict Intraday Temporal Rule**:
  For any trading candle at timestamp $T$, a news article is eligible **ONLY** if:
  $$\text{news\_pub\_timestamp} < T$$
- **Pre-Market & Previous Evening Policy**:
  - Previous evening news (published after market close) is eligible starting at 09:15 IST the next trading day.
  - Same-day intraday news (published at $T_{\text{news}}$) is eligible **ONLY** for candles where $T_{\text{candle}} > T_{\text{news}}$.
- **Leakage Prevention**: Future headlines (e.g. published at 11:00 AM) are strictly excluded from earlier candles (e.g. 10:00 AM candle).

---

## 5. Historical News Aggregation Rules

Aggregated per candle into 7 clean features:
1. `number_of_articles`: Count of eligible articles prior to candle timestamp.
2. `mean_sentiment`: Average `sentiment_score` of eligible articles (0.0 if no news).
3. `positive_probability_mean`: Average `positive_probability` (0.0 if no news).
4. `negative_probability_mean`: Average `negative_probability` (0.0 if no news).
5. `neutral_probability_mean`: Average `neutral_probability` (0.0 if no news).
6. `latest_news_timestamp`: ISO timestamp string of the most recent eligible article.
7. `has_news`: Boolean flag (`True` if `number_of_articles > 0`, else `False`).

---

## 6. Chroma Vector Store (Two Collections)

Per project design, Phase 4 populates **exactly two ChromaDB collections**:

1. `whole_market_daily_fingerprints`
   - Represents genuine cross-sectional market conditions per trading day (cross-sectional mean RSI, mean MACD, % above EMA5, market OBV).
   - Document ID format: `market_{trading_date}` (e.g. `market_2026-08-01`).

2. `per_stock_daily_fingerprints`
   - Represents daily stock state using 9-dim normalized indicator vector.
   - Document ID format: `stock_{symbol}_{trading_date}` (e.g. `stock_RELIANCE.NS_2026-08-01`).

---

## 7. Vector Embedding Approach

Fingerprints use normalized 9-dimensional real technical indicator vectors:
`[price_vs_ema5, rsi, obv_norm, bollinger_position, macd, macd_signal, macd_diff, price_vs_vwap, direction]`

No random seed vectors, fake win rates, or placeholder text are used.

---

## 8. Temporal Similarity Querying

Retrieval answers: *"How similar is the current market/stock state to past historical days?"*
- **Strict Query Filter**:
  $$\text{trading\_date} < \text{query\_date}$$
- Chroma query metadata filter: `where={"trading_date": {"$lt": query_date_str}}`.
- Output features: `market_similarity` (0.0–1.0) and `stock_similarity` (0.0–1.0).

---

## 9. Neo4j Graph Ingestion Schema

- **Schema**:
  `(Stock {symbol})-[:ON_DAY]->(TradingDay {date})-[:SHOWED_PATTERN]->(Pattern {name})`
  `(Pattern)-[:USES_INDICATOR]->(Indicator {name})`
  `(Pattern)-[:RESULTED_IN]->(Outcome {direction, avg_return})`
  `(TradingDay)-[:PART_OF]->(MarketRegime {name})`
- **Idempotence**: Constructed strictly using Cypher `MERGE` statements.
- **Graceful Fallback**: If Neo4j URI is unconfigured, ingestion logs status and skips cleanly without interrupting feature generation.

---

## 10. Data Quality Results Summary

Audit report generated at `data/processed/phase4_data_quality.json`:
- **Symbols Processed**: 48 NIFTY 50 equities.
- **Trading Days**: 59 days.
- **Candle Count**: 214,086 5-minute candles.
- **FinBERT Processing Success Rate**: 100.0%.
- **Chroma Market Fingerprints**: 59 daily market vectors.
- **Chroma Stock Fingerprints**: 2,727 daily stock vectors.
- **Temporal Leakage Checks**: 100% Passed.
- **Missing News**: Explicitly flagged with `has_news = False`. Zero synthetic fallback text generated.

---

## 11. Known Data Limitations

- **Free News API Horizon**: `yfinance` free news API returns ~10 recent news articles per ticker. Historical candles beyond the free API publication horizon have `has_news = False`.
- **Handling**: Missing news is treated as a legitimate real-world condition and is explicitly reported in `data/processed/phase4_data_quality.json`.

---

## 12. Reproducibility Instructions

To regenerate Phase 4 real news, FinBERT scores, Chroma collections, and joined feature datasets:

```bash
python -m app.ml.phase4_pipeline
```

---

## 13. Test Suite & Verification

To run Phase 4 unit tests alongside Phase 3 tests:

```bash
python -m pytest
```

All 72 unit tests (63 Phase 3 + 9 Phase 4 tests) pass with 100% success rate.

---

## 14. What Phase 4 Does NOT Do

> [!IMPORTANT]
> 1. Does **NOT** retrain ML models (LR, RF, LightGBM).
> 2. Does **NOT** modify Phase 3 model results (`data/processed/phase3_model_results.json`).
> 3. Does **NOT** connect Phase 4 features to the live recommendation endpoint.
> 4. Does **NOT** alter Phase 2 labels, stop-loss, or target parameters.
