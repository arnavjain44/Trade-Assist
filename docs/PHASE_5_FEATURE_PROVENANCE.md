# Phase 5 — Feature Provenance & Temporal Leakage Audit

This document provides the exhaustive temporal provenance audit for all features used in Phase 5.
Every feature satisfies the strict temporal causality test: **"What information would have been available at the exact decision timestamp $T$?"**

---

## 1. Feature Provenance Matrix

| Feature | Feature Group | Exact Source | Timestamp Basis | Future Data Possible? | Leakage Verification Rule |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `rsi` | Group A (Technical) | 14-period RSI on 5m OHLCV | $T$ (Current candle close) | **NO** | Calculated strictly up to candle index $i$. |
| `obv` | Group A (Technical) | On-Balance Volume on 5m candles | $T$ (Current candle close) | **NO** | Cumulative sum over past candles up to $i$. |
| `bollinger_position` | Group A (Technical) | Position within 20-period Bollinger Bands | $T$ (Current candle close) | **NO** | Uses 20-period rolling window strictly $\le T$. |
| `macd` | Group A (Technical) | 12-period EMA minus 26-period EMA | $T$ (Current candle close) | **NO** | Backward-looking exponential moving averages. |
| `macd_signal` | Group A (Technical) | 9-period EMA of MACD | $T$ (Current candle close) | **NO** | Computed strictly on past MACD series. |
| `macd_diff` | Group A (Technical) | MACD minus MACD signal | $T$ (Current candle close) | **NO** | Difference between contemporaneous values. |
| `price_vs_vwap` | Group A (Technical) | % distance of close from session VWAP | $\le T$ (Current session) | **NO** | Session VWAP resets daily at 09:15; cumulative to $T$. |
| `price_vs_ema5` | Group A (Technical) | % distance of close from 5-period EMA | $T$ (Current candle close) | **NO** | 5-period rolling exponential average. |
| `direction` | Group A (Technical) | Candidate trade direction (+1.0 LONG / -1.0 SHORT) | $T$ (Current candle) | **NO** | Explicit candidate evaluation side. |
| `sentiment_score` | Group B (News) | FinBERT sentiment ($P_{pos} - P_{neg}$) | $< T$ (Strictly prior news) | **NO** | Filtered by `news_timestamp < candle_timestamp`. |
| `has_news` | Group B (News) | Boolean indicator if prior news exists | $< T$ (Strictly prior news) | **NO** | Strict temporal cutoff; missing news produces NaN. |
| `number_of_articles` | Group B (News) | Count of prior articles | $< T$ (Strictly prior news) | **NO** | Count of articles published prior to candle timestamp. |
| `market_similarity` | Group C (Chroma) | Cosine similarity to market daily fingerprint | $< \text{date}(T)$ (Completed prior days) | **NO** | Filtered by `trading_date_int < query_date_int`. |
| `stock_similarity` | Group C (Chroma) | Cosine similarity to stock daily fingerprint | $< \text{date}(T)$ (Completed prior days) | **NO** | Filtered by `trading_date_int < query_date_int`. |
| *Neo4j features* | *Unavailable* | *Neo4j DB not connected* | *N/A* | *N/A* | **EXCLUDED** (Zero fabrication of unpopulated data). |

---

## 2. Leakage Defense Mechanisms

1. **Strict Train-Only Missing News Imputation**:
   - `SimpleImputer` is fitted **strictly on `df_train`**.
   - Imputation value is NEVER informed by Validation or Test distributions.
   - `has_news` is preserved as an explicit boolean feature so the model retains missingness awareness.

2. **Completed-Day Chroma Fingerprints**:
   - Daily fingerprints are constructed after market close of date $D$.
   - Intraday queries for date $D$ filter by `trading_date_int < query_date_int`, guaranteeing that only dates $\le D-1$ are accessible.
   - Current-day fingerprint retrieval is strictly prohibited and tested.

3. **Chronological Walk-Forward Purged OOF Calibration**:
   - Out-of-fold calibration uses 4 chronological walk-forward splits within `df_train`.
   - 240-minute purging horizon prevents trade outcome overlap across folds.

4. **Sacred Test Set Boundary**:
   - All models, class weights, calibrators, and decision thresholds are finalized on Validation (Days 43–51) before evaluating the Test set (Days 52–60).
   - Test data is never used for tuning or selection.

---
*Audit compiled automatically by Trade-Assist Phase 5 Verification Engine.*
