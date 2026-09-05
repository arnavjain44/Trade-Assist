# Phase 5.1 — Feature Provenance & Temporal Causality Audit

This document certifies the temporal provenance and causality of all features evaluated in Phase 5.1.
Every feature satisfies the strict temporal causality test: **"What information would have been available at the exact decision timestamp $T$?"**

---

## 1. Feature Provenance Matrix

| Feature | Category | Source | Timestamp Basis | Future Data Possible? | Causality Verification Rule |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `rsi` | Base Technical | 14-period RSI on 5m OHLCV | $T$ (Candle close) | **NO** | Calculated strictly up to candle $i$. |
| `obv` | Base Technical | On-Balance Volume on 5m candles | $T$ (Candle close) | **NO** | Cumulative sum over past candles up to $i$. |
| `bollinger_position` | Base Technical | Position within 20-period BB | $T$ (Candle close) | **NO** | Rolling 20-candle window strictly $\le T$. |
| `macd`, `macd_signal`, `macd_diff` | Base Technical | 12/26/9 EMA on close | $T$ (Candle close) | **NO** | Exponential moving averages on past closes. |
| `price_vs_vwap` | Base Technical | % distance of close from VWAP | $\le T$ (Current session) | **NO** | Daily VWAP resets at 09:15; cumulative to $T$. |
| `price_vs_ema5` | Base Technical | % distance of close from EMA5 | $T$ (Candle close) | **NO** | 5-period rolling exponential average. |
| `direction` | Candidate Flag | Hypothetical trade side (+1.0 / -1.0) | $T$ (Candle close) | **NO** | **Contemporaneous scenario parameter**: Known at decision time; strictly NOT derived from trade outcome or future price movement. |
| `sentiment_score` | Real News | FinBERT sentiment ($P_{pos} - P_{neg}$) | $< T$ (Strictly prior news) | **NO** | Enforces `pub_timestamp < candle_timestamp`. |
| `has_news` | Real News | Boolean indicator of prior news | $< T$ (Strictly prior news) | **NO** | Missing news preserved as explicit NaN/False. |
| `number_of_articles` | Real News | Count of prior articles | $< T$ (Strictly prior news) | **NO** | Count of articles published prior to candle. |
| `market_similarity` | Chroma Context | Cosine similarity to market fingerprint | $< \text{date}(T)$ (Prior days) | **NO** | Filtered by `trading_date_int < query_date_int`. |
| `stock_similarity` | Chroma Context | Cosine similarity to stock fingerprint | $< \text{date}(T)$ (Prior days) | **NO** | Filtered by `trading_date_int < query_date_int`. |
| `return_5m` | Enhanced Momentum | 1-candle return ($close_t / close_{t-1} - 1$) | $T$ (Candle close) | **NO** | Past 1-candle percentage price change. |
| `return_15m` | Enhanced Momentum | 3-candle return ($close_t / close_{t-3} - 1$) | $T$ (Candle close) | **NO** | Past 3-candle percentage price change. |
| `return_60m` | Enhanced Momentum | 12-candle return ($close_t / close_{t-12} - 1$) | $T$ (Candle close) | **NO** | Past 12-candle percentage price change. |
| `normalized_atr` | Enhanced Volatility | 14-period ATR divided by close | $T$ (Candle close) | **NO** | 14-period True Range on past OHLC. |
| `bollinger_bandwidth` | Enhanced Volatility | $(Upper - Lower) / Middle$ | $T$ (Candle close) | **NO** | 20-period bandwidth on past candles. |
| `ema5_slope` | Enhanced Trend | 3-candle slope of EMA5 | $T$ (Candle close) | **NO** | Difference between current and past EMA5. |
| `price_vs_ema20` | Enhanced Trend | % distance of close from EMA20 | $T$ (Candle close) | **NO** | 20-period EMA on past closes. |
| `rsi_delta_3` | Enhanced Momentum | $\text{RSI}_t - \text{RSI}_{t-3}$ | $T$ (Candle close) | **NO** | 3-candle change in past RSI. |
| `relative_volume` | Enhanced Volume | Volume / rolling 20-candle mean volume | $T$ (Candle close) | **NO** | 20-candle rolling volume mean. |
| `time_of_day_fraction` | Session Timing | Normalized minute of session (0 to 1) | $T$ (Candle timestamp) | **NO** | Intraday clock time (e.g. 10:15 / 375m). |
| `is_opening_session` | Session Timing | Boolean (first 45m of trading) | $T$ (Candle timestamp) | **NO** | Intraday clock time indicator. |

---

## 2. Direction Feature Contemporaneous Attestation

The `direction` feature represents the **hypothetical trade side being evaluated** (+1.0 for Long, -1.0 for Short):
1. **Decision Time Input**: When evaluating whether an entry signal is valid, the model evaluates $P(\text{Target Hit} \mid \text{Features}, \text{Direction}=+1)$ or $P(\text{Target Hit} \mid \text{Features}, \text{Direction}=-1)$.
2. **Zero Outcome Leakage**: It is set prior to looking at future price action; it is not a prediction of market trend.
3. **Symmetric Candidate Evaluation**: For every eligible candle, both a Long and Short scenario are independently generated and evaluated.

---
*Audit compiled automatically by Trade-Assist Phase 5.1 Verification Engine.*
