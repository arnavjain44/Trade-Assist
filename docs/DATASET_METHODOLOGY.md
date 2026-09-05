# Dataset Methodology & Feature Engineering Specifications (Phase 1)

## 1. Overview
This document specifies the dataset construction, historical ingestion, feature engineering, and data quality validation protocols implemented in **Phase 1** of the `Trade-Assist` intraday trading system.

The pipeline ingests real historical intraday market data for NIFTY 50 Indian equities, standardizes timezone boundaries, computes 6 curated technical indicators + normalized ML feature representations, enforces zero lookahead bias, and guarantees strict intraday VWAP session-reset logic.

---

## 2. Market Data Ingestion & Universe
- **Provider**: `yfinance` (Yahoo Finance API).
- **Timeframe**: 5-minute intraday candles (`5m`).
- **Lookback Period**: 60 trading days (`60d` — maximum supported by yfinance for 5-minute granularity).
- **Universe**: NIFTY 50 Equities (e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, etc.).
- **Timezone**: All timestamps are standardized to **Asia/Kolkata (IST)** with timezone-awareness enforced (`timestamp.dt.tz_convert("Asia/Kolkata")`).

---

## 3. Schema & Feature Definitions

### 3.1 Raw Market Data Schema (`data/raw/`)
| Column Name | Data Type | Description |
|---|---|---|
| `timestamp` | Datetime (IST) | Candle open timestamp with Asia/Kolkata timezone |
| `symbol` | String | NSE Ticker Symbol (e.g., `RELIANCE.NS`) |
| `open` | Float64 | Opening price of the 5-minute bar |
| `high` | Float64 | Highest price during the 5-minute bar |
| `low` | Float64 | Lowest price during the 5-minute bar |
| `close` | Float64 | Closing price of the 5-minute bar |
| `volume` | Float64 | Total traded volume during the 5-minute bar |

### 3.2 Engineered Features & Technical Indicators Schema (`data/processed/`)
| Feature Name | Formula / Definition | Range | Causality Guarantee |
|---|---|---|---|
| `ema_5` | 5-period Exponential Moving Average of `close` | $[0, +\infty)$ | Causal (uses past close prices) |
| `rsi` | 9-period Relative Strength Index ($100 - \frac{100}{1 + RS}$) | $[0, 100]$ | Causal (rolling 9-period gains/losses) |
| `obv` | Cumulative On-Balance Volume based on price sign change | $(-\infty, +\infty)$ | Causal (cumulative past signed volume) |
| `bollinger_middle` | 20-period Simple Moving Average (SMA) of `close` | $[0, +\infty)$ | Causal (rolling 20 past bars) |
| `bollinger_upper` | `bollinger_middle` + $2.0 \times \text{StdDev}(20)$ | $[0, +\infty)$ | Causal |
| `bollinger_lower` | `bollinger_middle` - $2.0 \times \text{StdDev}(20)$ | $[0, +\infty)$ | Causal |
| `macd` | 12-period EMA - 26-period EMA of `close` | $(-\infty, +\infty)$ | Causal |
| `macd_signal` | 9-period EMA of `macd` line | $(-\infty, +\infty)$ | Causal |
| `macd_diff` | `macd` - `macd_signal` (Histogram) | $(-\infty, +\infty)$ | Causal |
| **`vwap`** | **$\frac{\sum (\text{Typical Price} \times \text{Volume})}{\sum \text{Volume}}$ (Session-reset daily)** | $[0, +\infty)$ | **Causal & Strictly Intraday** |
| `bollinger_position` | $\frac{\text{close} - \text{bollinger\_lower}}{\text{bollinger\_upper} - \text{bollinger\_lower}}$ | $[0.0, 1.0]$ | Normalized indicator position |
| `price_vs_vwap` | $\frac{\text{close} - \text{vwap}}{\text{close}}$ | $(-\infty, +\infty)$ | Relative price offset from VWAP |
| `price_vs_ema5` | $\frac{\text{close} - \text{ema\_5}}{\text{close}}$ | $(-\infty, +\infty)$ | Relative price offset from EMA-5 |
| `sentiment_score` | FinBERT news sentiment compound score | $[-1.0, 1.0]$ | Reserved placeholder for ML phase |
| `market_similarity` | Vector distance to historical market fingerprint | $[0.0, 1.0]$ | Reserved placeholder for Neo4j/RAG |
| `stock_similarity` | Vector distance to historical per-stock fingerprint | $[0.0, 1.0]$ | Reserved placeholder for Neo4j/RAG |

---

## 4. Session-Resetting VWAP Methodology
VWAP calculation is strictly intraday:
1. Typical price is calculated per candle: $TP_t = \frac{\text{High}_t + \text{Low}_t + \text{Close}_t}{3}$.
2. Candles are grouped by calendar date in IST (`df["timestamp"].dt.date`).
3. Cumulative sum of $(TP \times \text{Volume})$ and cumulative sum of $\text{Volume}$ are accumulated **only within the same trading session**.
4. At 09:15 AM IST (session start), cumulative sums reset to zero, ensuring VWAP at the first candle equals the typical price of that candle.
5. **Cross-day leakage prevention**: VWAP never carries over across overnight market closures.

---

## 5. Missing Data & Leakage Policies
- **No Forward-Filling / Linear Interpolation**: Intraday bars missing due to exchange halts or exchange closures are left un-fabricated. Synthetic interpolation creates fake prices that distort technical indicator calculations.
- **Zero Lookahead Bias**: Feature calculations sort rows strictly by `timestamp` ascending. No centered windows or future period shift functions are permitted.
- **Single Reusable Pipeline**: The same `FeatureEngine.calculate_features()` entrypoint processes both offline Parquet datasets and real-time live trading API candles.

---

## 6. Automated Quality Audit Suite (`dataset_quality.json`)
The `DataQualityValidator` automatically verifies dataset integrity before model consumption:
1. **Duplicate Timestamps Check**: Ensures no multiple rows exist for the same timestamp per symbol.
2. **Price & Volume Validation**: Checks for impossible negative or zero prices, negative volume, or invalid data types.
3. **Timezone Verification**: Confirms timezone awareness (`Asia/Kolkata`).
4. **NaN / Inf Feature Audit**: Calculates exact coverage and null ratios across technical feature columns.
5. **Cross-Day VWAP Leakage Test**: Validates that $VWAP_0 = TP_0$ at every session start bar.
