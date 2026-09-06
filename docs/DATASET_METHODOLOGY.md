# Dataset Methodology & Feature Engineering Specifications (Phase 1 — Strengthened)

## 1. Overview
This document specifies the dataset construction, historical ingestion, feature engineering, data quality validation, and mathematical audit protocols implemented in **Phase 1** of the `Trade-Assist` intraday trading system.

The pipeline ingests real historical intraday market data for Indian equities (NIFTY 50 universe), standardizes timezone boundaries to **Asia/Kolkata (IST)**, computes 6 curated technical indicators + normalized ML feature representations, enforces zero lookahead bias, guarantees per-symbol calculation isolation, and enforces strict intraday VWAP session-reset logic.

---

## 2. Market Data Ingestion & Ticker Universe

### 2.1 Provider & Timeframe Specifications
- **Provider**: `yfinance` (Yahoo Finance API).
- **Timeframe**: 5-minute intraday candles (`5m`).
- **Lookback Period**: 60 trading days (`60d` — maximum supported by yfinance for 5-minute granularity).
- **Timezone**: All timestamps are standardized to **Asia/Kolkata (IST)** with explicit timezone-awareness (`timestamp.dt.tz_convert("Asia/Kolkata")`).

### 2.2 Target Universe vs Actual Ingested Universe
- **Target Universe**: NIFTY 50 Equities (50 tickers configured in `settings.DEFAULT_NSE_TICKERS`).
- **Available Ingested Universe**: **48 Equities** successfully ingested.
- **Unavailable / Skipped Symbols**: 2 symbols (`TATAMOTORS.NS` and `ZOMATO.NS`) returned HTTP 404 / no market data from yfinance due to ticker symbol changes / delisting on Yahoo Finance.
- **Policy on Missing Symbols**: In strict compliance with system rules, **zero fake tickers, replacement symbols, or synthetic candles were fabricated**. The dataset strictly comprises the 48 real market equities.

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
| Feature Name | Formula / Definition | Range | Causality & Isolation Guarantee |
|---|---|---|---|
| `ema_5` | 5-period Exponential Moving Average of `close` | $[0, +\infty)$ | Causal (uses past close prices); Per-symbol isolated |
| `rsi` | 9-period Relative Strength Index ($100 - \frac{100}{1 + RS}$) | $[0, 100]$ | Causal (rolling 9-period gains/losses); Per-symbol isolated |
| `obv` | Cumulative On-Balance Volume based on price sign change | $(-\infty, +\infty)$ | Causal (cumulative past signed volume); Per-symbol isolated |
| `bollinger_middle` | 20-period Simple Moving Average (SMA) of `close` | $[0, +\infty)$ | Causal (rolling 20 past bars); Per-symbol isolated |
| `bollinger_upper` | `bollinger_middle` + $2.0 \times \text{StdDev}(20)$ | $[0, +\infty)$ | Causal |
| `bollinger_lower` | `bollinger_middle` - $2.0 \times \text{StdDev}(20)$ | $[0, +\infty)$ | Causal |
| `macd` | 12-period EMA - 26-period EMA of `close` | $(-\infty, +\infty)$ | Causal |
| `macd_signal` | 9-period EMA of `macd` line | $(-\infty, +\infty)$ | Causal |
| `macd_diff` | `macd` - `macd_signal` (Histogram) | $(-\infty, +\infty)$ | Causal |
| **`vwap`** | **$\frac{\sum_{\tau=t_{start}}^t (\text{TP}_\tau \times V_\tau)}{\sum_{\tau=t_{start}}^t V_\tau}$ (Session-reset daily)** | $[0, +\infty)$ | **Causal, Per-Symbol Isolated & Strictly Intraday** |
| `bollinger_position` | $\frac{\text{close} - \text{bollinger\_lower}}{\text{bollinger\_upper} - \text{bollinger\_lower}}$ | $[0.0, 1.0]$ | Normalized indicator position |
| `price_vs_vwap` | $\frac{\text{close} - \text{vwap}}{\text{close}}$ | $(-\infty, +\infty)$ | Relative price offset from VWAP |
| `price_vs_ema5` | $\frac{\text{close} - \text{ema\_5}}{\text{close}}$ | $(-\infty, +\infty)$ | Relative price offset from EMA-5 |
| `sentiment_score` | FinBERT news sentiment compound score | $[-1.0, 1.0]$ | Reserved placeholder (`NaN` in Phase 1) |
| `market_similarity` | Vector distance to historical market fingerprint | $[0.0, 1.0]$ | Reserved placeholder (`NaN` in Phase 1) |
| `stock_similarity` | Vector distance to historical per-stock fingerprint | $[0.0, 1.0]$ | Reserved placeholder (`NaN` in Phase 1) |

---

## 4. Session-Resetting VWAP Mathematical Specification
VWAP calculation is strictly intraday and resets at every trading session boundary:
1. Typical Price is computed per 5-minute candle: $\text{TP}_t = \frac{\text{High}_t + \text{Low}_t + \text{Close}_t}{3}$.
2. Candles are grouped by local calendar date in **Asia/Kolkata (IST)** (`df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date`).
3. Within each session date:
   $$\text{VWAP}_t = \frac{\sum_{\tau=t_{start}}^t (\text{TP}_\tau \times \text{Volume}_\tau)}{\sum_{\tau=t_{start}}^t \text{Volume}_\tau}$$
4. **Session Start Candle ($t_{start} = \text{09:15 IST}$)**:
   - When $\text{Volume}_{t_{start}} > 0$, cumulative sums reset to zero, guaranteeing $\text{VWAP}_{t_{start}} = \text{TP}_{t_{start}}$.
5. **Zero-Volume Candle Handling**:
   - If a zero-volume candle occurs mid-session ($\text{Volume}_t = 0$), VWAP holds the previous valid session VWAP (`ffill`).
   - If a zero-volume candle occurs at session start, VWAP falls back to the candle's `close` price.
6. **Cross-Day Leakage Prevention**: VWAP never carries over across overnight market closures.

---

## 5. Per-Symbol Feature Calculation Isolation
- `FeatureEngine.calculate_features(df)` detects multi-symbol DataFrames (`"symbol"` column containing multiple tickers).
- When a multi-symbol dataset is passed, `FeatureEngine` automatically splits the DataFrame by symbol, computes all rolling, exponential moving average, cumulative OBV, and session VWAP indicators independently for each symbol, and concatenates the results.
- **Guarantee**: Indicators for Symbol B (e.g. `TCS.NS`) never inherit state or cumulative volume from Symbol A (e.g. `RELIANCE.NS`).

---

## 6. Zero Lookahead Bias & Causality Protocol
- All feature calculations sort rows strictly by `timestamp` ascending prior to computing features.
- No centered windows or future period shift functions are permitted.
- **Causality Verification Test**: Tested via `test_full_feature_causality`. Adding future candles after timestamp $T$ yields 100% identical feature values at $T$ and earlier across all 6 base indicators and 3 normalized features.

---

## 7. Session Boundaries & Timezone Handling
- **Timezone**: All timestamps are timezone-aware (`Asia/Kolkata`).
- **Session Boundaries**: Standard NSE intraday session spans 09:15 IST to 15:30 IST.
- **Calendar Date Grouping**: Intraday sessions are identified by the IST calendar date.
- **Trading Calendar Limitation Note**: Market holidays contain 0 trading bars from `yfinance`. The system groups session candles by local IST date without fabricating missing holiday dates.

---

## 8. Data Quality & Integrity Auditor (`DataQualityValidator`)
The `DataQualityValidator` performs exhaustive automated verification:
1. **Per-Symbol Duplicate Timestamp Check**: Duplicate timestamps are audited per symbol. Shared timestamps across different symbols (e.g. RELIANCE at 09:15 and TCS at 09:15) are correctly treated as valid parallel bars.
2. **OHLC Relationship Integrity**: Checks for logical candle errors:
   - $\text{High} < \max(\text{Open}, \text{Close})$
   - $\text{Low} > \min(\text{Open}, \text{Close})$
   - $\text{High} < \text{Low}$
   Violations are logged and reported as `invalid_ohlc_relationships`.
3. **Mathematical Session-Reset VWAP Audit**: Iterates through every symbol and every trading session, independently computing $\frac{\text{cumsum}(\text{TP} \times V)}{\text{cumsum}(V)}$ for every candle and asserting $\text{stored\_vwap} \approx \text{math\_vwap}$ via `np.isclose(atol=1e-3, rtol=1e-4)`.
4. **Price, Volume & Timezone Sanity**: Verifies zero negative prices, negative volumes, naive timestamps, and feature null ratios.

---

## 9. Dataset Reproducibility & Snapshot Metadata
- **Reproducibility Definition**: Re-running the dataset pipeline over the same market snapshot produces equivalent feature calculations. (Note: External market data providers can apply retroactive corporate action adjustments, so snapshot equivalence rather than byte-level parity across future API calls is documented).
- **Snapshot Metadata (`dataset_quality.json`)**:
  - **Provider**: `yfinance` v0.2.x
  - **Timeframe**: 5-minute (`5m`)
  - **Requested Period**: 60 trading days (`60d`)
  - **Date Range**: `2026-06-15 09:15:00+05:30` to `2026-09-04 15:15:00+05:30`
  - **Total Processed Rows**: **209,716 rows**
  - **Total Symbols**: **48 Equities**
  - **Code Version**: `1.0.0` (Phase 1)
