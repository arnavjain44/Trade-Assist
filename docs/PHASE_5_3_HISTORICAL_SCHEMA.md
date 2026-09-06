# Phase 5.3 — Canonical Historical Intraday Data Schema

## 1. Overview & Purpose

This document defines the **canonical historical data schema** for Trade-Assist.
All future historical data ingestion—regardless of origin (`yfinance`, Zerodha Kite Connect, TrueData, or open research archives)—must normalize into this standardized, auditable schema before passing into the validation, feature engineering, and modeling pipelines.

---

## 2. Core Intraday OHLCV Schema Specification

Each row represents an aggregated 5-minute trading candle for a specific equity ticker.

| Column Name | Data Type | Nullable? | Default Value | Description / Constraints |
| :--- | :--- | :---: | :---: | :--- |
| `timestamp` | `datetime64[ns, Asia/Kolkata]` | **NO** | *None* | Start timestamp of the 5-minute bar in Indian Standard Time (IST, UTC+05:30). Example: `2026-06-15 09:15:00+05:30`. |
| `symbol` | `string` / `category` | **NO** | *None* | Normalized equity symbol with exchange suffix (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`). |
| `open` | `float64` | **NO** | *None* | Opening traded price during the 5-minute window. Must be $> 0.0$. |
| `high` | `float64` | **NO** | *None* | Highest traded price during the 5-minute window. Must satisfy $high \ge \max(open, close, low)$. |
| `low` | `float64` | **NO** | *None* | Lowest traded price during the 5-minute window. Must satisfy $low \le \min(open, close, high)$ and $> 0.0$. |
| `close` | `float64` | **NO** | *None* | Closing traded price of the 5-minute window. Must be $> 0.0$. |
| `volume` | `float64` / `int64` | **NO** | `0.0` | Total shares traded during the 5-minute window. Must be $\ge 0.0$. |
| `trading_date` | `string` (`YYYY-MM-DD`) | **NO** | *Derived* | Calendar trading date extracted from timestamp (`timestamp.dt.date`). |
| `timezone` | `string` | **NO** | `"Asia/Kolkata"` | Explicit timezone identifier confirming no UTC ambiguity. |
| `source` | `string` | **NO** | `"yfinance"` | Data provider provenance identifier (`yfinance`, `zerodha_kite`, `truedata`, `curated_archive`). |

---

## 3. Corporate Action & Adjustment Metadata

To support both raw order-execution realism and continuous back-adjusted indicator calculations:

| Column Name | Data Type | Nullable? | Default Value | Description / Constraints |
| :--- | :--- | :---: | :---: | :--- |
| `is_adjusted` | `boolean` | **NO** | `True` | `True` if prices have been back-adjusted for splits, bonuses, and capital actions. |
| `adjustment_factor` | `float64` | **NO** | `1.0` | Cumulative price multiplier applied to historical prices ($P_{adj} = P_{raw} \times F_{adj}$). |
| `split_ratio` | `float64` | **NO** | `1.0` | Stock split or bonus ratio enacted on this candle date (e.g. `2.0` for 1:1 bonus). |
| `dividend_amount` | `float64` | **NO** | `0.0` | Cash dividend declared ex-date on this candle in INR. |
| `raw_close` | `float64` | YES | `NaN` | Original unadjusted closing price for historical execution auditing. |

---

## 4. Data Quality & Session Integrity Flags

Every candle is tagged with an auditable data quality flag by `HistoricalDataValidator`:

| Flag Name | Enum / Value | Description |
| :--- | :--- | :--- |
| `CLEAN` | `0` | Candle passed all validation checks; standard regular trading session bar. |
| `SUSPECT_VOLUME` | `1` | Volume is unusually high (> 20x 20-period median) or zero during regular market hours. |
| `ZERO_RANGE` | `2` | Flat bar ($open = high = low = close$) with positive volume (possible circuit filter lock). |
| `GAP_FILL` | `3` | Synthetic zero-volume bar filled to repair missing time slice (explicitly recorded). |
| `SPECIAL_SESSION` | `4` | Legitimate non-standard trading session (e.g. Diwali Muhurat trading, mock DR drill). |
| `INVALID_BAR` | `5` | Corrupted bar violating $low \le high$ or zero price; rejected from model ingestion. |

---

## 5. Storage & Partitioning Standards

- **Primary Storage Format**: Apache Parquet with Snappy or ZSTD compression.
- **Partitioning Strategy**:
  - Raw Ingestion: Partitioned by year and month (`data/raw/year=YYYY/month=MM/data.parquet`) or per-symbol (`data/raw/{SYMBOL}_raw_5m.parquet`).
  - Unified Canonical Feature Dataset: Single optimized columnar Parquet file (`data/processed/canonical_historical_5m.parquet`).
- **Index Invariant**: Continuous, strictly ascending chronological order per symbol (`df.sort_values(['symbol', 'timestamp'])`).
- **Immutability Rule**: Once validated and committed to a frozen research checkpoint, raw canonical parquet files are read-only.
