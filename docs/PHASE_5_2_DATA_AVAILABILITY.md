# Phase 5.2 — Historical Data Availability Audit

## 1. Executive Summary

This audit evaluates the historical data available in the Trade-Assist repository to determine whether a genuine multi-year validation ($\ge 2$ years, ideally 3–5 years) of the LONG predictive edge can be conducted.

### Core Verdict
- **Available Historical Span**: Exactly **59 trading days** (from **2026-06-15 09:15:00 IST** to **2026-09-04 15:15:00 IST**), spanning ~2.7 calendar months.
- **Multi-Year Data Status**: **UNAVAILABLE**. No intraday 5-minute OHLCV data prior to June 15, 2026 exists in the local repository.
- **Scientific Integrity Mandate**: Under Phase 5.2 absolute protection rules, Trade-Assist **strictly refuses to fabricate synthetic multi-year candles, clone historical periods, or manufacture artificial news/fingerprints**.
- **Action Taken**: The data limitation is formally documented here. The LONG edge robustness pipeline is executed across the **maximum trustworthy historical dataset available** (the 59-day / 48-symbol / 428,172-candidate dataset), evaluating symbol breadth, weekly/monthly temporal persistence, causal market regimes, time-of-day windows, pre-declared threshold frontiers, and statistical confidence intervals, while explicitly reporting the multi-year constraint.

---

## 2. Quantitative Data Inventory

### A. Intraday OHLCV Candle Data
| Parameter | Audited Value | Status / Notes |
| :--- | :--- | :--- |
| **Earliest Candle Timestamp** | `2026-06-15 09:15:00+05:30` | Start of Phase 2 dataset |
| **Latest Candle Timestamp** | `2026-09-04 15:15:00+05:30` | End of Phase 5 canonical Test set |
| **Total Calendar Duration** | 82 calendar days (~2.7 months) | Sub-annual |
| **Total Trading Days** | **59 trading days** | 75 five-minute bars per day per symbol |
| **Intraday Bar Interval** | 5 minutes | Consistent across all symbols |
| **Total Raw Candles** | 209,716 bars (`data/raw/combined_raw.parquet`) | 100% complete, zero missing OHLC bars |
| **Total Labeled Candidates** | 428,172 candidate rows (Long + Short) | 422,460 VALID label status |
| **Volume Availability** | 100% available (0 NaNs) | Continuous |

### B. Universe & Symbol Coverage
- **Total Symbols**: **48 symbols** (NSE Nifty 50 constituents with complete data).
- **Symbol List**: `ADANIENT.NS`, `ADANIPORTS.NS`, `APOLLOHOSP.NS`, `ASIANPAINT.NS`, `AXISBANK.NS`, `BAJAJ-AUTO.NS`, `BAJFINANCE.NS`, `BEL.NS`, `BHARTIARTL.NS`, `BPCL.NS`, `BRITANNIA.NS`, `CIPLA.NS`, `COALINDIA.NS`, `DIVISLAB.NS`, `DRREDDY.NS`, `EICHERMOT.NS`, `GRASIM.NS`, `HAL.NS`, `HCLTECH.NS`, `HDFCBANK.NS`, `HDFCLIFE.NS`, `HEROMOTOCO.NS`, `HINDALCO.NS`, `HINDUNILVR.NS`, `ICICIBANK.NS`, `INDUSINDBK.NS`, `INFY.NS`, `IOC.NS`, `ITC.NS`, `JSWSTEEL.NS`, `KOTAKBANK.NS`, `LT.NS`, `M&M.NS`, `MARUTI.NS`, `NESTLEIND.NS`, `NTPC.NS`, `ONGC.NS`, `POWERGRID.NS`, `RELIANCE.NS`, `SBILIFE.NS`, `SBIN.NS`, `SUNPHARMA.NS`, `TATASTEEL.NS`, `TCS.NS`, `TITAN.NS`, `TRENT.NS`, `ULTRACEMCO.NS`, `WIPRO.NS`.
- **Symbol Completeness**: All 48 symbols have synchronized timestamps across the entire 59-day period.

### C. Technical Indicators
All 18 base technical indicator columns are 100% complete with **0 missing values (0.00% NaN)** across the entire 428,172 rows:
- Exponential Moving Averages: `ema_5` (0 NaNs)
- Momentum & Oscillators: `rsi` (0 NaNs), `macd`, `macd_signal`, `macd_diff` (0 NaNs)
- Volume Dynamics: `obv` (0 NaNs)
- Volatility & Envelopes: `bollinger_middle`, `bollinger_upper`, `bollinger_lower`, `bollinger_position` (0 NaNs)
- Intraday Session Reference: `vwap`, `price_vs_vwap`, `price_vs_ema5` (0 NaNs)

### D. News & Sentiment Context Coverage
- **Raw News Articles**: 424 real financial articles stored in `data/processed/raw_news_cache.json`.
- **Symbols Covered in News**: 47 of 48 symbols.
- **Article Date Range**: Earliest article `2025-09-16 19:18:01 IST`, latest article `2026-09-04 02:57:34 IST`.
- **Candle Alignment (`has_news`)**:
  - `has_news == True`: 325,922 rows (76.12%)
  - `has_news == False`: 102,250 rows (23.88%)
- **Downstream Imputation**: Preserved as explicit `NaN` when missing; imputed strictly using train-only medians in downstream pipelines to prevent lookahead.

### E. Vector Context Coverage (ChromaDB)
- **Market Fingerprints**: 59 daily embeddings in `market_fingerprints` collection.
- **Stock Fingerprints**: 2,832 daily stock embeddings in `stock_fingerprints` collection.
- **Missing Values**: `market_similarity` (0 NaNs), `stock_similarity` (0 NaNs).
- **Causality Enforcement**: Strictly restricted to `trading_date_int < query_date_int`.

### F. Knowledge Graph Context (Neo4j)
- **Status**: **OFFLINE / UNAVAILABLE**.
- Ingestion pipelines gracefully fall back to zero-weighted graph features as audited in Phase 4 and Phase 5.

---

## 3. Data Limitations & Gap Analysis

| Dimension | Target Multi-Year Requirement | Audited Reality | Gap Severity | Impact on Phase 5.2 |
| :--- | :--- | :--- | :---: | :--- |
| **History Length** | $\ge 2$ years (500+ days) | 59 trading days (~2.7 months) | **CRITICAL** | Cannot perform multi-year regime shift or multi-year walk-forward testing. |
| **Annual Cycles** | 2–5 annual market cycles | 1 single seasonal quarter (Monsoon 2026) | **HIGH** | Strategy cannot be tested against historical bear markets (e.g. 2022 rate hike cycle or 2020 crash). |
| **Symbol Breadth** | $\ge 50$ liquid stocks | 48 Nifty constituents | **LOW** | Adequate for broad cross-sectional symbol robustness. |
| **News Depth** | Multi-year historical archives | 424 articles (mostly recent months) | **MEDIUM** | Older days have sparser news density. |

---

## 4. Recommended Next Data-Acquisition Steps

To enable a genuine multi-year validation study in future phases:
1. **Historical 5m OHLCV Acquisition**:
   - Ingest 2–5 years of adjusted 5-minute continuous futures or equity spot data (e.g., from NSE Data Feed, TrueData, or Zerodha historical API) covering 2021–2026.
2. **Historical News Archiving**:
   - Source historical Indian business press archives (Reuters India, Economic Times, Mint) matching the 2021–2026 candle dates to maintain realistic sentiment coverage.
3. **Daily Vector Pre-computation**:
   - Batch-compute daily market and stock fingerprints across the extended multi-year calendar and store in ChromaDB.

---

## 5. Protocol for Phase 5.2 Execution

Because multi-year data is not yet acquired, Trade-Assist operates strictly under **Truth-in-Data** principles:
1. **No Data Fabrication**: We will not artificially duplicate or simulate multi-year data.
2. **Exhaustive Available Robustness**: We execute the comprehensive robustness test suite across the complete 59-day / 48-symbol / 428,172-row dataset.
3. **Transparent Reporting**: All temporal breakdowns will be reported by Day, Week, and Month (June, July, August, September 2026).
4. **Final Classification Constraint**: Even if the LONG edge remains profitable and has PF > 1.5, the final classification cannot be `A. ROBUST LONG EDGE` across multi-year cycles; it will be classified honestly based on its empirical persistence across the available period.
