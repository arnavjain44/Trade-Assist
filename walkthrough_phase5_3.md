# Phase 5.3 — Historical Data Expansion & Data-Quality Foundation Walkthrough

## 1. Executive Summary & Objective

Phase 5.2 demonstrated that the primary bottleneck in Trade-Assist is **historical data depth**, not model architecture or hyperparameter tuning. The existing dataset spans only **59 trading days** (~2.7 months), making true multi-year walk-forward validation impossible.

The purpose of **Phase 5.3** was to:
1. Conduct an exhaustive audit of all data sources, APIs, and caches in the repository.
2. Formally discover and document the technical and architectural reasons for lookback limits.
3. Evaluate technically and legally realistic sources for acquiring $\ge 2$ years (target: 3–5 years) of 5-minute intraday NSE equity data.
4. Establish a canonical historical schema, a strict temporal causality contract, and an automated data quality validation suite.
5. Create a clean, deterministic dataset builder architecture ready to ingest multi-year data once acquired.

> [!IMPORTANT]
> **No Synthetic Observations**: In accordance with Phase 5.3 rules, no models were trained, no prediction metrics were optimized, and no synthetic candles or fake news articles were manufactured.
>
> **Core Conclusion**: **"Data infrastructure is ready; historical acquisition remains the blocker."**

---

## 2. Key Findings: Ingestion Pipeline Audit

### A. The 60-Day Lookback Wall in `yfinance`
Empirical testing on Yahoo Finance's underlying API revealed a hard lookback constraint:
- Requesting `interval="5m"` with start dates beyond 60 days returns:
  `$RELIANCE.NS: 5m data not available for startTime=... and endTime=.... The requested range must be within the last 60 days.`
- **Conclusion**: `yfinance` is fundamentally incapable of supplying $\ge 2$ years of 5-minute intraday data. Continuing to rely on `yfinance` will perpetually cap historical analysis at 60 calendar days.

### B. The 10-Article News Ceiling
- The free `yf.Ticker(symbol).news` endpoint returns only the **~10 most recent articles** per symbol.
- It does not support historical archive querying. Older trading days in the dataset have `has_news = False` and `sentiment_score = NaN`.

### C. Chroma & Neo4j State
- **ChromaDB**: Holds 59 daily whole-market and 2,832 daily per-stock vector embeddings. Architecture is 100% ready for historical expansion.
- **Neo4j**: Offline / No URI configured in `.env`. Graceful offline deterministic fallback is verified and functional.

---

## 3. Evaluated Historical Data Options for Multi-Year Research

| Data Source | Feasible 5m History | Cost / Setup | Survivorship & Corporate Actions | Feasibility & Recommendation |
| :--- | :---: | :---: | :--- | :--- |
| **Yahoo Finance (`yfinance`)** | **Max 60 calendar days** | Free | Poor; delisted 404s; unadjusted/adjusted ambiguity | **DISQUALIFIED for multi-year** |
| **Zerodha Kite Connect Historical API** | **Up to 7+ years** | Paid (₹2,000/mo API + ₹2,000/mo Historical) | Split/bonus adjusted; official NSE broker data | **TOP PAID CANDIDATE** for live algorithmic trading |
| **TrueData / GlobalDataFeeds** | **10+ years** | Commercial (₹1,800–₹3,500/mo) | Authorized NSE vendor; institutional grade | **TOP INSTITUTIONAL CANDIDATE** |
| **Curated Open Historical Research Archives** | **5 to 10 years** | Free / Open Data | Open research dumps; requires validation | **TOP FREE CANDIDATE** for academic/offline ML research |

---

## 4. Symbol Universe & Survivorship Bias Solution

Using the current 2026 Nifty 50 constituents to backtest 2021 introduces severe **survivorship bias**:
- **The Problem**: Current members (e.g. `TRENT.NS`, `HAL.NS`, `BEL.NS`) were promoted into the index precisely because they had massive multi-year rallies. Underperforming companies that were demoted during 2021–2024 (e.g. `UPL`, `SHREECEM`, `ZEEL`) would be excluded, artificially exaggerating long-side returns.
- **The Solution**: Trade-Assist establishes **Point-in-Time Index Membership** (semi-annual rebalancing in March and September) as the canonical target standard, with fixed contemporary membership allowed only as a documented baseline.

---

## 5. Built Components & Deliverables

### A. Comprehensive Data Source Audit
- **File**: [`docs/PHASE_5_3_DATA_SOURCE_AUDIT.md`](file:///d:/proj1/proj%20files/docs/PHASE_5_3_DATA_SOURCE_AUDIT.md)
- Details all current data feeds, API constraints, and realistic multi-year expansion paths.

### B. Canonical Historical Schema
- **File**: [`docs/PHASE_5_3_HISTORICAL_SCHEMA.md`](file:///d:/proj1/proj%20files/docs/PHASE_5_3_HISTORICAL_SCHEMA.md)
- Standardizes timestamps (`datetime64[ns, Asia/Kolkata]`), OHLCV bounds, corporate action adjustment ratios (`adjustment_factor`, `split_ratio`), and quality flags (`CLEAN`, `SUSPECT_VOLUME`, `ZERO_RANGE`, `SPECIAL_SESSION`).

### C. Strict Temporal Causality Contract
- **File**: [`docs/PHASE_5_3_TEMPORAL_CONTRACT.md`](file:///d:/proj1/proj%20files/docs/PHASE_5_3_TEMPORAL_CONTRACT.md)
- Defines explicit timestamp rules: OHLCV $\le T$, Indicators $\le T$, News $< T$, Chroma fingerprints $<\text{date}(T)$, Neo4j historical trades $< T$.

### D. Automated Historical Data Validator
- **File**: [`app/ml/historical_data_validator.py`](file:///d:/proj1/proj%20files/app/ml/historical_data_validator.py)
- Detects duplicate candles, mathematically invalid OHLC math ($high < low$), zero/negative prices, impossible volume, naive/non-IST timezones, weekend/holiday bars, and partial trading sessions (standard 75 bars/session).
- Zero silent mutations: every anomaly is audited and recorded.

### E. Deterministic Dataset Builder Architecture
- **File**: [`app/ml/historical_dataset_builder.py`](file:///d:/proj1/proj%20files/app/ml/historical_dataset_builder.py)
- End-to-end pipeline: raw OHLCV $\rightarrow$ validation $\rightarrow$ clean candles $\rightarrow$ causal indicators $\rightarrow$ VWAP (09:15 session reset) $\rightarrow$ news alignment $\rightarrow$ Chroma fingerprints $\rightarrow$ Neo4j graph context $\rightarrow$ Phase 2 triple-barrier labels.

### F. Unit Test Suite
- **File**: [`tests/test_phase5_3_data.py`](file:///d:/proj1/proj%20files/tests/test_phase5_3_data.py)
- 8 dedicated unit tests verifying schema integrity, deduplication, price/volume validation, timezone normalization, off-hours filtering, temporal news causality, and deterministic execution.

---

## 6. Verification & Test Suite Status
- **Full Test Suite**: **99 passed** across the entire repository in 16.58s (`python -m pytest`).
- Zero regressions across Phase 2, 3, 4, 5, 5.1, or 5.2 test suites.
- Live production code (`app/api/`, `app/agent/`, `frontend/`) remains **100% untouched**.
