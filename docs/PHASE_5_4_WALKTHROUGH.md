# Phase 5.4 — Historical Data Acquisition & Provider Architecture: Walkthrough

## 1. Introduction: Why Data Was the Blocker
In Phase 5.1 and 5.2, we tested whether an intraday trading edge could deliver high precision and profitable economics. The models showed strong preliminary promise on the LONG side. However, when we investigated the foundation, we discovered that our entire historical dataset contained only **59 trading days** (~2.7 calendar months).

You cannot prove an investment strategy works across multiple market cycles (bull markets, bear markets, high-interest-rate environments, inflation shocks) using only 59 trading days. Therefore, Phase 5.3 concluded:
> **"Data infrastructure is ready; historical acquisition remains the blocker."**

The goal of Phase 5.4 was to build and verify the exact pipeline that can safely acquire $\ge 2$ years of 5-minute historical data for Indian equities (NSE / Nifty 50).

---

## 2. Which Data Source Was Selected and Why?
We audited all potential data sources:
1. **Yahoo Finance (`yfinance`) — DISQUALIFIED**: We empirically confirmed that Yahoo Finance enforces a hard server-side limit of 60 calendar days for 5-minute intraday candles. Any request older than 60 days returns an immediate error. It is physically impossible to get multi-year 5-minute data from Yahoo Finance.
2. **Zerodha Kite Connect API — PRIMARY BROKER SELECTION**: Zerodha provides official, exchange-cleared historical data back 7+ years (2017–present) for NSE equities. It is split- and bonus-adjusted.
3. **Local CSV / Research Archive Adapter — IMMEDIATE OFFLINE SELECTION**: We also built a native loader for local CSV/Parquet archives, allowing us to ingest freely available academic or open quantitative datasets immediately.

---

## 3. How Historical Data Is Downloaded (Architecture)

We created a provider-agnostic package in `app/ml/historical_data/`:
```
app/ml/historical_data/
├── base.py                     # BaseHistoricalProvider abstract contract
├── models.py                   # AcquisitionRequest, ChunkInfo, AcquisitionReport
├── storage.py                  # RawStorageManager (Immutable raw Parquet + SHA-256)
├── universe.py                 # HistoricalUniverseManager (Point-in-Time membership)
├── downloader.py               # HistoricalDownloader (Chunking, retries, validation)
└── providers/
    ├── kite_adapter.py         # Zerodha Kite Connect API adapter
    ├── yfinance_adapter.py     # Yahoo Finance adapter (with 60-day rejection rule)
    └── local_csv_adapter.py    # Local CSV / Parquet research archive adapter
```

---

## 4. How Pagination & Chunking Works
External market data APIs do not let you download 5 years of minute-by-minute data in one single call. 
- Zerodha Kite Connect limits 5-minute requests to **at most 100 days per call**.
- Our `HistoricalDownloader.generate_chunks()` takes any date range (e.g. 2022-01-01 to 2024-12-31) and slices it into contiguous, non-overlapping 100-day blocks.
- If a temporary network glitch or rate-limit happens on chunk 3, the downloader automatically retries with exponential backoff.
- At chunk boundaries, the downloader automatically drops overlapping candles so no timestamps are duplicated.

---

## 5. How Raw Data Is Preserved (Immutability)
In quantitative science, raw data must be **sacrosanct and immutable**:
- Raw downloads are stored in `data/raw/historical/`.
- Every raw file has a companion JSON metadata file (`.meta.json`) containing:
  - Download timestamp
  - Provider name
  - Requested start and end dates
  - Actual min and max timestamps
  - Cryptographic **SHA-256 checksum**
- If anyone accidentally alters or corrupts the raw data, the system detects the hash mismatch and halts immediately.
- Clean, validated data is stored separately in `data/processed/historical/`. Raw data is never overwritten.

---

## 6. Timestamp Semantics & Causality
A major source of lookahead bugs in quantitative finance is timestamp confusion:
- Does `09:15:00` represent the moment the bar started (open), or when it finished (close)?
- **Zerodha Kite** timestamps represent the **candle start time** (`09:15` covers trades from 09:15 to 09:20).
- **Trade-Assist Canonical Schema** standardizes on the **candle close/end time** (`09:20`).
- The `KiteHistoricalProvider` explicitly shifts timestamps by $+5$ minutes to ensure all decision models only see data after the bar has closed.

---

## 7. How Validation Works
We updated `app/ml/historical_data_validator.py` to audit all downloaded candles:
1. **Deduplication**: Removes duplicate (symbol, timestamp) collisions.
2. **OHLC Relational Math**: Confirms $High \ge \max(Open, Close, Low)$ and $Low \le \min(Open, Close, High)$.
3. **Price & Volume Sanity**: Prices must be strictly positive ($> 0$), volume non-negative ($\ge 0$).
4. **Timezone Verification**: Strictly asserts Indian Standard Time (`Asia/Kolkata` / `UTC+05:30`).
5. **Session Hours & Special Sessions**: Standard trading is 09:15 to 15:30 IST. Crucially, the validator now recognizes legitimate special sessions, such as **Diwali Muhurat Trading** (evening 18:00–19:15) and **Saturday Disaster Recovery Sessions**, without incorrectly flagging them as errors.
6. **Session Categorization**: Accurately classifies trading days into standard sessions (75 bars), special sessions, and partial sessions.

---

## 8. Corporate Actions & Survivorship Bias
- **Corporate Actions**: Stock splits and bonus issues are accounted for using split- and bonus-adjusted series. Cash dividends are not subtracted from intraday prices to prevent artificial negative prices and fractional distortion.
- **Survivorship Bias**: Tested models must not assume today's Nifty 50 constituents were the same in 2018. `HistoricalUniverseManager` supports `Universe(t)`, where constituent membership is determined by the official NSE semi-annual reconstitution circulars.

---

## 9. What Coverage Was Actually Obtained?
- **Current In-Repo Coverage**: 209,716 candles across 48 Nifty equities spanning 59 trading days (June to September 2026).
- **Historical Years Prior to 2026**: 0% acquired.
- **Dry-Run Test Results**:
  - `KiteHistoricalProvider` dry-run across 2023–2024 generated 8 contiguous 100-day chunks with verified boundaries.
  - Live execution without credentials failed safely with `status="BLOCKED_AUTH"`.
  - `LocalCsvHistoricalProvider` successfully loaded, verified, and saved raw and clean historical datasets.

---

## 10. Why ML Training Is Intentionally Deferred
We did **NOT** train models in this phase.
Training ML models on an unverified, 59-day dataset or fabricated data would be unscientific and misleading. The acquisition pipeline is now built, verified by 15 dedicated unit tests and 99 regression tests, and fully ready for real data ingestion.
