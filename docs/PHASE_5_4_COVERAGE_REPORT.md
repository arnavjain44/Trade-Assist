# Phase 5.4 — Empirical Historical Data Coverage Report

## 1. Executive Summary
This report presents the empirical coverage, data quality, and temporal distribution of the historical intraday data currently residing in Trade-Assist, as verified by `HistoricalDataValidator` and the Phase 5.4 acquisition infrastructure.

---

## 2. Global Coverage Metrics

| Metric | Empirical Value | Audit Notes |
| :--- | :--- | :--- |
| **Earliest Timestamp** | `2026-06-15 09:15:00+05:30` | Validated IST (`UTC+05:30`) |
| **Latest Timestamp** | `2026-09-04 15:15:00+05:30` | Validated IST (`UTC+05:30`) |
| **Calendar Span** | 82 calendar days (~2.7 months) | Hard Yahoo Finance lookback limit |
| **Trading Days Count** | **59 trading days** | Uniform across all 48 symbols |
| **Symbol Count** | 48 active equities | 2 skipped (`TATAMOTORS.NS`, `ZOMATO.NS` due to Yahoo 404s) |
| **Total 5m Candles (Raw)** | **209,716 rows** | Stored in `data/raw/` |
| **Mean Rows per Symbol** | 4,369.1 rows | Range: 4,361 to 4,374 bars |
| **Trading Days per Symbol** | 59 days | Uniform |
| **Missing Sessions** | 0 expected sessions dropped | All 59 trading days accounted for |
| **Duplicate Count** | 0 duplicates in clean dataset | Verified by `HistoricalDataValidator` |
| **Invalid Row Count** | 0 mathematical violations | Verified: $H \ge \max(O, C, L)$ and $L \le \min(O, C, H)$ |
| **Incomplete Sessions** | 0 unexpected partial sessions | Standard regular sessions (~74–75 bars) |
| **Timezone Status** | **100% Asia/Kolkata (IST)** | Zero naive timestamps |
| **OHLC Validity** | **100% Valid** | Open, High, Low, Close $> 0$ |
| **Volume Validity** | **100% Valid** | Volume $\ge 0$ |
| **Source Provenance** | `yfinance` (Legacy) / `kite` (Ready) | Logged per parquet partition |
| **Adjustment Status** | Split- & Bonus-Adjusted | Explicitly documented |

---

## 3. Symbol × Year Historical Coverage Matrix

The matrix below illustrates the current repository coverage across calendar years:

| Symbol | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **TCS.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **INFY.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **HDFCBANK.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **ICICIBANK.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **SBIN.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **BHARTIARTL.NS**| ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **ITC.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **KOTAKBANK.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **LT.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **AXISBANK.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **HINDUNILVR.NS**| ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **BAJFINANCE.NS**| ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **MARUTI.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **TATASTEEL.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **ASIANPAINT.NS**| ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **TITAN.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **SUNPHARMA.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **WIPRO.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| **HCLTECH.NS** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |
| *(Remaining 28 Symbols)* | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (59d)** |

**Key Finding**:
- In-Repo Coverage is **100% complete for the recent 59 trading days (2026)**.
- Historical years prior to 2026 are **0% acquired**.
- The acquisition infrastructure built in Phase 5.4 is ready to populate 2018–2025 as soon as broker credentials (Zerodha Kite) or an offline research archive (CSV/Parquet) are supplied.

---

## 4. Acquisition Pipeline Verification Status

| Pipeline Stage | Implementation Module | Verification Status |
| :--- | :--- | :--- |
| **Provider Agnostic Interface** | `app/ml/historical_data/base.py` | **READY** (BaseHistoricalProvider contract) |
| **Kite Connect Adapter** | `app/ml/historical_data/providers/kite_adapter.py` | **READY** (Dry-run verified; blocked only by live credentials) |
| **Local Archive Adapter** | `app/ml/historical_data/providers/local_csv_adapter.py` | **READY & TESTED** (Successfully ingested & validated local raw data) |
| **yfinance Adapter** | `app/ml/historical_data/providers/yfinance_adapter.py` | **READY** (Enforces 60-day limit rejection) |
| **Deterministic Chunking** | `HistoricalDownloader.generate_chunks` | **READY** (Verified 100-day chunk boundaries) |
| **Immutable Raw Storage** | `RawStorageManager` | **READY** (SHA-256 cryptographic verification tested) |
| **Data Quality Validation** | `HistoricalDataValidator` | **READY** (Special session & Muhurat handling added) |
| **Unit Test Suite** | `tests/test_phase5_4_acquisition.py` | **READY** (15 / 15 tests passing) |
