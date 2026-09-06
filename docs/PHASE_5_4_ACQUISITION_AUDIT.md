# Phase 5.4 — Historical Data Acquisition Audit

## 1. Executive Summary
Phase 5.3 established the data-quality validator, temporal causality contract, and canonical schema for Trade-Assist. However, the existing historical dataset in the repository comprises only 59 trading days (~2.7 calendar months, 209,716 candles across 48 tickers).

This audit evaluates all existing data acquisition mechanisms, loaders, and configurations in Trade-Assist to determine what must be retained, what must be replaced, and why a provider-agnostic multi-year historical acquisition pipeline is required.

---

## 2. Ingestion Code & API Audit

### A. Existing Downloader & Fetcher Modules
1. **`app/ml/dataset_builder.py` (`DatasetBuilder`)**:
   - **Provider**: `yfinance` (`yf.Ticker.history(period="60d", interval="5m")`).
   - **Coverage**: Maximum 60 calendar days of 5-minute candles.
   - **Timezone**: Converts timestamps to `Asia/Kolkata` (IST, UTC+05:30).
   - **Persistence**: Writes raw parquet files to `data/raw/{SYMBOL}_raw_5m.parquet` and `data/raw/combined_raw.parquet`.
   - **Pagination**: None. Requests all 60 days in a single monolithic API call.
   - **Limitation**: Hard Yahoo Finance server rejection when requesting >60 days of 5m intraday data. Cannot provide $\ge 2$ years.

2. **`app/core/data_fetcher.py` (`StockDataFetcher`)**:
   - **Provider**: `yfinance` (`yf.Ticker.history(period="1mo", interval="1d")`).
   - **Role**: Live production assistant fetcher for daily OHLCV bars across `DEFAULT_NSE_TICKERS`.
   - **Timeout**: 8-second hard timeout per ticker via `asyncio.wait_for`.
   - **Limitation**: Scrapes live unauthenticated Yahoo Finance daily data; irrelevant for multi-year intraday research.

3. **`app/ml/news_processor.py` (`YFinanceNewsProvider`)**:
   - **Provider**: `yfinance` (`yf.Ticker.news`).
   - **Coverage**: ~10 recent news articles per symbol (total 424 articles in `raw_news_cache.json`).
   - **Limitation**: No historical news archive queries. Yahoo Finance has no historical news API.

4. **`app/ml/historical_dataset_builder.py` (`HistoricalDatasetBuilder`)** *(Built in Phase 5.3)*:
   - **Role**: Architecture orchestrator that loads raw historical OHLCV, runs `HistoricalDataValidator`, computes technical features, aligns point-in-time context, labels trades via `HistoricalTradeLabeler`, and asserts data schema.
   - **Limitation**: Requires pre-existing raw OHLCV files; does not fetch or paginate from external data APIs directly.

---

## 3. Existing On-Disk Assets & Directories

### A. Raw Data Directory (`data/raw/`)
- `combined_raw.parquet`: 209,716 rows across 48 Nifty tickers spanning 2026-06-15 to 2026-09-04.
- 48 individual `{SYMBOL}_raw_5m.parquet` files (~4,369 rows each).
- 2 tickers missing (`TATAMOTORS.NS`, `ZOMATO.NS`) due to Yahoo Finance ticker resolution/404 failures.

### B. Processed Data Directory (`data/processed/`)
- `combined_processed.parquet`: 27.3 MB, processed technical features.
- `labeled_dataset.parquet`: 41.0 MB, triple-barrier labeled dataset.
- `phase5_features.parquet`: 41.3 MB, feature dataset with context joins.
- `raw_news_cache.json`: 424 articles, 2025–2026 only.
- `finbert_sentiment_cache.json`: 70.7 KB, pre-computed sentiment scores.
- Phase 3, 4, 5, 5.1, 5.2 result artifacts.

### C. Chroma & Neo4j
- `chroma_db/chroma.sqlite3`: 59 daily market fingerprints and 2,832 daily stock fingerprints.
- Neo4j: Configured via `app/config.py` (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`), but unpopulated (falls back to deterministic mock graph relationships).

---

## 4. Environment Variables & Authentication

| Variable | Location | Current Value | Required for Phase 5.4 |
| :--- | :--- | :--- | :--- |
| `DEFAULT_NSE_TICKERS` | `app/config.py` | 50 tickers | Universe candidate pool |
| `HISTORICAL_DATA_PROVIDER` | Missing | None | Provider selection (`kite`, `truedata`, `local_csv`, `yfinance`) |
| `HISTORICAL_DATA_API_KEY` | Missing | None | Broker / Vendor API Key |
| `HISTORICAL_DATA_API_SECRET` | Missing | None | Broker / Vendor API Secret |
| `HISTORICAL_DATA_ACCESS_TOKEN` | Missing | None | Active session access token |
| `HISTORICAL_DATA_RAW_DIR` | Missing | `data/raw/historical` | Dedicated raw historical directory |
| `HISTORICAL_DATA_CLEAN_DIR` | Missing | `data/processed/historical` | Dedicated clean historical directory |

**Safety Protocol**: An `.env.example` file will be created documenting these variables. Actual credentials will never be committed. The acquisition system will fail gracefully with clear error messages if required credentials are not supplied.

---

## 5. Technical Comparison: Existing vs. Required Pipeline

| Capability | Current State (`yfinance`) | Required Phase 5.4 State |
| :--- | :--- | :--- |
| **Historical Depth (5m)** | 60 calendar days (Hard API limit) | $\ge 2$ years (target: 3–7 years) |
| **Provider Architecture** | Hardcoded to `yfinance` | Provider-agnostic abstract adapter interface |
| **Pagination / Chunking** | Monolithic (single call) | Deterministic date chunking with retry & resume |
| **Raw Storage** | Overwritten in `data/raw/` | Immutable raw storage in `data/raw/historical/` with metadata |
| **Corporate Actions** | Implicit / unverified adjustments | Explicit corporate action policy & metadata |
| **Symbol Universe** | Static contemporary 50 tickers | Support for Point-in-Time membership $U(t)$ |
| **Validation** | Basic checks in builder | Rigorous multi-stage checks via `HistoricalDataValidator` |
| **Session Auditing** | Assumed 75 bars | Accounts for partial/special sessions (Muhurat, Saturday mock/DR) |

---

## 6. Conclusion
The existing `yfinance` download logic in `app/ml/dataset_builder.py` is hard-capped at 60 calendar days and cannot be extended for multi-year validation. A dedicated, modular acquisition layer (`app/ml/historical_data/`) must be constructed with clear separation of concerns, immutable raw storage, and provider adapters.
