# Phase 5.3 — Comprehensive Data Source Audit & Historical Expansion Feasibility

## 1. Executive Summary

Phase 5.2 empirically demonstrated that while the LONG intraday prediction edge exhibits positive economics (Net Profit Factor > 1.6 to 2.3 at high thresholds), the current historical dataset of **59 trading days** (~2.7 months) is insufficient for genuine multi-year walk-forward validation. 

The primary bottleneck for Trade-Assist is **historical data depth**, not ML architecture or hyperparameter tuning.

This audit:
1. Conducts an exhaustive inspection of all existing data sources, scripts, caches, and APIs in the repository.
2. Formally documents why the current data collection pipeline is capped at 60 calendar days.
3. Evaluates technically and legally realistic sources for acquiring $\ge 2$ years (ideally 3–5 years) of 5-minute intraday NSE equity data.
4. Analyzes the symbol universe problem (survivorship bias in Nifty 50 constituents).
5. Assesses the feasibility of expanding news sentiment, Chroma vector context, and Neo4j graph data to multi-year horizons.

---

## 2. Audit of Existing Repository Data Ingestion

### A. Core Ingestion Pipeline (`yfinance`)
- **Primary Library**: `yfinance` (v0.2.37+ in `requirements.txt`).
- **Implementation Modules**:
  - `app/ml/dataset_builder.py` (`DatasetBuilder.fetch_symbol_raw_history`)
  - `app/core/data_fetcher.py` (`DataFetcher.fetch_ticker_data_async`)
  - `app/ml/news_processor.py` (`YFinanceNewsProvider`)
- **API Endpoints Utilized**:
  1. `yf.Ticker(symbol).history(period="60d", interval="5m")`: Intraday candle extraction.
  2. `yf.Ticker(symbol).news`: Recent news headline and summary extraction.
- **Hard Lookback Limits Discovered**:
  - **Intraday 5-minute lookback is strictly limited to 60 calendar days** by Yahoo Finance's underlying API server. When attempting `start="2024-01-01"` with `interval="5m"`, Yahoo Finance returns:
    ```
    $RELIANCE.NS: 5m data not available for startTime=... and endTime=.... The requested range must be within the last 60 days.
    ```
  - **News retrieval is strictly capped at ~10 recent articles per ticker**. The Yahoo Finance news endpoint does not provide historical news archive lookback.
- **Rate Limits & Blocking**:
  - Unauthenticated scraping of Yahoo Finance endpoints is subject to IP-based rate limiting (HTTP 429 Too Many Requests) and crumb/cookie invalidation.
  - Per-ticker timeout is set to 8.0s in `app/core/data_fetcher.py`.
- **Timezone Handling**:
  - Returned timestamps from `yfinance` are localized to `Asia/Kolkata` (IST, UTC+05:30) in `DatasetBuilder.fetch_symbol_raw_history()`.

### B. Existing On-Disk Datasets & Caches
| File Path | Description | Records / Scope | Date Range |
| :--- | :--- | :--- | :--- |
| `data/raw/combined_raw.parquet` | Raw concatenated 5m candles across all 48 symbols | 209,716 rows | 2026-06-15 09:15 to 2026-09-04 15:15 |
| `data/raw/{SYMBOL}_raw_5m.parquet` | Per-symbol raw 5m OHLCV files (48 files) | ~4,369 rows each | 2026-06-15 to 2026-09-04 |
| `data/processed/phase5_features.parquet` | Fully processed feature dataset with context & labels | 428,172 candidate rows | 2026-06-15 to 2026-09-04 |
| `data/processed/raw_news_cache.json` | Real financial news cache for 47 symbols | 424 articles | 2025-09-16 to 2026-09-04 |
| `chroma_db/chroma.sqlite3` | Chroma vector database with market & stock daily fingerprints | 59 market / 2,832 stock embeddings | 2026-06-15 to 2026-09-04 |

### C. Environment Variables & Authentication
- `DEFAULT_NSE_TICKERS`: 50 Nifty constituents defined in `app/config.py`.
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Neo4j connection parameters. (Currently unconfigured/None; system falls back to offline deterministic mock data).
- LLM API Keys (`GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`): Used for live agent reasoning, not for historical candle ingestion.

---

## 3. Evaluation of Realistic Historical Data Acquisition Options

To achieve $\ge 2$ years (target: 3–5 years) of 5-minute intraday NSE data, the following options were evaluated:

| Source | Historical Depth (5m) | Cost / Auth Requirements | Reliability & Corporate Actions | Automated Collection Feasibility | Legal & Research Suitability | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Yahoo Finance (`yfinance`)** | **Max 60 calendar days** | Free / No auth | High error rate; 404s on symbol changes; unadjusted/adjusted ambiguity | Native, but cannot exceed 60 days | Public research | **DISQUALIFIED for multi-year** (Hard API constraint) |
| **Zerodha Kite Connect Historical API** | **Up to 7+ years** (via 60-day chunked pagination) | Paid (₹2,000/mo Kite Connect + ₹2,000/mo Historical API add-on); requires Demat account + daily TOTP login | High institutional quality; splits/bonuses adjusted; official NSE broker feed | High via Python `kiteconnect` SDK | Licensed retail algorithmic trading | **TOP PAID CANDIDATE** if credentials available |
| **TrueData / GlobalDataFeeds** | **10+ years** tick/1m/5m continuous | Paid commercial subscription (starts ₹1,800–₹3,500/mo); REST / WebSocket API | Official NSE authorized data vendor; institutional grade corporate action adjustments | High via Python REST API | Commercial / Institutional research | **TOP INSTITUTIONAL VENDOR** |
| **Upstox / Fyers API** | **1 to 3 years** 5m data | Free API for funded broker account holders; App ID + Access Token | Good; NSE equity and F&O; adjusted for splits | High via Python SDK | Brokerage trading & research | **VIABLE BROKER OPTION** |
| **Curated Open Historical Archives (Kaggle / GitHub Open Data)** | **5 to 10 years** (2015–2024) | Free / Open Access (CC-BY 4.0 / Public Domain) | Varies; requires rigorous schema and quality validation via `HistoricalDataValidator` | Immediate batch download (parquet/csv) | Academic and open-source ML research | **TOP FREE RESEARCH CANDIDATE** |

---

## 4. Symbol Universe & Survivorship Bias Analysis

Phase 2–5 utilized a fixed list of **48 Nifty 50 constituents** as of mid-2024.

### Survivorship Bias Risk
Using the **current** Nifty 50 membership to backtest across 2021–2026 creates severe **survivorship bias**:
1. **Winner Selection**: Current members (e.g. `TRENT.NS`, `BEL.NS`, `HAL.NS`) were promoted into the index precisely because they had massive 300%–500% bull runs over the preceding 2–3 years.
2. **Loser Exclusion**: Companies that deteriorated, suffered debt defaults, or were demoted from the Nifty 50 during 2021–2024 (e.g. `UPL`, `SHREECEM`, `GAIL`, `BPCL`, `ZEEL`) would be excluded from historical tests, artificially inflating long-side returns.

### Symbol Universe Methodologies
| Methodology | Description | Feasibility in Trade-Assist | Survivorship Bias Impact |
| :--- | :--- | :---: | :--- |
| **A. Fixed Contemporary Universe** | Fix the current 48 symbols across the entire multi-year lookback. | High (immediate) | **High Bias**: Must be explicitly disclosed as a survivorship-biased study. |
| **B. Point-in-Time Index Membership** | Reconstitute the index on historical rebalancing dates (semi-annually in March and September). | Medium (requires historical NSE Nifty change circulars) | **Zero Bias (Gold Standard)**: Truly reflects the investable universe at decision time $T$. |
| **C. Broad Liquid Universe (Nifty 100 / Nifty 200)** | Expand the universe to the top 100 or 200 liquid stocks to dilute individual stock survivorship effects. | Medium | **Low Bias**: Broad cross-sectional liquidity dampens individual promotion effects. |

**Recommendation for Phase 5.3+**:
Implement **Methodology B (Point-in-Time Membership)** as the canonical architecture, with **Methodology A** allowed only as a documented baseline with explicit survivorship disclaimers.

---

## 5. Corporate Actions & Price Adjustments

Intraday technical indicator calculations are highly sensitive to corporate actions:
1. **Stock Splits & Bonuses**:
   - A 1:1 bonus or 2:1 split causes an unadjusted price to drop by 50% overnight.
   - On an unadjusted chart, this generates a massive artificial gap-down, breaking 20-period VWAP, EMA, RSI, and triggering false stop-losses or false short signals.
2. **Dividends**:
   - Cash dividends of 2%–5% create ex-dividend price drops.
3. **Adjustment Standard**:
   - **Back-Adjustment (Ratio Adjustment)**: Historical prices before the corporate action date are multiplied by the adjustment ratio $R = P_{ex} / P_{cum}$.
   - Volume is inversely divided by $R$ to preserve dollar volume.
   - For intraday research, continuous back-adjusted 5m candles must be stored alongside raw unadjusted prices and adjustment metadata flags.

---

## 6. Contextual History Expansion Feasibility

### A. Real News & FinBERT Sentiment
- **Current Limitation**: `raw_news_cache.json` holds 424 articles, predominantly from 2025–2026.
- **Expansion Bottleneck**: Free Yahoo Finance endpoints do not support archival queries for 2021–2024.
- **Protocol**: If historical news is unavailable for earlier dates, Trade-Assist strictly sets `has_news = False`, `number_of_articles = 0`, and `sentiment_score = NaN`. **Never fabricate historical news**.

### B. ChromaDB Historical Fingerprints
- **Feasibility**: **100% Feasible**.
- Once expanded OHLCV data is acquired, daily whole-market and per-stock fingerprints can be deterministically pre-computed chronologically from historical candles $\le T$.
- Two collections will be populated: `whole_market_daily_fingerprints` and `per_stock_daily_fingerprints`.
- Strictly enforces `trading_date_int < query_date_int`.

### C. Neo4j Knowledge Graph
- **Current State**: Offline / No URI configured in `.env`.
- **Requirements for Activation**: Local or cloud Neo4j instance running with APOC plugin, `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER=neo4j`, `NEO4J_PASSWORD=...`.
- **Temporal Cutoff**: Ingestion enforces `(Stock)-[:ON_DAY]->(TradingDay {date})` with query filtering `WHERE td.date < $query_date`.

---

## 7. Action Plan & Next Steps

1. Establish canonical historical data schema (`docs/PHASE_5_3_HISTORICAL_SCHEMA.md`).
2. Build reusable data validator (`app/ml/historical_data_validator.py`) to catch gaps, bad ticks, and non-IST timestamps.
3. Define strict temporal causality contract (`docs/PHASE_5_3_TEMPORAL_CONTRACT.md`).
4. Design the end-to-end dataset builder (`app/ml/historical_dataset_builder.py`).
5. For multi-year data acquisition, evaluate ingestion from verified historical archives or Kite Connect API.
