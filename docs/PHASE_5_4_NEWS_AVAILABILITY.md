# Phase 5.4 — Historical News & Sentiment Availability Audit

## 1. Executive Summary
In Phase 4 and Phase 5, Trade-Assist integrated FinBERT-derived news sentiment into the feature set using 424 real news articles fetched from Yahoo Finance. However, this coverage spanned only the recent ~59 trading days (late 2025 to mid-2026).

Expanding the intraday dataset to $\ge 2$ years raises a critical question:
> **Can real historical financial news be acquired for all Nifty 50 equities with precise, tamper-proof publication timestamps back to 2018–2023?**

This document establishes the empirical reality of historical news availability, explains why synthetic or backfilled news is strictly forbidden, and confirms the protocol for the multi-year benchmark.

---

## 2. Audit of Data Sources for Historical Financial News

### A. Zerodha Kite Connect API
- **Historical News Support**: **NONE**.
- Kite Connect is strictly an exchange market data and order routing broker API. It does not ingest, index, or redistribute financial news or press releases.

### B. TrueData / GlobalDataFeeds
- **Historical News Support**: **NONE**.
- Authorized NSE vendors provide tick, minute, and daily price/volume and order book data, not unstructured textual news archives.

### C. Yahoo Finance (`yfinance`)
- **Historical News Support**: **Capped at ~10 recent articles per ticker**.
- Yahoo Finance API does not maintain a searchable archival news endpoint. Attempting to query news prior to the most recent week/month returns zero results.

### D. Commercial Financial News Archives (Bloomberg / Refinitiv / Dow Jones)
- **Coverage**: Comprehensive multi-decade archive with millisecond timestamps.
- **Cost**: Institutional grade ($20,000–$50,000+/year).
- **Availability**: Unavailable to this project.

### E. Public / Academic Financial News Datasets (Kaggle / Open Datasets)
- **Examples**: Kaggle "Daily Financial News for 6000+ Stocks" or Reuters Indian market scrapers.
- **Limitations**:
  - Often lack intraday publication timestamps (many only record publication date, not hour:minute, making causal alignment impossible without lookahead bias).
  - Sparse coverage for Indian NSE equities compared to US NASDAQ/NYSE equities.

---

## 3. Strict Integrity Policy: Zero Fabrication

To maintain absolute scientific validity:
1. **NO Synthetic News**: We will NEVER use LLMs or scripts to generate synthetic headlines or sentiments for past dates.
2. **NO Backfilled or Reused News**: We will NEVER replicate 2025/2026 headlines into 2022/2023.
3. **NO Timestamp Tampering**: If an article does not contain a verified publication timestamp, it is disqualified.
4. **Causal Alignment Rule**: Any news incorporated into candle at timestamp $T$ must satisfy:
   $$t_{\text{published}} < t_{\text{candle\_start}} < t_{\text{candle\_end}}$$
5. **Absence Handling**: When news is unavailable, the pipeline records:
   - `has_news = False`
   - `news_count = 0`
   - `news_sentiment = NaN` (or imputed neutral in feature engineering, with an explicit missingness indicator).

---

## 4. Multi-Year Benchmark Decision

Since verified multi-year historical news archives are currently unavailable without institutional subscriptions:
> **The multi-year historical dataset for Phase 5.4+ will establish a rigorous Technical-Only Multi-Year Benchmark first.**

The architecture (`app/ml/historical_dataset_builder.py` and `app/ml/historical_data/`) supports causal news joins if and when a verified multi-year news dataset is procured, but model evaluation on multi-year data will not be blocked by the absence of historical news.
