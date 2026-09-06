# Phase 5.4 — Historical Data Source Evaluation & Selection

## 1. Objective
To evaluate and select an authorized, reliable, and technically viable data source capable of delivering $\ge 2$ years (target: 3–7 years) of 5-minute intraday OHLCV data for Indian equities (NSE / Nifty 50).

---

## 2. Source Category Evaluation

Following the required hierarchical evaluation order:

### A. Authorized Broker / Vendor API Already Available in Project
- **Audit Finding**: None currently configured or integrated.
- **Repository State**: Only `yfinance` is referenced in `requirements.txt` and codebase. No broker API keys, Demat account tokens, or vendor clients exist in the repo.

---

### B. Authorized Historical Data Providers ($\ge 2$ Years NSE 5-minute OHLCV)

#### 1. Zerodha Kite Connect Historical Data API (Recommended Broker API)
- **Documentation**: [https://kite.trade/docs/connect/v3/historical/](https://kite.trade/docs/connect/v3/historical/)
- **Historical Depth**: Up to 7+ years of continuous 5-minute data (from 2017/2018 onwards).
- **5-Minute Availability**: Fully supported (`5minute` interval).
- **NSE Coverage**: 100% of NSE cash equities and indices.
- **Symbol Coverage**: All Nifty 50 equities mapped via official `instruments` master token list.
- **Rate Limits**: 3 requests per second for historical data endpoint (`/instruments/historical/...`).
- **Pagination / Chunking Limits**: **Max 100 days per request** for `5minute` interval (enforced server-side).
- **Authentication**: `api_key` + daily interactive login exchanging request token for `access_token`.
- **Approximate Cost**: ₹2,000/month (Kite Connect API) + ₹2,000/month (Historical API add-on) = ₹4,000/month.
- **Corporate Action Handling**: Cash market equity data is split- and bonus-adjusted.
- **Timestamp Semantics**: ISO 8601 in Indian Standard Time (`UTC+05:30`), e.g., `2024-01-01T09:15:00+0530`. Represents **candle start time**.
- **Licensing & Research Use**: Fully compliant with retail algorithmic trading and quantitative research under Zerodha developer terms.
- **Local Storage**: Data can be cached and stored locally in Parquet/CSV format for backtesting and ML.
- **Suitability**: **TOP CANDIDATE** for live acquisition once credentials are provided.

#### 2. TrueData / GlobalDataFeeds (Authorized Commercial Vendor)
- **Documentation**: [https://www.truedata.in/](https://www.truedata.in/)
- **Historical Depth**: 10+ years of tick/1-minute/5-minute continuous NSE data.
- **5-Minute Availability**: Fully supported via REST API.
- **NSE Coverage**: Official NSE-authorized data vendor.
- **Symbol Coverage**: All NSE equities, indices, and derivatives.
- **Rate Limits**: Configurable per commercial tier.
- **Authentication**: API Key and Username/Password bearer token.
- **Approximate Cost**: ₹1,800 to ₹3,500/month.
- **Corporate Action Handling**: Institutional-grade adjustment matrices; splits and bonuses accounted for.
- **Timestamp Semantics**: IST.
- **Suitability**: Highly suitable for institutional-scale multi-decade research.

---

### C. Local & Open Research Archives
- **Source**: Curated historical research archives (e.g., historical Nifty 50 5m CSV/Parquet archives from academic or open quantitative repositories covering 2017–2024).
- **Historical Depth**: 5 to 7 years.
- **Cost**: Free (Open Data / CC-BY).
- **Authentication**: None.
- **Corporate Action Handling**: Varies by archive source. Must be audited with `HistoricalDataValidator`.
- **Suitability**: **TOP FREE RESEARCH OPTION**. The system must include a `LocalCsvAdapter` to ingest, validate, and normalize such archives immediately without incurring ongoing API subscription fees.

---

### D. Yahoo Finance (`yfinance`) — DISQUALIFIED
- **Historical Depth**: Max 60 calendar days for 5-minute interval.
- **Verification**: Re-confirmed server rejection `5m data not available for startTime=... The requested range must be within the last 60 days.`
- **Suitability**: **DISQUALIFIED for multi-year research**. Retained solely as a fallback for short-term (<60 days) demonstration or live daily feeds.

---

## 3. Decision & Architecture Strategy

1. **Primary Commercial Target**: **Zerodha Kite Connect Historical API**. We build a production-grade, credential-safe adapter with 100-day chunked pagination, rate limiting (3 req/s), and candle start-to-end normalization.
2. **Immediate Research Target**: **Local CSV / Archive Adapter (`LocalCsvAdapter`)**. Enables immediate, free loading of multi-year historical files (e.g., downloaded Kaggle/GitHub 5m datasets) through the exact same canonical pipeline.
3. **Credential Handling**: System operates in a safe dry-run/mock mode when API keys are unconfigured, failing clearly without inventing credentials or fabricating data.
