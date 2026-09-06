# Phase 5.3 — Strict Temporal Causality Contract

## 1. Objective & Philosophy

In quantitative machine learning for algorithmic trading, the single most lethal threat to model validity is **lookahead bias** (future information leakage). A model trained or evaluated with even a 1-second or 1-bar lookahead leak will exhibit illusory, unreplicable out-of-sample profitability.

This document formalizes the **unbreakable Temporal Contract** governing all current and future data ingestion, feature generation, context retrieval, threshold selection, and model training in Trade-Assist.

---

## 2. The Decision Boundary Rule

For any trade recommendation, model inference, or feature vector evaluated at **Decision Timestamp $T$**:

$$\text{All Information} \le T \quad \text{(or } < T \text{ where specified)}$$

Every data pipeline in Trade-Assist must satisfy the question:
> **"Did this exact bit of data exist, and was it publicly accessible to an intraday market participant, strictly at or before timestamp $T$?"**

If the answer is anything other than an unambiguous **YES**, that data point is strictly forbidden.

---

## 3. Explicit Data Category Contracts

| Data Category | Timestamp Boundary Requirement | Strict Operational Rule | Violation Example (FORBIDDEN) |
| :--- | :--- | :--- | :--- |
| **Intraday OHLCV Candles** | $\le T$ (Closed bars only) | Candle $i$ covering $[T - 5\text{m}, T]$ closes at timestamp $T$. Its Open, High, Low, Close, and Volume are fully finalized and accessible at $T$. | Using the close of candle $[T, T+5\text{m}]$ before it has finished trading. |
| **Technical Indicators** | $\le T$ (Past closed bars) | Moving averages, RSI, MACD, Bollinger Bands, and VWAP must be computed strictly using candles with `timestamp` $\le T$. | Calculating rolling indicators using centered windows or future candles ($T+1$). |
| **Intraday VWAP** | $\le T$ (Current session) | Session cumulative volume-weighted average price resets to zero at 09:15:00 IST every morning and accumulates strictly up to candle $T$. | Incorporating prices from later in the afternoon when evaluating morning candles. |
| **Financial News Headlines** | **$< T$ (Strictly prior publication)** | News article publication timestamp must satisfy `pub_timestamp < T`. If `pub_timestamp == T` or missing, it cannot be used at $T$. | Evaluating a candle at 10:00:00 using a news article published at 10:02:00 or released at 10:00 with ingestion delay. |
| **Chroma Market Fingerprints** | **$<\text{date}(T)$ (Prior days only)** | Daily market fingerprint vectors represent completed daily summaries and are queryable only with `trading_date_int < query_date_int`. | Embedding the current day's complete market return before the market closes at 15:30. |
| **Chroma Stock Fingerprints** | **$<\text{date}(T)$ (Prior days only)** | Daily individual stock fingerprints are queryable only with `trading_date_int < query_date_int`. | Using day $D$'s high-low range to predict trades occurring in the morning of day $D$. |
| **Neo4j Graph Context** | **$< T$ (Historical patterns)** | Graph edges linking patterns to outcomes must represent completed historical trades with exit timestamp $< T$. | Querying a graph edge representing a pattern trade whose exit occurs at $T + 30\text{m}$. |
| **Triple-Barrier Labels** | **Target variable ONLY** | Triple barrier outcomes ($+2.2\%$ target hit, $-0.9\%$ stop hit, or 240m timeout return) occur in interval $(T, T + 240\text{m}]$. They exist strictly as supervised training targets $y$. | Including `exit_reason`, `exit_price`, `realized_return`, or `direction` as a predictive feature in $X$. |
| **Missing Data Imputation** | **Train-set statistics ONLY** | Imputers (e.g. median sentiment for missing news) must be fitted strictly on the chronological Training split, then applied transform-only to Val and Test. | Fitting an imputer on the full dataset, which leaks validation/test distribution medians into training. |
| **Threshold Selection** | **Validation-set ONLY** | Classification probability thresholds $P^*$ must be optimized strictly on the Validation set under the minimum trade guard ($N \ge 30$). | Selecting or adjusting $P^*$ based on test-set precision or test-set return. |

---

## 4. Forbidden Operations Catalog

The following operations are unconditionally barred across the codebase:
1. **No Forward Shifting**: Code like `.shift(-1)` or `.rolling(..., center=True)` is forbidden on feature matrices.
2. **No Data Snooping on Test**: No hyperparameter grid search, threshold tuning, or feature selection may inspect rows where `trading_date >= test_start_date`.
3. **No Synthetic Holiday/Night Candles**: Never fabricate candles for Saturday, Sunday, or overnight hours (15:31 to 09:14 IST).
4. **No Survivorship Fabrication**: Never use current index members to backtest past years without explicitly logging the survivorship bias limitation.
5. **No Synthetic News Insertion**: Never replace missing news with generated or artificial sentiment text. If no news exists prior to $T$, `has_news = False` and `sentiment_score = NaN`.

---

## 5. Automated Verification & Enforcement

Every pull request and build must pass the automated temporal causality tests in `tests/test_phase5_3_data.py`:
- `test_ohlcv_causality()`
- `test_news_timestamp_causality()`
- `test_chroma_date_causality()`
- `test_train_only_preprocessor_isolation()`

Any commit violating this contract will immediately fail automated CI testing.
