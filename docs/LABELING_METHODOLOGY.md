# Historical Trade Outcome Labeling Methodology (Phase 2)

## 1. Overview & Disclaimer
This document specifies the methodology, trade construction rules, outcome decision logic, and data quality validation protocols implemented in **Phase 2** of the `Trade-Assist` system.

> [!IMPORTANT]
> **The label represents a historical hypothetical trading outcome. It is not a prediction.**
> The eventual ML model will learn patterns mapping historical feature representations available at entry timestamp $T_e$ to subsequent historical trade outcomes.

---

## 2. Strategy Parameters
Strategy parameters define the hypothetical trade geometry. They are explicit configurable constants in `app/config.py` (`Settings`):

- **`LABEL_TARGET_PCT`**: `0.022` (+2.2% target)
- **`LABEL_STOP_LOSS_PCT`**: `0.009` (-0.9% stop-loss)
- **`LABEL_MAX_HOLD_MINUTES`**: `240` (240 minutes / 4 hours maximum intraday holding horizon)

---

## 3. Trade Candidate Generation & Entry Definition

For every eligible historical 5-minute candle at entry timestamp $T_e$ with close price $P_{close}$:

Two independent hypothetical trade candidates are generated:

### 3.1 LONG Candidate (`direction = 1`)
- **`entry_price`**: $P_{close}$
- **`target_price`**: $P_{close} \times (1 + 0.022)$
- **`stop_price`**: $P_{close} \times (1 - 0.009)$

### 3.2 SHORT Candidate (`direction = -1`)
- **`entry_price`**: $P_{close}$
- **`target_price`**: $P_{close} \times (1 - 0.022)$
- **`stop_price`**: $P_{close} \times (1 + 0.009)$

---

## 4. Outcome Determination Rules

Future candles occurring after $T_e$ are evaluated chronologically.

### 4.1 Entry Candle Exclusion
- The entry candle $T_e$ **itself is strictly EXCLUDED** from outcome determination.
- Only candles with timestamp $T_{future} > T_e$ determine the trade outcome.

### 4.2 Same-Day Intraday Restriction
- Only future candles belonging to the **SAME trading session/date in Asia/Kolkata (IST)** ($T_{future}.\text{date} == T_e.\text{date}$) are evaluated.
- Future candles on subsequent calendar or trading days ($T_{future}.\text{date} > T_e.\text{date}$) are **strictly forbidden**.

### 4.3 Maximum Holding Horizon
- Only future candles within $T_{future} \le T_e + \text{timedelta(minutes=240)}$ are evaluated.

---

## 5. Label Assignment & Outcome Categories

| Outcome Category | Conditions | `label` | `label_status` | `exit_reason` | Excluded from ML Training? |
|---|---|---|---|---|---|
| **Target Reached First** | Target condition satisfied before stop condition | `1` | `"VALID"` | `"TARGET"` | No (Included) |
| **Stop Reached First** | Stop condition satisfied before target condition | `0` | `"VALID"` | `"STOP"` | No (Included) |
| **Timeout (Horizon Reached)** | Neither target nor stop hit within 240m horizon | `0` | `"VALID"` | `"TIMEOUT"` | No (Included; tracked separately) |
| **Same-Candle Ambiguity** | Both target and stop satisfied on the same future 5m bar | `None` / `0` | `"AMBIGUOUS"` | `"AMBIGUOUS"` | **YES (Excluded)** |
| **Insufficient Future Data** | 0 future candles available in same session (e.g. 15:15 IST entry) | `None` / `0` | `"INSUFFICIENT_FUTURE_DATA"` | `"INSUFFICIENT_FUTURE_DATA"` | **YES (Excluded)** |

---

## 6. Same-Candle Ambiguity Specification
A 5-minute OHLC bar provides `open`, `high`, `low`, and `close`, but does NOT reveal intrabar price sequence.
If a future candle satisfies both $\text{High} \ge \text{target\_price}$ AND $\text{Low} \le \text{stop\_price}$ (for LONG), or $\text{Low} \le \text{target\_price}$ AND $\text{High} \ge \text{stop\_price}$ (for SHORT):
- No arbitrary assumption is made about whether high or low occurred first.
- The candidate is marked `label_status = "AMBIGUOUS"`.
- Ambiguous samples are excluded from supervised training datasets to prevent label noise.

---

## 7. Feature vs Outcome Column Isolation
To prevent data leakage during machine learning model training, the dataset strictly isolates feature input columns from outcome label columns:

### 7.1 Feature Matrix Columns (Inputs to ML)
`open`, `high`, `low`, `close`, `volume`, `ema_5`, `rsi`, `obv`, `bollinger_middle`, `bollinger_upper`, `bollinger_lower`, `macd`, `macd_signal`, `macd_diff`, `vwap`, `bollinger_position`, `price_vs_vwap`, `price_vs_ema5`, `sentiment_score`, `market_similarity`, `stock_similarity`.

### 7.2 Outcome Columns (Targets / Metadata only)
`entry_price`, `target_price`, `stop_price`, `direction`, `label`, `label_status`, `exit_timestamp`, `exit_price`, `exit_reason`, `holding_period_minutes`, `realized_return`.

> [!CAUTION]
> Outcome columns MUST NOT be included in feature matrices during model training or inference.

---

## 8. Sample Overlap & Purging Metadata
Generating candidates for every 5-minute bar produces overlapping trade horizons.
- The dataset is **not randomly shuffled**; rows are stored in chronological order (`symbol`, `timestamp`, `direction`).
- Complete metadata (`exit_timestamp`, `holding_period_minutes`) is preserved for every sample, enabling purged and embargoed cross-validation splits in future model training phases.
