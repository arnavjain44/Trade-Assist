# Phase 5.4 — Corporate Action & Price Adjustment Policy

## 1. Executive Summary
In quantitative intraday trading research on equities, corporate actions (stock splits, bonus issues, rights issues, special cash dividends, and demergers) introduce artificial price discontinuities if left unadjusted. 

For example, a 1:1 bonus issue cuts the market price in half overnight (e.g. from ₹3,000 to ₹1,500). On an unadjusted chart, this generates an artificial -50% price drop that catastrophically corrupts:
- Moving averages, session VWAP, and Bollinger Bands
- Momentum indicators (RSI drops to near 0, MACD diverges falsely)
- Triple-barrier labeling (triggers immediate artificial stop-loss hit on long trades)
- Volatility metrics (ATR spikes artificially by an order of magnitude)

This document establishes Trade-Assist's corporate action policy for multi-year historical datasets.

---

## 2. Adjustment Regimes Defined

1. **Raw (Unadjusted)**:
   - Contains actual traded prices at the historical timestamp $T$.
   - **Use Case**: Execution sanity checks, slippage modeling, and order book matching.
   - **Limitation**: Unusable across multi-year indicator calculations without explicit event adjustments.

2. **Split- & Bonus-Adjusted (Capital Action Adjusted)**:
   - Historical prices prior to the ex-date are adjusted by the split/bonus ratio:
     $$P_{\text{adjusted}}(t) = P_{\text{raw}}(t) \times \frac{P_{\text{ex}}}{P_{\text{cum}}}$$
   - Volume is inversely scaled to preserve dollar turnover:
     $$V_{\text{adjusted}}(t) = V_{\text{raw}}(t) \times \frac{P_{\text{cum}}}{P_{\text{ex}}}$$
   - **Use Case**: **Standard quantitative standard for intraday technical and ML research**. Preserves continuous price charts and geometric returns while avoiding artificial indicator distortion.

3. **Total-Return Adjusted (Dividend Adjusted)**:
   - Additionally adjusts historical prices downwards for ordinary cash dividends.
   - **Limitation for Intraday Trading**: Cash dividends represent true economic overnight price drops (stocks open ex-dividend lower). In intraday trading, dividend adjustment creates artificial fractional prices and historical bias over multi-year spans.

---

## 3. Data Source Specifics

| Provider | Stock Splits | Bonus Issues | Cash Dividends | Demergers / Mergers | Adjustment Standard |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Zerodha Kite Connect Historical API** | **Adjusted** | **Adjusted** | **Unadjusted** | Handled per exchange circular | **Split- & Bonus-Adjusted** (Ideal for intraday) |
| **TrueData** | **Adjusted** | **Adjusted** | Separate dividend series | Split-adjusted continuous | **Split- & Bonus-Adjusted** |
| **Local Research Archives** | Varies | Varies | Varies | Archive-dependent | Must be validated via anomaly validator |
| **Yahoo Finance (`yfinance`)** | Mixed | Mixed | Auto-adjust optional | Often creates historical distortion | Inconsistent across symbols |

---

## 4. Trade-Assist Canonical Policy

1. **ML Dataset Standard**:
   - The multi-year intraday feature engineering and triple-barrier labeling pipelines **strictly require split- and bonus-adjusted OHLCV data**.
   - Dividends remain unadjusted in price series (cash dividends are accounted for in overnight holding, not intraday 5m price structures).

2. **Metadata Tracking**:
   - Every historical dataset file must record its adjustment regime in its companion metadata JSON:
     ```json
     {
       "adjustment_regime": "split_and_bonus_adjusted",
       "dividend_adjusted": false,
       "provider": "kite",
       "verified_by_validator": true
     }
     ```

3. **Incompatible Regime Rejection**:
   - Never concatenate raw unadjusted data with split-adjusted data across a split event date.
   - `HistoricalDataValidator` inspects overnight price ratios between consecutive sessions: if $| \ln(P_{\text{open}}(t) / P_{\text{close}}(t-1)) | > 0.35$ without an earnings/macro shock, the bar is flagged for corporate action review.
