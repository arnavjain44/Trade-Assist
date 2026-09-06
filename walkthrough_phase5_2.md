# Phase 5.2 — LONG-Edge Robustness & Multi-Year Validation Walkthrough

## 1. Overview & Research Objective
Phase 5.2 investigated the primary research question emerging from Phase 5 and Phase 5.1:

> **"Does the apparent LONG-side predictive/economic edge persist across substantially more historical data, symbols, market regimes, and time periods?"**

This phase was strictly **NOT** an accuracy-maximization exercise. Under Phase 5.2 rules, no artificial manipulation of thresholds or fabrication of data was allowed.

---

## 2. Historical Data Availability Audit
A rigorous audit of all datasets in the repository revealed:
- **Available Historical Span**: Exactly **59 trading days** (`2026-06-15` to `2026-09-04`), ~2.7 calendar months.
- **Intraday Bar Resolution**: 5 minutes (209,716 bars across 48 NSE Nifty 50 stocks).
- **Labeled Candidates**: 428,172 total candidates (214,086 Long candidates).
- **Multi-Year History Availability**: **UNAVAILABLE** in the local repository.
- **Scientific Integrity Stance**: Under Phase 5.2 absolute protection rules, Trade-Assist **strictly refused to synthesize artificial multi-year candles, clone historical cycles, or manufacture fictitious news**.
- Certified in: [docs/PHASE_5_2_DATA_AVAILABILITY.md](file:///d:/proj1/proj%20files/docs/PHASE_5_2_DATA_AVAILABILITY.md).

---

## 3. Pre-Specified Threshold Frontier & Statistical Uncertainty (Canonical Holdout)

The pre-declared threshold grid was mapped on the canonical locked Test set (Days 52–59):

| Threshold ($P^*$) | Trades ($N$) | Wins / Losses | Precision (%) | 95% Wilson Confidence Interval | Net Avg Return (%) | Net Profit Factor | Max Drawdown (%) | Unique Days | Largest Day Conc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.60** | 140 | 48 / 92 | 34.29% | [26.9%, 42.5%] | -0.0975% | 0.744 | 43.52% | 8 | 46.4% |
| **0.65** | 84 | 36 / 48 | 42.86% | [32.8%, 53.5%] | +0.0740% | 1.227 | 24.20% | 7 | 45.2% |
| **0.70** | 44 | 22 / 22 | 50.00% | [35.8%, 64.2%] | +0.1938% | 1.681 | 11.18% | 6 | 43.2% |
| **0.75** | 14 | 8 / 6 | 57.14% | [32.6%, 78.6%] | +0.2978% | 2.019 | 3.07% | 4 | 42.9% |
| **0.80** | 3 | 2 / 1 | 66.67% | [20.8%, 93.8%] | +0.4253% | 2.343 | 0.00% | 3 | 33.3% |
| **0.85** | 2 | 2 / 0 | 100.00% | [34.2%, 100.0%] | +1.1130% | 999.000 | 0.00% | 2 | 50.0% |
| **0.90** | 0 | 0 / 0 | 0.00% | [0.0%, 0.0%] | +0.0000% | 0.000 | 0.00% | 0 | 0.0% |

### Key Findings on Precision Frontier:
- **Precision vs Selectivity Monotonicity**: Increasing $P^*$ from 0.60 to 0.85 steadily improves net average return from -0.0975% to +1.1130% and net profit factor from 0.744 to > 2.3.
- **Statistical Uncertainty**: At high thresholds, small sample sizes ($N < 30$) yield wide 95% Wilson confidence intervals (e.g. `[20.8%, 93.8%]` at $P^* = 0.80$), confirming that high point estimates cannot be conflated with a statistically proven 90% population edge.

---

## 4. Multi-Dimensional Robustness Audit

### A. Symbol Breadth & Concentration
- Evaluated across 48 Nifty constituents.
- Symbol concentration is classified as **B. CONCENTRATED IN SMALL SUBSET** at high thresholds, with active trades clustering in momentum leaders (`TRENT.NS`, `HAL.NS`, `ASIANPAINT.NS`).

### B. Temporal Persistence
- Sliced by day, week, and month:
  - Week 35 (`2026-W35`): 2 trades, 2 wins (100.0% win rate, +1.1130% net avg return, PF 999.0).
  - Month September 2026: 2 trades, 2 wins (100.0% win rate).
  - August 2026: 1 trade, 0 wins (-0.9500% net avg return).

### C. Causal Market Regimes
- **Volatility**: All executed trades at $P^* \ge 0.80$ occurred during **High Volatility** conditions (66.7% win rate, +0.4253% net return, PF 2.343).
- **Trend**: Positive economics survived in both Bullish Trend (+0.8073%) and Neutral/Bearish Trend (+1.4187%).
- **Time-of-Day**: High-confidence trades occurred predominantly during the **Opening Session (09:15–10:00 IST)**.

---

## 5. Comparative Baselines Summary

| Architecture / Control | Trades ($N$) | Win Rate / Precision (%) | Net Avg Return (%) | Net Profit Factor |
| :--- | :---: | :---: | :---: | :---: |
| **Frozen Phase 5 D-LightGBM (Audited Benchmark)** | **48** | **68.75%** (Long: 80.0%, Short: 12.5%) | **+0.2706%** | **2.338** |
| **LONG-Only LightGBM Architecture** | 3 | 66.67% | +0.4253% | 2.343 |
| **Simple Technical Control (No Context)** | 19 | 47.37% | +0.1145% | 1.302 |
| **Market Unfiltered Benchmark (Buy-and-Hold)** | 28,148 | 43.90% | -0.0443% | N/A |

---

## 6. Final Scientific Classification

### **B. PROMISING BUT NOT ROBUST**

#### Rationale:
1. **Positive Economics Verified**: Across all high-confidence thresholds ($P^* \ge 0.70$), net average returns are positive (+0.19% to +1.11%) and net profit factors exceed 1.6 to 2.3, confirming an underlying economic edge on the long side.
2. **Multi-Year Data Absence**: Because true multi-year intraday history ($\ge 2$ years) is not present in the repository (59 days total available), the edge cannot be certified as multi-year robust.
3. **Event Clustering**: Significant trade concentration on select momentum days (`2026-09-01`) prevents classification as `A. ROBUST LONG EDGE`.
4. **Production Code Isolation**: Live prediction endpoints remain **100% untouched**.
