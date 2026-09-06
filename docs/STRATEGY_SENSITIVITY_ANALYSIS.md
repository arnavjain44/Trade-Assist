# Phase 2.5 — Strategy & Label Sensitivity Analysis

## Executive Summary

This document presents the complete scientific **Strategy & Label Sensitivity Analysis** conducted for the Trade-Assist ML Pipeline. Following Phase 1 (Data Quality Verification) and Phase 2 (Real Historical Outcome Labeling), Phase 2.5 evaluates how changing trade parameters—specifically **Target Profit Percentage (`TARGET_PCT`)**, **Stop Loss Percentage (`STOP_LOSS_PCT`)**, and **Maximum Holding Horizon (`MAX_HOLD_MINUTES`)**—impacts label distribution, trade outcome geometry, and before-and-after transaction cost economics.

### Key Analysis Highlights
* **Dataset Scope**: Evaluated across **419,432 trade candidates** (209,716 LONG, 209,716 SHORT) generated from **209,716 historical 5-minute candles** across **48 NIFTY 50 equities** over ~60 trading days (April–July 2024).
* **Baseline Configuration (Phase 2)**: `TARGET_PCT` = +2.2%, `STOP_LOSS_PCT` = -0.9%, `MAX_HOLD_MINUTES` = 240m.
  * Yields **4,667 positive outcomes (1.13% positive rate)**, **57,327 stop-out losses (13.85%)**, and **351,772 timeouts (85.02%)**.
* **Configurations Evaluated**: **14 distinct parameter combinations** (Configs A–G target/stop variations, and 60m, 120m, 240m, 375m holding horizons).
* **Dataset Immutability**: The primary Phase 2 labeled dataset (`data/processed/labeled_dataset.parquet`) and quality log (`data/processed/label_quality.json`) were **100% preserved without modification**.

---

## 1. Master Strategy Sensitivity Comparison Table

The table below summarizes all 14 evaluated configurations. Gross metrics are calculated strictly before costs; Net metrics include a realistic **0.05% (5 bps)** round-trip transaction cost (STT, brokerage, exchange fees, and slippage).

| Config ID | Target % | Stop % | Max Hold (m) | Is Baseline | Valid Candidates | Win Rate (%) | Timeout Rate (%) | Gross Avg Return (%) | Gross Profit Factor | Net Avg Return (%) | Net Profit Factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | 1.0% | 0.5% | 240 | False | 413,625 | **10.10%** | 55.10% | +0.0039% | 1.019 | -0.0461% | 0.806 |
| **B** | 1.0% | 0.9% | 240 | False | 413,604 | **10.79%** | 75.55% | +0.0006% | 1.002 | -0.0494% | 0.806 |
| **C** | 1.5% | 0.9% | 240 | False | 413,707 | **3.68%** | 82.51% | +0.0025% | 1.011 | -0.0475% | 0.816 |
| **D** | 2.0% | 0.9% | 240 | False | 413,751 | **1.55%** | 84.60% | +0.0050% | 1.021 | -0.0450% | 0.825 |
| **E (Baseline)** | 2.2% | 0.9% | 240 | **True** | 413,766 | **1.13%** | 85.02% | +0.0057% | 1.025 | -0.0443% | 0.828 |
| **F** | 2.5% | 0.9% | 240 | False | 413,766 | **0.74%** | 85.41% | +0.0063% | 1.027 | -0.0437% | 0.831 |
| **G** | 3.0% | 0.9% | 240 | False | 413,767 | **0.37%** | 85.77% | +0.0069% | 1.030 | -0.0431% | 0.833 |
| **E_60m** | 2.2% | 0.9% | 60 | False | 413,766 | **0.17%** | 96.27% | +0.0014% | 1.010 | -0.0486% | 0.711 |
| **E_120m** | 2.2% | 0.9% | 120 | False | 413,766 | **0.48%** | 91.53% | +0.0028% | 1.015 | -0.0472% | 0.780 |
| **E_375m** | 2.2% | 0.9% | 375 | False | 413,766 | **1.58%** | 82.55% | +0.0052% | 1.021 | -0.0448% | 0.835 |
| **A_60m** | 1.0% | 0.5% | 60 | False | 413,688 | **2.44%** | 81.55% | +0.0022% | 1.016 | -0.0478% | 0.709 |
| **C_60m** | 1.5% | 0.9% | 60 | False | 413,746 | **0.63%** | 95.82% | +0.0008% | 1.006 | -0.0492% | 0.708 |
| **A_120m** | 1.0% | 0.5% | 120 | False | 413,650 | **5.68%** | 67.84% | +0.0032% | 1.018 | -0.0468% | 0.771 |
| **C_120m** | 1.5% | 0.9% | 120 | False | 413,731 | **1.64%** | 90.38% | +0.0013% | 1.007 | -0.0487% | 0.772 |

---

## 2. Quantitative Sensitivity Analysis

### A. Target & Stop Loss Sensitivity
1. **Target Scaling Effect**: Keeping `STOP_LOSS_PCT` constant at 0.9% (holding at 240m), increasing `TARGET_PCT` from 1.0% (Config B) to 3.0% (Config G) decreases the positive rate non-linearly:
   * **1.0% Target (B)**: 10.79% positive rate (44,609 target hits)
   * **1.5% Target (C)**: 3.68% positive rate (15,223 target hits)
   * **2.0% Target (D)**: 1.55% positive rate (6,397 target hits)
   * **2.2% Target (E)**: 1.13% positive rate (4,667 target hits)
   * **3.0% Target (G)**: 0.37% positive rate (1,544 target hits)
2. **Stop Loss Tightening Effect**: Comparing Config A (1.0% target / 0.5% stop) against Config B (1.0% target / 0.9% stop):
   * Tightening the stop loss to 0.5% drastically increases stop-outs from **13.67% (56,519)** to **34.80% (143,948)**.
   * However, it reduces timeouts from **75.55%** to **55.10%**.

### B. Holding Horizon Sensitivity
Holding horizon controls how long an un-triggered trade remains active before being force-closed at market close / session exit:
* **60-Minute Horizon (E_60m)**: Only **0.17%** hit the +2.2% target; **96.27%** end in TIMEOUT.
* **120-Minute Horizon (E_120m)**: **0.48%** hit the target; **91.53%** end in TIMEOUT.
* **240-Minute Baseline (E)**: **1.13%** hit target; **85.02%** end in TIMEOUT.
* **375-Minute Full Session (E_375m)**: **1.58%** hit target; **82.55%** end in TIMEOUT.

> **Takeaway**: Intraday price movements take time to develop. A 60-minute window is mathematically too short for a 2.2% move in NIFTY 50 large-cap equities.

### C. LONG vs SHORT Directional Asymmetry Analysis

Across all configurations, LONG trades exhibited a slightly higher positive outcome rate than SHORT trades during the sample period (April–July 2024), reflecting a mild underlying bullish market regime in NIFTY 50:

| Config | LONG Positive Rate (%) | SHORT Positive Rate (%) | LONG Avg Return (%) | SHORT Avg Return (%) |
| :--- | :---: | :---: | :---: | :---: |
| **A (1.0%/0.5%)** | 10.41% | 9.79% | +0.0079% | -0.0000% |
| **B (1.0%/0.9%)** | 11.09% | 10.48% | +0.0088% | -0.0077% |
| **C (1.5%/0.9%)** | 4.16% | 3.19% | +0.0132% | -0.0081% |
| **E (Baseline 2.2%/0.9%)** | **1.34%** | **0.91%** | **+0.0173%** | **-0.0059%** |

### D. Economic & Transaction Cost Impact

Without transaction costs, all baseline and candidate configurations demonstrate a gross Profit Factor slightly above 1.0 (range 1.002 to 1.030).

However, introducing a realistic **0.05% (5 bps)** transaction cost:
* **Gross Avg Return (Baseline E)**: +0.0057% (+0.57 bps) per candidate trade.
* **Net Avg Return (Baseline E)**: **-0.0443% (-4.43 bps)** per candidate trade.
* **Net Profit Factor (Baseline E)**: **0.828** (unprofitable under random/unfiltered candidate execution).

> **Crucial Insight**: Indiscriminate, unfiltered execution across every 5-minute candle leads to negative net expected value due to friction. The machine learning model in Phase 3 must learn to **filter out the 98.87% low-probability signals** and select only high-conviction trades where expected gross return easily exceeds transaction costs.

---

## 3. Formal Decision Analysis (Questions A–E)

### Question A: Is the Phase 2 Baseline 1.13% Positive Rate Realistic or Broken?
**Finding**: The 1.13% positive rate is **mathematically sound, realistic, and uncorrupted**.
* **Reasoning**: NIFTY 50 large-cap stocks have a typical single-day intraday volatility of 1.2% to 1.8%. Expecting a +2.2% price move within a 4-hour window from an arbitrary 5-minute candle without signal filtering is a high-threshold event.
* **Conclusion**: The labeling algorithm correctly captures real market physics. The low baseline positive rate reflects the stringent target, not a pipeline flaw.

### Question B: Which Configurations Are Promising vs. Poor?
1. **Promising Configurations**:
   * **Config B (1.0% Target / 0.9% Stop / 240m Hold)**: Yields a **10.79% positive rate** (44,609 positive samples). Provides a balanced class distribution for ML training while maintaining positive gross expected return.
   * **Config C (1.5% Target / 0.9% Stop / 240m Hold)**: Yields a **3.68% positive rate** (15,223 positive samples). Offers higher target capture with ~3.2x more positive samples than baseline.
2. **Poor Configurations**:
   * **Config A (1.0% Target / 0.5% Stop)**: Excessive stop-out rate (34.80%), high sensitivity to noise/bid-ask spread.
   * **Configs E_60m / C_60m (60-minute holds)**: Over 95% timeout rate; insufficient time for intraday price discovery.
   * **Config G (3.0% Target)**: Extreme class imbalance (0.37% positive rate, only 1,544 instances in 419k candidates).

### Question C: What Is the Root Cause of the ~85% Timeout Rate?
The root cause is **market micro-structure & volatility scale**:
1. 5-minute candles in Indian equities frequently experience micro-range consolidation.
2. At a +2.2% target and -0.9% stop, 85% of intraday trades neither surge 2.2% nor collapse 0.9% within 4 hours.
3. Crucially, **timeouts are NOT lost money**: the average realized return for baseline timeouts (`avg_timeout_return_pct`) is **+0.1242% (+12.42 bps)**.

### Question D: Mathematical & Empirical Evidence for Phase 3 ML
* The labeled dataset provides **clean causal targets** with **zero lookahead leakage** (entry candle close excluded, outcome evaluated strictly from next candle).
* Target hit returns yield **exact target percentage (+2.2%)**, stop hits yield **exact stop percentage (-0.9%)**, and timeouts preserve **exact realized return at exit**.
* Class imbalance (1.13% to 10.79%) is well within the effective range for XGBoost / LightGBM weighted binary classification loss functions (e.g. `scale_pos_weight`).

### Question E: Strategic Recommendation for Phase 3 Model Setup
* **Primary Recommendation**: Retain **Baseline Config E (2.2% target / 0.9% stop)** as the primary high-conviction target, OR adopt a **multi-label setup** (e.g. predicting Config B +1.0% move as an intermediate trigger, and Config E +2.2% move as the primary trade filter).
* **Classification Loss Guidance**: Use `scale_pos_weight` (~75:1 for Config E, ~8:1 for Config B) or focal loss during LightGBM / XGBoost training.

---

## 4. Verification & Immutability Attestation

1. **Dataset Integrity**: `data/processed/labeled_dataset.parquet` checksum and row count (419,432 candidates) match Phase 2 output exactly.
2. **Test Suite Status**: All 45 unit tests across `tests/test_labeling.py` and `tests/test_strategy_sensitivity.py` **PASSED (100% success rate)**.
3. **Artifact Exports**:
   * `data/processed/strategy_sensitivity_results.json` (Full structural JSON export)
   * `data/processed/strategy_sensitivity_results.csv` (CSV summary table)

---
*Report generated automatically by Trade-Assist ML Pipeline (Phase 2.5).*
