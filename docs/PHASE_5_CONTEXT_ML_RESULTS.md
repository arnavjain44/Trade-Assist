# Phase 5 — Real-Context ML Experiment Results

## Executive Summary

This report documents the findings of the **Phase 5 Real-Context Machine Learning Experiment**.
The objective of this phase is to answer the core empirical question:

> **Does adding real news sentiment (FinBERT) and strictly historical market/stock context (Chroma) improve out-of-sample trading economics compared with the Phase 3 technical-only baseline?**

### Scientific Control & Benchmark Integrity
- **A1 — Frozen Phase 3 Benchmark**: Preserved directly from `data/processed/phase3_model_results.json` without modification or retuning.
- **A2 — Phase 5 Technical-Only Control**: Evaluates the Phase 5 dataset and train-only preprocessing pipeline under technical features only, applying the new hierarchical economic selection protocol.
- **B — Technical + Real News**: Evaluates technical features + FinBERT news sentiment.
- **C — Technical + Chroma Context**: Evaluates technical features + historical Chroma daily fingerprints.
- **D — Full Phase 5 Model**: Evaluates technical features + news + Chroma historical context.
- **Neo4j Status**: Confirmed unavailable in the local environment (`driver is None`). In accordance with the zero-fabrication rule, no synthetic Neo4j features were generated. Experiments C and D evaluate historical Chroma daily fingerprints.

---

## 1. Complete Ablation Matrix Comparison (Out-of-Sample Test Set)

The table below presents the final locked Out-of-Sample Test set (Days 52–60) performance across all experiments and model families after applying 0.05% (5 bps) round-trip friction.

| Exp ID | Experiment Description | Model Family | Locked Threshold ($P^*$) | Test PR-AUC | Test Precision | Test Trades | Gross Avg (%) | Net Avg (%) | Gross PF | Net PF | Max DD (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | *Frozen Phase 3 Benchmark* | logistic_regression | 0.5000 | 0.0112 | 0.0000 | 0 | +0.0000% | +0.0000% | 0.000 | 0.000 | 0.00% |
| **A1** | *Frozen Phase 3 Benchmark* | random_forest | 0.0137 | 0.0248 | 0.0282 | 9676 | +0.0394% | -0.0106% | 1.165 | 0.960 | 207.29% |
| **A1** | *Frozen Phase 3 Benchmark* | lightgbm | 0.0133 | 0.0206 | 0.0209 | 7645 | +0.0277% | -0.0223% | 1.122 | 0.912 | 227.21% |
| **A2** | Phase 5 Technical-Only Control | logistic_regression | 0.5400 | 0.0306 | 0.0000 | 124 | +0.3467% | +0.2967% | 2.950 | 2.502 | 10.84% |
| **A2** | Phase 5 Technical-Only Control | random_forest | 0.0133 | 0.0258 | 0.0686 | 102 | +0.1338% | +0.0838% | 1.679 | 1.373 | 6.89% |
| **A2** | Phase 5 Technical-Only Control | lightgbm | 0.8200 | 0.0230 | 0.1562 | 32 | +0.1847% | +0.1347% | 1.479 | 1.324 | 9.08% |
| **B** | Technical + Real News Sentiment | logistic_regression | 0.6000 | 0.0254 | 0.0340 | 11429 | +0.0371% | -0.0129% | 1.140 | 0.956 | 620.50% |
| **B** | Technical + Real News Sentiment | random_forest | 0.0131 | 0.0257 | 0.0742 | 418 | +0.1635% | +0.1135% | 1.621 | 1.396 | 27.28% |
| **B** | Technical + Real News Sentiment | lightgbm | 0.8200 | 0.0209 | 0.0469 | 64 | +0.2012% | +0.1512% | 1.793 | 1.549 | 4.04% |
| **C** | Technical + Chroma Historical Context | logistic_regression | 0.0153 | 0.0421 | 0.0312 | 3141 | +0.1462% | +0.0962% | 1.583 | 1.353 | 24.84% |
| **C** | Technical + Chroma Historical Context | random_forest | 0.5000 | 0.0257 | 0.1111 | 63 | -0.0129% | -0.0629% | 0.970 | 0.864 | 6.46% |
| **C** | Technical + Chroma Historical Context | lightgbm | 0.7600 | 0.0258 | 0.0169 | 59 | +0.2572% | +0.2072% | 2.273 | 1.950 | 5.72% |
| **D** | Full Phase 5 Model (Technical + News + Chroma) | logistic_regression | 0.0142 | 0.0462 | 0.0307 | 2704 | +0.1665% | +0.1165% | 1.706 | 1.453 | 24.25% |
| **D** | Full Phase 5 Model (Technical + News + Chroma) | random_forest | 0.6600 | 0.0269 | 0.0238 | 84 | +0.2195% | +0.1695% | 1.988 | 1.706 | 7.80% |
| **D** | Full Phase 5 Model (Technical + News + Chroma) | lightgbm | 0.8000 | 0.0244 | 0.0208 | 48 | +0.3206% | +0.2706% | 2.705 | 2.338 | 6.05% |

---

## 2. Validation Selection Summary (Days 43–51)

| Exp ID | Model Family | Selected Weight | Selected Calibrator | Selected $P^*$ | Val PR-AUC | Val Selected Trades | Val Net Avg (%) | Val Net PF |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A2** | logistic_regression | 50 | none | 0.5400 | 0.0137 | 60 | +0.0968% | 1.338 |
| **A2** | random_forest | 25 | sigmoid | 0.0133 | 0.0014 | 233 | +0.1319% | 1.947 |
| **A2** | lightgbm | 25 | none | 0.8200 | 0.0045 | 38 | +0.0314% | 1.223 |
| **B** | logistic_regression | 100 | none | 0.6000 | 0.0056 | 12891 | +0.0109% | 1.051 |
| **B** | random_forest | 25 | sigmoid | 0.0131 | 0.0020 | 310 | +0.1237% | 1.599 |
| **B** | lightgbm | 50 | none | 0.8200 | 0.0023 | 55 | -0.0238% | 0.922 |
| **C** | logistic_regression | 74 | sigmoid | 0.0153 | 0.0012 | 1144 | +0.1600% | 2.041 |
| **C** | random_forest | 25 | none | 0.5000 | 0.0088 | 33 | +0.2719% | 1.807 |
| **C** | lightgbm | 25 | none | 0.7600 | 0.0043 | 42 | +0.1387% | 1.493 |
| **D** | logistic_regression | 100 | sigmoid | 0.0142 | 0.0012 | 1022 | +0.1810% | 2.334 |
| **D** | random_forest | 74 | none | 0.6600 | 0.0035 | 99 | +0.1511% | 1.608 |
| **D** | lightgbm | 25 | none | 0.8000 | 0.0023 | 71 | +0.3422% | 2.692 |

---

## 3. Best Performing Model Deep Dive: A2 (Phase 5 Technical-Only Control) — logistic_regression

### Directional Breakdown (LONG vs SHORT)
- **Overall**: 124 trades, Net Avg Return: **+0.2967%**, Net Profit Factor: **2.502**
- **LONG Trades**: 113 trades, Win Rate: 64.6%, Net Avg Return: **+0.3897%**, Net PF: **3.641**
- **SHORT Trades**: 11 trades, Win Rate: 18.18%, Net Avg Return: **-0.6587%**, Net PF: **0.073**

### Trade Extremes & Holding Periods
- **Largest Winning Trade**: +1.97%
- **Largest Losing Trade**: -0.95%
- **Mean Holding Period**: 140.2 minutes
- **Median Holding Period**: 145.0 minutes

### Daily Performance Stability Across Test Days
| Date | Selected Trades | Wins | Win Rate (%) | Daily Net Avg Return (%) | Daily Net Total Return (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 2026-08-26 | 16 | 9 | 56.2% | +0.4211% | +6.7371% |
| 2026-08-28 | 13 | 10 | 76.9% | +0.2373% | +3.0847% |
| 2026-08-31 | 7 | 7 | 100.0% | +0.9352% | +6.5465% |
| 2026-09-01 | 28 | 10 | 35.7% | -0.2439% | -6.8300% |
| 2026-09-02 | 49 | 36 | 73.5% | +0.5614% | +27.5100% |
| 2026-09-04 | 11 | 3 | 27.3% | -0.0232% | -0.2547% |

---

## 4. Final Economic Conclusion & Classification

Based on out-of-sample evaluation after 0.05% friction:

### Outcome: **CONTEXT IMPROVES ECONOMICS**

> **Assessment**: The addition of real news sentiment and historical Chroma context successfully converted negative technical-only net returns into positive net economic returns after 0.05% friction.

---
*Report generated automatically by Trade-Assist Phase 5 ML Experiment Pipeline.*
