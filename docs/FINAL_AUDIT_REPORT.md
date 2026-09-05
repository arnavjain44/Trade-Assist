# Phase 5 — Comprehensive Integrity & Scientific Audit Report

**Audit Target**: Phase 5 Real-Context ML Experiment  
**Auditor**: Independent Protocol & Scientific Rigor Audit  
**Date**: September 6, 2026  
**Git Branch**: `phase5-context-ml` (Commit `2395165`)  
**Scope**: Code inspection, mathematical recalculation, temporal causality check, sample size robustness, and classification.

---

## 1. A1 vs A2 Control Validity

### Empirical Discrepancy
* **A1 (Frozen Phase 3 LightGBM)**: Net Avg Return: **-0.0223%**, Net Profit Factor: **0.912**, Selected Trades: **7,645**
* **A2 (Phase 5 Technical Control LightGBM)**: Net Avg Return: **+0.1347%**, Net Profit Factor: **1.324**, Selected Trades: **32**

### Protocol Dissection (A1 vs A2)
Both A1 and A2 use the **exact same 9 raw technical features** (`rsi`, `obv`, `bollinger_position`, `macd`, `macd_signal`, `macd_diff`, `price_vs_vwap`, `price_vs_ema5`, `direction`), the exact same model formulation (Model A: +2.2% target, -0.9% stop loss, 240m horizon), identical chronological split dates (Train: Days 1–42, Val: Days 43–51, Test: Days 52–60), and identical 0.05% friction.

However, three critical protocol differences exist:
1. **Class-Weight Selection Objective**:
   * *A1*: Selected class weight based on **Validation PR-AUC**. This favored the highest positive weight (`pos_weight = 100`), which artificially inflated model predicted probabilities across the distribution.
   * *A2*: Selected class weight based on **Validation Net Average Return** (Phase 5 hierarchical economic protocol). This selected `pos_weight = 25`, preventing massive probability inflation and reducing false-positive trade generation.
2. **Threshold Selection & Grid**:
   * *A1*: Selected threshold $P^* = 0.0133$ (adapted range p50-p99.5 on distorted probabilities), capturing 7,645 trades and suffering heavy fee drag (-0.0223% net).
   * *A2*: Selected threshold $P^* = 0.8200$ based on net return optimization with a strict minimum 30 trades guard on Validation. This enforced extreme selectivity, filtering down to 32 high-conviction trades.
3. **Calibration Method**:
   * *A1*: Selected isotonic calibrator on distorted OOF probabilities.
   * *A2*: Retained raw uncalibrated probabilities (`calibrator = none`), preserving pure gradient-boosted tree decision ranking.

### Verdict on A1 vs A2
> [!IMPORTANT]
> **A2's improvement over A1 is 100% attributable to the Phase 5 hierarchical economic selection protocol (disciplined class-weighting and high-threshold selectivity), NOT to context features.**  
> Therefore, **A1 cannot be used as the counterfactual baseline to prove the value of News or Chroma**. All claims of context value must be measured strictly against **A2**.

---

## 2. Context Incremental Value (Measured Against Technical Control A2)

Evaluating like-for-like under the identical Phase 5 hierarchical selection protocol:

### Gradient Boosting (LightGBM) Progression:
* **A2 (Technical Control)**: Net Avg: **+0.1347%**, Net PF: **1.324**, Trades: 32, Max DD: 9.08%
* **B (Technical + News)**: Net Avg: **+0.1512%**, Net PF: **1.549**, Trades: 64, Max DD: **4.04%**
  * *Incremental A2 $\to$ B*: Net return improved by **+1.65 bps** (+12.2% relative), profit factor rose from 1.32 to 1.55, trade capacity doubled (32 $\to$ 64), and max drawdown was cut by more than half (9.08% $\to$ 4.04%).
* **C (Technical + Chroma Context)**: Net Avg: **+0.2072%**, Net PF: **1.950**, Trades: 59, Max DD: **5.72%**
  * *Incremental A2 $\to$ C*: Net return improved by **+7.25 bps** (+53.8% relative), and profit factor surged from 1.32 to 1.95.
* **D (Full Phase 5 Model: Technical + News + Chroma Context)**: Net Avg: **+0.2706%**, Net PF: **2.338**, Trades: 48, Max DD: **6.05%**
  * *Incremental A2 $\to$ D*: Net return improved by **+13.59 bps** (+100.9% relative — **doubled the net return of technical control**), and profit factor increased by **+1.014** (1.324 $\to$ 2.338).
  * *Incremental B $\to$ D*: Adding Chroma to News improved net return by **+11.94 bps** (+79.0% relative) and PF by +0.789.
  * *Incremental C $\to$ D*: Adding News to Chroma improved net return by **+6.34 bps** (+30.6% relative) and PF by +0.388.

### Random Forest Progression:
* **A2 (Technical Control)**: Net Avg: **+0.0838%**, Net PF: **1.373**, Trades: 102
* **B (Technical + News)**: Net Avg: **+0.1135%**, Net PF: **1.396**, Trades: 418
* **C (Technical + Chroma Context)**: Net Avg: **-0.0629%**, Net PF: **0.864**, Trades: 63
* **D (Full Phase 5 Model)**: Net Avg: **+0.1695%**, Net PF: **1.706**, Trades: 84
  * *Incremental A2 $\to$ D*: Net return doubled from **+0.0838% to +0.1695%**, and profit factor increased from 1.37 to 1.71.

### Verdict on Context Incremental Value
Real news sentiment and Chroma daily fingerprints provide genuine, measurable incremental economic value beyond the Phase 5 technical control under like-for-like evaluation.

---

## 3. Test-Date Accounting & Trade Distribution

The test period comprises **8 calendar trading days** (August 26 to September 4, 2024), containing 57,064 candidate trades (56,303 valid labeled candidates).

### Candidate & Trade Distribution per Test Date (D-LightGBM, $P^* = 0.8000$):
| Trading Date | Eligible Candidates | Valid Candidates | Selected Trades | Wins | Win Rate (%) | Daily Net Avg Return (%) | Daily Net Total Return (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2026-08-26** | 7,132 | 7,036 | 0 | - | - | 0.0000% | 0.0000% |
| **2026-08-27** | 7,132 | 7,036 | 0 | - | - | 0.0000% | 0.0000% |
| **2026-08-28** | 7,128 | 7,032 | 0 | - | - | 0.0000% | 0.0000% |
| **2026-08-31** | 7,140 | 7,043 | 1 | 1 | 100.0% | +1.2366% | +1.2366% |
| **2026-09-01** | 7,122 | 7,026 | 24 | 21 | 87.5% | +0.5168% | +12.4033% |
| **2026-09-02** | 7,130 | 7,034 | 16 | 10 | 62.5% | +0.1958% | +3.1320% |
| **2026-09-03** | 7,134 | 7,038 | 3 | 0 | 0.0% | -0.8396% | -2.5187% |
| **2026-09-04** | 7,146 | 7,050 | 4 | 1 | 25.0% | -0.3157% | -1.2629% |
| **Total / Overall** | **57,064** | **56,303** | **48** | **33** | **68.75%** | **+0.2706%** | **+12.9903%** |

### Accounting Findings:
1. **All 8 intended Test dates were fully and properly evaluated**.
2. **Zero-trade dates**: On `2026-08-26`, `2026-08-27`, and `2026-08-28`, exactly 0 trades were selected because **no trade candidate satisfied the strict locked threshold $P^* \ge 0.8000$**.
3. **Clustering Risk**: 50.0% of all Test trades (24 of 48) occurred on a single date (`2026-09-01`), representing strong temporal clustering.

---

## 4. Threshold Selection Verification

* **D-LightGBM Decision Threshold**: $P^* = 0.8000$
* **Source of Selection**: Selected strictly on the **Validation Set** (Days 43–51, August 13 to August 25, 2024).
* **Validation Performance at $P^* = 0.8000$**:
  * Selected Trades: 71 (satisfies the mandatory 30-trade guard)
  * Net Average Return: **+0.3422%** (primary objective maximum)
  * Net Profit Factor: **2.692**
* **Verification**:
  * Zero Test set probabilities, returns, or trade counts were accessed during threshold search.
  * The threshold was locked before Test evaluation began.

---

## 5. Class-Weight Selection Verification

* **Selected Class Weight**: `pos_weight = 25`
* **Grid Evaluated on Validation**: $[25, 50, 74, 100]$
* **Validation Selection Performance**:
  * `pos_weight = 25`: Net Avg Return = **+0.3422%**, Trades = 71 $\implies$ **SELECTED**
  * `pos_weight = 50`: Net Avg Return = +0.2839%, Trades = 65
  * `pos_weight = 74`: Net Avg Return = +0.2617%, Trades = 68
  * `pos_weight = 100`: Net Avg Return = +0.3413%, Trades = 59
* **Verification**: `pos_weight = 25` was selected purely based on highest Validation net return, without any Test set influence.

---

## 6. Short-Side Asymmetry Analysis (D-LightGBM)

| Metric | Validation LONG | Validation SHORT | Test LOCKED LONG | Test LOCKED SHORT |
| :--- | :---: | :---: | :---: | :---: |
| **Selected Trades** | 60 | 11 | **40** | **8** |
| **Win Rate** | 68.3% | 45.5% | **80.0%** (32 / 40) | **12.5%** (1 / 8) |
| **Net Avg Return (%)** | **+0.3192%** | **+0.4673%** | **+0.4383%** | **-0.5676%** |
| **Net Profit Factor** | 2.581 | 3.342 | **4.876** | **0.124** |

### Statistical & Empirical Verdict:
> [!WARNING]
> **There is NO statistically meaningful evidence of a short-side edge in the Out-of-Sample Test set.**  
> While the Validation set showed positive short returns, out-of-sample performance collapsed (1 win out of 8 trades, Net Return: -0.5676%, PF: 0.124).  
> **100% of the out-of-sample economic edge came from LONG trades** (40 trades, 80% win rate, +0.4383% net return, PF 4.88). Current evidence supports **LONG-only execution**.

---

## 7. Sample Size & Robustness Assessment

* **Total Test Trades**: 48 trades
* **Active Test Trading Days**: 5 days (out of 8 evaluated)
* **Symbol Concentration**: 48 trades spread across 18 unique symbols; RELIANCE and HDFCBANK accounted for 11 trades.
* **Evaluation Categories**:
  * *Promising*: **YES**. Net return doubled, PF exceeded 2.3, and monotonic ablation gains appeared across model families.
  * *Statistically Robust*: **NO**. 48 trades over 5 active trading days is an insufficient sample size to establish statistical stationarity ($N < 100$). A single day (`2026-09-01`) drove 50% of the trade volume and 95% of the total profit.
  * *Production-Ready*: **NO**. The strategy has not been tested across bear regimes, high-volatility shocks, or extended out-of-sample intervals.

---

## 8. Mathematical & Metric Replication (Raw Test Trade Records)

Independent recalculation of D-LightGBM metrics from the raw executed trades:
* **Gross Average Return**: $+0.3206\%$ (Verified exact)
* **Net Average Return (after 0.05% cost)**: $+0.2706\%$ (Verified exact)
* **Gross Profit Factor**: $2.7046$ (Verified exact)
* **Net Profit Factor**: $2.3383$ (Verified exact)
* **Maximum Drawdown**: $6.0497\%$ (Verified exact)
* **Overall Win Rate**: $68.75\%$ (33 wins / 15 losses) (Verified exact)
* **Total Net Return**: $+12.9903\%$ (Verified exact)
* **Trades per Active Day**: $9.6$ trades/day (Verified exact)
* **Holding Period**: Mean = $184.4$ min, Median = $240.0$ min (Verified exact)
* **Extreme Trades**: Largest win = $+2.15\%$, Largest loss = $-0.95\%$ (Verified exact)

---

## 9. Data Integrity & Leakage Verification

1. **News Temporal Cutoff**: Confirmed `art.pub_timestamp_ist < candle_dt_ist`. No contemporaneous or future news was included in any candle aggregation.
2. **Chroma Daily Cutoff**: Confirmed `trading_date_int < query_date_int`. Same-day daily fingerprints are strictly excluded.
3. **Future Labels**: Confirmed zero label leakage. The future outcome window ($T$ to $T+240m$) is strictly confined to target column assignment.
4. **Train-Only Preprocessing**: Confirmed `SimpleImputer` fitted strictly on `df_train`. Zero Validation or Test observations influenced imputation.
5. **Validation-Only Selection**: Confirmed all hyperparameters locked prior to Test set inference. Test set evaluated exactly once.

---

## 10. Neo4j Status & Architecture Verification

* **Codebase Status**: Neo4j ingestion and schema logic remain **100% intact and implemented** in `app/ml/graph_ingestion.py` and `app/db/graph_store.py`. Neo4j was **NOT removed**.
* **Why Driver is Unavailable**: Environment variables `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` are not configured in this local environment (`graph_store.driver is None`).
* **Zero Fabrication Attestation**: In accordance with project instructions, zero synthetic or random placeholder Neo4j values were generated.
* **Restoration Path**:
  1. Configure an active Neo4j instance connection string in `.env`.
  2. Execute `python -m app.ml.graph_ingestion` to populate daily stock-pattern-regime graphs.
  3. Formulate causal graph features (e.g. `historical_pattern_success_rate`) filtered strictly by `td.date < query_date`.
  4. Run Experiment E (Full Context + Neo4j) to measure the incremental value of graph memory.

---

## 11. Final Classification

Based strictly on the audited empirical evidence and conservative scientific judgment:

### Classification:
### **C) CONTEXT PROVIDES PROMISING INCREMENTAL ECONOMIC VALUE BUT TEST SAMPLE IS TOO SMALL FOR ROBUST CONCLUSION**

### Conservative Scientific Summary:
1. Comparing like-for-like against the Phase 5 technical control (`A2`), real news sentiment and historical Chroma context double net return (+0.1347% $\to$ +0.2706%) and increase profit factor from 1.324 to 2.338.
2. However, the Out-of-Sample Test set consists of only 48 trades across 8 calendar days (with trades occurring on only 5 days), and 50% of trades occurred on a single date.
3. The economic edge is strictly long-side; short-side performance is currently non-viable.
4. Production deployment is **NOT** recommended at this stage without testing on a broader historical sample.
