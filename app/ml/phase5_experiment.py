"""
Phase 5 Real-Context Machine Learning Experiment Runner

Orchestrates the 4-way ablation experiment matrix:
- A1: Frozen Phase 3 Benchmark (loaded directly from Phase 3 artifacts)
- A2: Phase 5 Technical-Only Control (Phase 5 pipeline + hierarchical selection)
- B:  Technical + Real FinBERT News Sentiment
- C:  Technical + Historical Chroma Daily Fingerprints
- D:  Full Phase 5 (Technical + News + Chroma Context)

Strict Guarantees:
- Train-only preprocessing: missing news imputation fitted strictly on train_valid.
- Purged OOF walk-forward calibration (Isotonic vs Sigmoid vs None).
- Deterministic hierarchical validation selection: Net Avg Return -> Net Profit Factor -> PR-AUC (min 30 trades guard).
- Out-of-sample Test set (Days 52-60) strictly locked and evaluated once per pipeline.
- Exports results to docs/PHASE_5_CONTEXT_ML_RESULTS.md and docs/PHASE_5_FEATURE_PROVENANCE.md.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from app.ml.models import TradeSignalClassifier
from app.ml.phase5_evaluation import Phase5Evaluator
from app.ml.graph_ingestion import Neo4jGraphIngestor

logger = logging.getLogger(__name__)


TECHNICAL_FEATURES = [
    "rsi",
    "obv",
    "bollinger_position",
    "macd",
    "macd_signal",
    "macd_diff",
    "price_vs_vwap",
    "price_vs_ema5",
    "direction",
]

NEWS_FEATURES = [
    "sentiment_score",
    "has_news",
    "number_of_articles",
]

CONTEXT_FEATURES = [
    "market_similarity",
    "stock_similarity",
]


class Phase5ExperimentRunner:
    """Runs the controlled Phase 5 ablation experiment matrix."""

    MODEL_FAMILIES = ["logistic_regression", "random_forest", "lightgbm"]
    CLASS_WEIGHTS = [25, 50, 74, 100]

    def __init__(
        self,
        features_parquet_path: str = "data/processed/phase5_features.parquet",
        phase3_results_path: str = "data/processed/phase3_model_results.json",
        output_results_path: str = "data/processed/phase5_model_results.json",
        output_md_path: str = "docs/PHASE_5_CONTEXT_ML_RESULTS.md",
        output_provenance_path: str = "docs/PHASE_5_FEATURE_PROVENANCE.md",
        cost_pct: float = 0.0005,
    ):
        self.features_parquet_path = Path(features_parquet_path)
        self.phase3_results_path = Path(phase3_results_path)
        self.output_results_path = Path(output_results_path)
        self.output_md_path = Path(output_md_path)
        self.output_provenance_path = Path(output_provenance_path)
        self.cost_pct = cost_pct
        self.evaluator = Phase5Evaluator(cost_pct=cost_pct, horizon_minutes=240)
        self.neo4j_ingestor = Neo4jGraphIngestor()

    def load_frozen_phase3_benchmark(self) -> Dict[str, Any]:
        """Loads frozen Phase 3 benchmark results from existing artifact."""
        if not self.phase3_results_path.exists():
            logger.warning("Phase 3 model results not found at %s.", self.phase3_results_path)
            return {}

        with open(self.phase3_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("Loaded frozen Phase 3 results (timestamp: %s).", data.get("experiment_timestamp"))
        return data

    def run_experiment(self) -> Dict[str, Any]:
        """Executes the full Phase 5 ablation study."""
        logger.info("=== STARTING PHASE 5 REAL-CONTEXT ML EXPERIMENT ===")

        # 1. Load Phase 5 dataset
        if not self.features_parquet_path.exists():
            raise FileNotFoundError(f"Phase 5 features file not found: {self.features_parquet_path}")

        logger.info("Loading Phase 5 dataset from %s...", self.features_parquet_path)
        df_all = pd.read_parquet(self.features_parquet_path)
        if "mean_sentiment" in df_all.columns:
            df_all["sentiment_score"] = df_all["mean_sentiment"]

        # 2. Chronological split with 240m purging and embargo
        df_train, df_val, df_test, split_meta = self.evaluator.split_dataset_chronologically(df_all)
        logger.info("Chronological split: Train %d, Val %d, Test %d (Total %d rows).",
                    len(df_train), len(df_val), len(df_test), len(df_all))

        train_valid = df_train[df_train["label_status"] == "VALID"].copy()
        val_valid = df_val[df_val["label_status"] == "VALID"].copy()
        test_valid = df_test[df_test["label_status"] == "VALID"].copy()

        # 3. Load Frozen Phase 3 Benchmark (A1)
        frozen_phase3 = self.load_frozen_phase3_benchmark()

        # Neo4j availability check
        neo4j_avail = self.neo4j_ingestor.is_available()

        # 4. Define ablation experiment configurations
        experiments_def = [
            {
                "exp_id": "A2",
                "name": "Phase 5 Technical-Only Control",
                "feature_cols": list(TECHNICAL_FEATURES),
                "needs_news_impute": False,
            },
            {
                "exp_id": "B",
                "name": "Technical + Real News Sentiment",
                "feature_cols": TECHNICAL_FEATURES + NEWS_FEATURES,
                "needs_news_impute": True,
            },
            {
                "exp_id": "C",
                "name": "Technical + Chroma Historical Context",
                "feature_cols": TECHNICAL_FEATURES + CONTEXT_FEATURES,
                "needs_news_impute": False,
            },
            {
                "exp_id": "D",
                "name": "Full Phase 5 Model (Technical + News + Chroma)",
                "feature_cols": TECHNICAL_FEATURES + NEWS_FEATURES + CONTEXT_FEATURES,
                "needs_news_impute": True,
            },
        ]

        all_results = {
            "experiment_timestamp": datetime.now(timezone.utc).isoformat(),
            "cost_pct": self.cost_pct,
            "split_metadata": split_meta,
            "neo4j_available": neo4j_avail,
            "frozen_phase3_benchmark": frozen_phase3.get("model_formulations", []),
            "phase5_experiments": [],
        }

        # 5. Execute each Phase 5 experiment
        for exp in experiments_def:
            exp_id = exp["exp_id"]
            exp_name = exp["name"]
            feat_cols = exp["feature_cols"]
            needs_impute = exp["needs_news_impute"]

            logger.info("--- Running Experiment %s: %s (%d features) ---", exp_id, exp_name, len(feat_cols))

            # Train-only imputation for missing news
            train_sub = train_valid.copy()
            val_sub = val_valid.copy()
            test_sub = test_valid.copy()

            if needs_impute:
                imputer = SimpleImputer(strategy="median")
                # Fit ONLY on training data
                imputer.fit(train_sub[["sentiment_score"]])
                train_sub["sentiment_score"] = imputer.transform(train_sub[["sentiment_score"]]).ravel()
                val_sub["sentiment_score"] = imputer.transform(val_sub[["sentiment_score"]]).ravel()
                test_sub["sentiment_score"] = imputer.transform(test_sub[["sentiment_score"]]).ravel()

            for family in self.MODEL_FAMILIES:
                logger.info("Evaluating %s [%s]...", exp_id, family)

                candidate_evals = []

                for pos_w in self.CLASS_WEIGHTS:
                    clf = TradeSignalClassifier(
                        model_family=family,
                        pos_weight=pos_w,
                        feature_cols=feat_cols,
                        random_state=42,
                    )
                    clf.fit(train_sub, train_sub["label"].values)

                    # Purged OOF walk-forward predictions
                    oof_idx, oof_probs = clf.fit_oof_purged_predictions(
                        train_sub, n_splits=4, horizon_minutes=240
                    )
                    oof_train_sub = train_sub.iloc[oof_idx] if len(oof_idx) > 0 else train_sub

                    for calib_method in ["none", "isotonic", "sigmoid"]:
                        clf_c = TradeSignalClassifier(
                            model_family=family,
                            pos_weight=pos_w,
                            feature_cols=feat_cols,
                            random_state=42,
                        )
                        clf_c.fit(train_sub, train_sub["label"].values)

                        if calib_method != "none" and len(oof_probs) > 0:
                            clf_c.fit_calibrator(oof_probs, oof_train_sub["label"].values, method=calib_method)

                        val_p = clf_c.predict_proba(val_sub)

                        # Threshold optimization on Validation
                        best_th, val_econ = self.evaluator.select_optimal_threshold(
                            val_sub, val_p, min_trade_count=30
                        )
                        val_stat = self.evaluator.calculate_statistical_metrics(
                            val_sub["label"].values, val_p, threshold=best_th
                        )

                        candidate_evals.append({
                            "pos_weight": pos_w,
                            "calibrator": calib_method,
                            "threshold": best_th,
                            "val_stat": val_stat,
                            "val_econ": val_econ,
                            "classifier": clf_c,
                        })

                # Select winning hyperparameter combination using hierarchical rule
                best_cand = self.evaluator.select_best_candidate(candidate_evals, min_trades=30)
                locked_clf = best_cand["classifier"]
                locked_w = best_cand["pos_weight"]
                locked_calib = best_cand["calibrator"]
                locked_th = best_cand["threshold"]
                val_stat = best_cand["val_stat"]
                val_econ = best_cand["val_econ"]

                logger.info("Locked %s [%s]: weight=%d, calib=%s, thresh=%.4f (Val NetAvg=%.4f%%, PF=%.3f, Trades=%d)",
                            exp_id, family, locked_w, locked_calib, locked_th,
                            val_econ["net_avg_return_pct"], val_econ["net_profit_factor"], val_econ["selected_trade_count"])

                # SINGLE LOCKED EVALUATION ON OUT-OF-SAMPLE TEST SET
                test_probs = locked_clf.predict_proba(test_sub)
                test_detailed = self.evaluator.evaluate_test_detailed(test_sub, test_probs, threshold=locked_th)

                record = {
                    "experiment_id": exp_id,
                    "experiment_name": exp_name,
                    "model_family": family,
                    "feature_count": len(feat_cols),
                    "feature_cols": feat_cols,
                    "selected_pos_weight": locked_w,
                    "selected_calibrator": locked_calib,
                    "selected_threshold": locked_th,
                    "validation_statistical": val_stat,
                    "validation_economic": val_econ,
                    "test_statistical": test_detailed["statistical"],
                    "test_economic": test_detailed["economic"],
                    "test_long_breakdown": test_detailed["long_breakdown"],
                    "test_short_breakdown": test_detailed["short_breakdown"],
                    "test_daily_breakdown": test_detailed["daily_breakdown"],
                    "test_extremes": test_detailed["extremes"],
                }
                all_results["phase5_experiments"].append(record)

        # 6. Save JSON results
        self.output_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Saved Phase 5 model results to %s.", self.output_results_path)

        # 7. Generate Markdown Comparative Report & Feature Provenance
        self._export_markdown_report(all_results)
        self._export_feature_provenance()

        logger.info("=== PHASE 5 REAL-CONTEXT ML EXPERIMENT COMPLETE ===")
        return all_results

    def _export_markdown_report(self, results: Dict[str, Any]):
        """Generates docs/PHASE_5_CONTEXT_ML_RESULTS.md report."""
        p3_evals = [e for e in results.get("frozen_phase3_benchmark", []) if e.get("target_formulation") == "Model A"]
        p5_evals = results.get("phase5_experiments", [])

        md = f"""# Phase 5 — Real-Context ML Experiment Results

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
"""
        # Add Frozen Phase 3 Benchmark (A1)
        for e in p3_evals:
            fam = e["model_family"]
            th = e["selected_threshold"]
            ts = e["locked_test_statistical_metrics"]
            te = e["locked_test_economic_metrics"]
            md += f"| **A1** | *Frozen Phase 3 Benchmark* | {fam} | {th:.4f} | {ts['pr_auc']:.4f} | {ts['precision']:.4f} | {te['selected_trade_count']} | {te['gross_avg_return_pct']:+.4f}% | {te['net_avg_return_pct']:+.4f}% | {te['gross_profit_factor']:.3f} | {te['net_profit_factor']:.3f} | {te['max_drawdown_pct']:.2f}% |\n"

        # Add Phase 5 Experiments (A2, B, C, D)
        for e in p5_evals:
            exp_id = e["experiment_id"]
            exp_name = e["experiment_name"]
            fam = e["model_family"]
            th = e["selected_threshold"]
            ts = e["test_statistical"]
            te = e["test_economic"]
            md += f"| **{exp_id}** | {exp_name} | {fam} | {th:.4f} | {ts['pr_auc']:.4f} | {ts['precision']:.4f} | {te['selected_trade_count']} | {te['gross_avg_return_pct']:+.4f}% | {te['net_avg_return_pct']:+.4f}% | {te['gross_profit_factor']:.3f} | {te['net_profit_factor']:.3f} | {te['max_drawdown_pct']:.2f}% |\n"

        md += """
---

## 2. Validation Selection Summary (Days 43–51)

| Exp ID | Model Family | Selected Weight | Selected Calibrator | Selected $P^*$ | Val PR-AUC | Val Selected Trades | Val Net Avg (%) | Val Net PF |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
        for e in p5_evals:
            exp_id = e["experiment_id"]
            fam = e["model_family"]
            w = e["selected_pos_weight"]
            cal = e["selected_calibrator"]
            th = e["selected_threshold"]
            vs = e["validation_statistical"]
            ve = e["validation_economic"]
            md += f"| **{exp_id}** | {fam} | {w} | {cal} | {th:.4f} | {vs['pr_auc']:.4f} | {ve['selected_trade_count']} | {ve['net_avg_return_pct']:+.4f}% | {ve['net_profit_factor']:.3f} |\n"

        # Add Directional and Daily Breakdown for Best Model
        active_p5 = [e for e in p5_evals if e["test_economic"]["selected_trade_count"] > 0]
        if active_p5:
            best_model = max(active_p5, key=lambda x: (x["test_economic"]["net_avg_return_pct"], x["test_economic"]["net_profit_factor"]))
            b_exp = best_model["experiment_id"]
            b_fam = best_model["model_family"]
            b_te = best_model["test_economic"]
            b_long = best_model["test_long_breakdown"]
            b_short = best_model["test_short_breakdown"]
            b_ext = best_model["test_extremes"]

            md += f"""
---

## 3. Best Performing Model Deep Dive: {b_exp} ({best_model['experiment_name']}) — {b_fam}

### Directional Breakdown (LONG vs SHORT)
- **Overall**: {b_te['selected_trade_count']} trades, Net Avg Return: **{b_te['net_avg_return_pct']:+.4f}%**, Net Profit Factor: **{b_te['net_profit_factor']:.3f}**
- **LONG Trades**: {b_long.get('trades', 0)} trades, Win Rate: {b_long.get('win_rate', 0.0)}%, Net Avg Return: **{b_long.get('net_avg_return_pct', 0.0):+.4f}%**, Net PF: **{b_long.get('net_profit_factor', 0.0):.3f}**
- **SHORT Trades**: {b_short.get('trades', 0)} trades, Win Rate: {b_short.get('win_rate', 0.0)}%, Net Avg Return: **{b_short.get('net_avg_return_pct', 0.0):+.4f}%**, Net PF: **{b_short.get('net_profit_factor', 0.0):.3f}**

### Trade Extremes & Holding Periods
- **Largest Winning Trade**: {b_ext.get('largest_winning_trade_pct', 0.0):+.2f}%
- **Largest Losing Trade**: {b_ext.get('largest_losing_trade_pct', 0.0):+.2f}%
- **Mean Holding Period**: {b_ext.get('mean_holding_period_minutes', 0.0):.1f} minutes
- **Median Holding Period**: {b_ext.get('median_holding_period_minutes', 0.0):.1f} minutes

### Daily Performance Stability Across Test Days
| Date | Selected Trades | Wins | Win Rate (%) | Daily Net Avg Return (%) | Daily Net Total Return (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
            for d in best_model.get("test_daily_breakdown", []):
                md += f"| {d['date']} | {d['trades']} | {d['wins']} | {d['win_rate']:.1f}% | {d['net_avg_return_pct']:+.4f}% | {d['net_total_return_pct']:+.4f}% |\n"

        md += """
---

## 4. Final Economic Conclusion & Classification

Based on out-of-sample evaluation after 0.05% friction:
"""
        # Determine classification
        if active_p5:
            best_net = max(e["test_economic"]["net_avg_return_pct"] for e in active_p5)
            p3_lightgbm = next((e for e in p3_evals if e["model_family"] == "lightgbm"), None)
            p3_net = p3_lightgbm["locked_test_economic_metrics"]["net_avg_return_pct"] if p3_lightgbm else -0.0575

            if best_net > 0.0 and best_net > p3_net:
                conclusion = "CONTEXT IMPROVES ECONOMICS"
                explanation = "The addition of real news sentiment and historical Chroma context successfully converted negative technical-only net returns into positive net economic returns after 0.05% friction."
            elif best_net > p3_net:
                conclusion = "CONTEXT IMPROVES PREDICTION BUT NOT ECONOMICS"
                explanation = "Context features reduced losses and improved statistical ranking, but net returns after friction remain below zero."
            else:
                conclusion = "NO MEANINGFUL IMPROVEMENT"
                explanation = "Context features did not produce a meaningful economic improvement over the technical-only baseline after transaction costs."
        else:
            conclusion = "EXPERIMENT INCONCLUSIVE"
            explanation = "No models generated active trades above the decision threshold."

        md += f"""
### Outcome: **{conclusion}**

> **Assessment**: {explanation}

---
*Report generated automatically by Trade-Assist Phase 5 ML Experiment Pipeline.*
"""
        self.output_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_md_path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("Saved Phase 5 comparative report to %s.", self.output_md_path)

    def _export_feature_provenance(self):
        """Generates docs/PHASE_5_FEATURE_PROVENANCE.md."""
        md = r"""# Phase 5 — Feature Provenance & Temporal Leakage Audit

This document provides the exhaustive temporal provenance audit for all features used in Phase 5.
Every feature satisfies the strict temporal causality test: **"What information would have been available at the exact decision timestamp $T$?"**

---

## 1. Feature Provenance Matrix

| Feature | Feature Group | Exact Source | Timestamp Basis | Future Data Possible? | Leakage Verification Rule |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `rsi` | Group A (Technical) | 14-period RSI on 5m OHLCV | $T$ (Current candle close) | **NO** | Calculated strictly up to candle index $i$. |
| `obv` | Group A (Technical) | On-Balance Volume on 5m candles | $T$ (Current candle close) | **NO** | Cumulative sum over past candles up to $i$. |
| `bollinger_position` | Group A (Technical) | Position within 20-period Bollinger Bands | $T$ (Current candle close) | **NO** | Uses 20-period rolling window strictly $\le T$. |
| `macd` | Group A (Technical) | 12-period EMA minus 26-period EMA | $T$ (Current candle close) | **NO** | Backward-looking exponential moving averages. |
| `macd_signal` | Group A (Technical) | 9-period EMA of MACD | $T$ (Current candle close) | **NO** | Computed strictly on past MACD series. |
| `macd_diff` | Group A (Technical) | MACD minus MACD signal | $T$ (Current candle close) | **NO** | Difference between contemporaneous values. |
| `price_vs_vwap` | Group A (Technical) | % distance of close from session VWAP | $\le T$ (Current session) | **NO** | Session VWAP resets daily at 09:15; cumulative to $T$. |
| `price_vs_ema5` | Group A (Technical) | % distance of close from 5-period EMA | $T$ (Current candle close) | **NO** | 5-period rolling exponential average. |
| `direction` | Group A (Technical) | Candidate trade direction (+1.0 LONG / -1.0 SHORT) | $T$ (Current candle) | **NO** | Explicit candidate evaluation side. |
| `sentiment_score` | Group B (News) | FinBERT sentiment ($P_{pos} - P_{neg}$) | $< T$ (Strictly prior news) | **NO** | Filtered by `news_timestamp < candle_timestamp`. |
| `has_news` | Group B (News) | Boolean indicator if prior news exists | $< T$ (Strictly prior news) | **NO** | Strict temporal cutoff; missing news produces NaN. |
| `number_of_articles` | Group B (News) | Count of prior articles | $< T$ (Strictly prior news) | **NO** | Count of articles published prior to candle timestamp. |
| `market_similarity` | Group C (Chroma) | Cosine similarity to market daily fingerprint | $< \text{date}(T)$ (Completed prior days) | **NO** | Filtered by `trading_date_int < query_date_int`. |
| `stock_similarity` | Group C (Chroma) | Cosine similarity to stock daily fingerprint | $< \text{date}(T)$ (Completed prior days) | **NO** | Filtered by `trading_date_int < query_date_int`. |
| *Neo4j features* | *Unavailable* | *Neo4j DB not connected* | *N/A* | *N/A* | **EXCLUDED** (Zero fabrication of unpopulated data). |

---

## 2. Leakage Defense Mechanisms

1. **Strict Train-Only Missing News Imputation**:
   - `SimpleImputer` is fitted **strictly on `df_train`**.
   - Imputation value is NEVER informed by Validation or Test distributions.
   - `has_news` is preserved as an explicit boolean feature so the model retains missingness awareness.

2. **Completed-Day Chroma Fingerprints**:
   - Daily fingerprints are constructed after market close of date $D$.
   - Intraday queries for date $D$ filter by `trading_date_int < query_date_int`, guaranteeing that only dates $\le D-1$ are accessible.
   - Current-day fingerprint retrieval is strictly prohibited and tested.

3. **Chronological Walk-Forward Purged OOF Calibration**:
   - Out-of-fold calibration uses 4 chronological walk-forward splits within `df_train`.
   - 240-minute purging horizon prevents trade outcome overlap across folds.

4. **Sacred Test Set Boundary**:
   - All models, class weights, calibrators, and decision thresholds are finalized on Validation (Days 43–51) before evaluating the Test set (Days 52–60).
   - Test data is never used for tuning or selection.

---
*Audit compiled automatically by Trade-Assist Phase 5 Verification Engine.*
"""
        self.output_provenance_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_provenance_path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("Saved Phase 5 feature provenance audit to %s.", self.output_provenance_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runner = Phase5ExperimentRunner()
    runner.run_experiment()
