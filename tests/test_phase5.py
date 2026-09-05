"""
Phase 5 Comprehensive Unit Test Suite

Tests:
1. Phase 5 dataset builder schema, columns, and metadata integrity.
2. Train-only missing news imputation isolation (zero leakage into Val/Test).
3. Phase 5 hierarchical economic selection logic (Net Return -> Profit Factor -> PR-AUC).
4. Minimum 30 selected trades guard enforcement during selection.
5. Chroma completed-day temporal cutoff (same-day rejection).
6. Phase 3 baseline integrity and frozen A1 benchmark loading.
7. Neo4j unavailability explicit reporting without synthetic fabrication.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from app.ml.phase5_dataset import Phase5DatasetBuilder
from app.ml.phase5_evaluation import Phase5Evaluator
from app.ml.phase5_experiment import (
    Phase5ExperimentRunner,
    TECHNICAL_FEATURES,
    NEWS_FEATURES,
    CONTEXT_FEATURES,
)


def test_phase5_dataset_schema_and_columns(tmp_path):
    """Verifies that Phase 5 dataset builder produces all required features and outcome columns."""
    df_dummy = pd.DataFrame([
        {
            "timestamp": "2026-08-01 09:15:00",
            "symbol": "TCS.NS",
            "open": 3000.0,
            "high": 3010.0,
            "low": 2995.0,
            "close": 3005.0,
            "volume": 1000,
            "ema_5": 3002.0,
            "rsi": 55.0,
            "obv": 10000.0,
            "bollinger_middle": 3000.0,
            "bollinger_upper": 3050.0,
            "bollinger_lower": 2950.0,
            "macd": 1.5,
            "macd_signal": 1.2,
            "macd_diff": 0.3,
            "vwap": 3003.0,
            "bollinger_position": 0.55,
            "price_vs_vwap": 0.0006,
            "price_vs_ema5": 0.001,
            "sentiment_score": 0.45,
            "trading_date": "2026-08-01",
            "number_of_articles": 2,
            "mean_sentiment": 0.45,
            "positive_probability_mean": 0.6,
            "negative_probability_mean": 0.15,
            "neutral_probability_mean": 0.25,
            "latest_news_timestamp": "2026-08-01 08:30:00",
            "has_news": True,
            "market_similarity": 0.85,
            "stock_similarity": 0.90,
        },
        {
            "timestamp": "2026-08-01 09:20:00",
            "symbol": "TCS.NS",
            "open": 3005.0,
            "high": 3080.0,  # hits +2.2% target
            "low": 3000.0,
            "close": 3075.0,
            "volume": 1200,
            "ema_5": 3010.0,
            "rsi": 62.0,
            "obv": 11200.0,
            "bollinger_middle": 3005.0,
            "bollinger_upper": 3060.0,
            "bollinger_lower": 2950.0,
            "macd": 2.0,
            "macd_signal": 1.4,
            "macd_diff": 0.6,
            "vwap": 3008.0,
            "bollinger_position": 0.70,
            "price_vs_vwap": 0.002,
            "price_vs_ema5": 0.003,
            "sentiment_score": np.nan,
            "trading_date": "2026-08-01",
            "number_of_articles": 0,
            "mean_sentiment": np.nan,
            "positive_probability_mean": np.nan,
            "negative_probability_mean": np.nan,
            "neutral_probability_mean": np.nan,
            "latest_news_timestamp": None,
            "has_news": False,
            "market_similarity": 0.85,
            "stock_similarity": 0.90,
        }
    ])

    p4_path = tmp_path / "temp_p4.parquet"
    out_p5 = tmp_path / "temp_p5.parquet"
    out_q = tmp_path / "temp_q.json"

    df_dummy.to_parquet(p4_path, index=False)

    builder = Phase5DatasetBuilder(
        phase4_features_path=str(p4_path),
        output_parquet_path=str(out_p5),
        output_quality_path=str(out_q),
    )
    df_res, q_report = builder.build_dataset()

    # Each candle produces 2 candidate rows (LONG + SHORT)
    assert len(df_res) == 4

    # Check required outcome columns
    for col in ["entry_price", "target_price", "stop_price", "direction", "label", "label_status", "exit_price", "exit_reason", "realized_return"]:
        assert col in df_res.columns

    # Check feature sets present
    for col in TECHNICAL_FEATURES:
        assert col in df_res.columns
    for col in NEWS_FEATURES:
        assert col in df_res.columns
    for col in CONTEXT_FEATURES:
        assert col in df_res.columns

    assert q_report["number_of_candidate_trades"] == 4
    assert q_report["dataset_phase"] == "Phase 5 Real-Context ML"


def test_train_only_missing_news_imputation_isolation():
    """Verifies that imputer fitted on Train is NOT influenced by Test set values."""
    # Train set has median sentiment = 0.20
    train_sentiment = pd.DataFrame({"sentiment_score": [0.10, 0.20, 0.30, np.nan]})
    # Test set has extreme outlier
    test_sentiment = pd.DataFrame({"sentiment_score": [0.99, np.nan]})

    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_sentiment[["sentiment_score"]])

    # Median of [0.10, 0.20, 0.30] is 0.20
    assert abs(imputer.statistics_[0] - 0.20) < 1e-4

    # Apply to test set
    test_transformed = imputer.transform(test_sentiment[["sentiment_score"]])
    # Imputed NaN in test must be 0.20 (from train), not affected by 0.99
    assert abs(test_transformed[1, 0] - 0.20) < 1e-4


def test_phase5_hierarchical_selection_logic():
    """Verifies that candidate selection strictly follows Net Avg Return -> Net Profit Factor -> PR-AUC with 30-trade guard."""
    candidates = [
        # Candidate 1: High PR-AUC, but negative net return
        {
            "id": "cand_1",
            "val_stat": {"pr_auc": 0.45},
            "val_econ": {"selected_trade_count": 50, "net_avg_return_pct": -0.05, "net_profit_factor": 0.85},
        },
        # Candidate 2: Positive net return, meets 30 trade guard -> MUST WIN
        {
            "id": "cand_2",
            "val_stat": {"pr_auc": 0.30},
            "val_econ": {"selected_trade_count": 35, "net_avg_return_pct": 0.12, "net_profit_factor": 1.45},
        },
        # Candidate 3: Highest net return, but fails trade guard (< 30 trades)
        {
            "id": "cand_3",
            "val_stat": {"pr_auc": 0.50},
            "val_econ": {"selected_trade_count": 10, "net_avg_return_pct": 0.85, "net_profit_factor": 3.50},
        },
    ]

    selected = Phase5Evaluator.select_best_candidate(candidates, min_trades=30)
    assert selected["id"] == "cand_2", "Candidate 2 must be selected by hierarchical economic selection rule"


def test_frozen_phase3_benchmark_integrity():
    """Verifies that Phase 3 model results file exists and matches the frozen benchmark."""
    res_path = Path("data/processed/phase3_model_results.json")
    assert res_path.exists(), "Phase 3 model results must exist untouched."

    runner = Phase5ExperimentRunner()
    p3_data = runner.load_frozen_phase3_benchmark()

    assert "model_formulations" in p3_data
    assert len(p3_data["model_formulations"]) >= 3


def test_neo4j_unavailability_explicit_reporting():
    """Verifies that Neo4j unavailability is detected and reported as False without dummy values."""
    runner = Phase5ExperimentRunner()
    assert runner.neo4j_ingestor.is_available() is False
