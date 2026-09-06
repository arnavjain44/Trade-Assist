"""
Unit Tests for Phase 1: Phase 5 D-LightGBM Model Artifact

Verifies:
1. Artifact file exists on disk.
2. Artifact loads successfully with joblib.
3. Model is actually LightGBM (LGBMClassifier).
4. Feature order is exactly the 14 Phase 5 features in canonical sequence.
5. Threshold is strictly 0.8000.
6. Preprocessing (SimpleImputer) is present and fitted on training data.
7. Model is deterministic and not synthetic.
8. Known-row probability matches research implementation identically.
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
import lightgbm as lgb

from app.ml.models import TradeSignalClassifier
from app.ml.phase5_evaluation import Phase5Evaluator

ARTIFACT_PATH = "data/models/phase5_d_lightgbm.joblib"
PHASE5_FEATURES_PATH = "data/processed/phase5_features.parquet"

EXPECTED_FEATURE_COLS = [
    "rsi",
    "obv",
    "bollinger_position",
    "macd",
    "macd_signal",
    "macd_diff",
    "price_vs_vwap",
    "price_vs_ema5",
    "direction",
    "sentiment_score",
    "has_news",
    "number_of_articles",
    "market_similarity",
    "stock_similarity",
]


def test_artifact_exists():
    """Verify exported artifact file exists."""
    assert os.path.exists(ARTIFACT_PATH), f"Model artifact missing at {ARTIFACT_PATH}"


def test_artifact_loads_and_schema():
    """Verify artifact loads and contains all required keys and correct types."""
    artifact = joblib.load(ARTIFACT_PATH)
    assert isinstance(artifact, dict)

    required_keys = [
        "model", "imputer", "feature_cols", "threshold", "pos_weight",
        "model_family", "target_pct", "stop_loss_pct", "cost_pct",
        "split_metadata", "train_rows", "training_timestamp", "calibrator"
    ]
    for k in required_keys:
        assert k in artifact, f"Missing key '{k}' in artifact"


def test_model_is_lightgbm_not_synthetic():
    """Verify model is genuine LightGBM and not a synthetic RandomForest."""
    artifact = joblib.load(ARTIFACT_PATH)
    clf = artifact["model"]
    assert isinstance(clf, TradeSignalClassifier)
    assert clf.model_family == "lightgbm"
    assert isinstance(clf.model, lgb.LGBMClassifier)
    assert clf.model.scale_pos_weight == 25.0
    assert clf.model.random_state == 42
    assert artifact["model_family"] == "lightgbm"


def test_feature_order_exact():
    """Verify the 14 features match Phase 5 canonical sequence exactly."""
    artifact = joblib.load(ARTIFACT_PATH)
    assert artifact["feature_cols"] == EXPECTED_FEATURE_COLS
    assert len(artifact["feature_cols"]) == 14


def test_threshold_exact():
    """Verify threshold is exactly 0.8000."""
    artifact = joblib.load(ARTIFACT_PATH)
    assert artifact["threshold"] == 0.8000


def test_preprocessing_imputer_present():
    """Verify SimpleImputer is fitted and present."""
    artifact = joblib.load(ARTIFACT_PATH)
    imputer = artifact["imputer"]
    assert isinstance(imputer, SimpleImputer)
    assert hasattr(imputer, "statistics_")
    # Median sentiment score value fitted on Phase 5 training data
    assert np.isfinite(imputer.statistics_[0])


def test_artifact_known_row_parity():
    """Verify exported artifact reproduces exact research probabilities on known frozen rows."""
    if not os.path.exists(PHASE5_FEATURES_PATH):
        pytest.skip("Phase 5 features parquet not present.")

    artifact = joblib.load(ARTIFACT_PATH)
    clf = artifact["model"]
    imputer = artifact["imputer"]

    df_all = pd.read_parquet(PHASE5_FEATURES_PATH)
    if "mean_sentiment" in df_all.columns and "sentiment_score" not in df_all.columns:
        df_all["sentiment_score"] = df_all["mean_sentiment"]

    evaluator = Phase5Evaluator(cost_pct=0.0005, horizon_minutes=240)
    _, _, df_test, _ = evaluator.split_dataset_chronologically(df_all)
    test_valid = df_test[df_test["label_status"] == "VALID"].copy()

    # Pick 5 fixed deterministic rows across different symbols and directions
    fixed_indices = [0, 10, 100, 500, 1000]
    sample = test_valid.iloc[fixed_indices].copy()
    sample["sentiment_score"] = imputer.transform(sample[["sentiment_score"]]).ravel()

    probs = clf.predict_proba(sample)

    # Known frozen probabilities for these exact rows from deterministic LightGBM
    expected_probs = [0.05497710, 0.00815247, 0.58776196, 0.45764691, 0.12473692]

    for i, exp_p in enumerate(expected_probs):
        actual_p = probs[i]
        assert abs(actual_p - exp_p) < 1e-6, (
            f"Parity failure on row index {fixed_indices[i]}: expected {exp_p:.6f}, got {actual_p:.6f}"
        )
