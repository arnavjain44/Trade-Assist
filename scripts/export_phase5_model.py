"""
Phase 5 D-LightGBM Model Export Script

Trains the audited winning Phase 5 LightGBM model configuration on the frozen
Phase 5 training split and exports the model, imputer, and metadata into a single
production-ready artifact: data/models/phase5_d_lightgbm.joblib.

Guarantees:
- Deterministic training with random_state=42.
- Median imputer fitted strictly on Training data only.
- Exact 14 Phase 5 features in canonical sequence.
- Threshold P* = 0.8000 preserved.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from app.ml.models import TradeSignalClassifier
from app.ml.phase5_evaluation import Phase5Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PHASE5_FEATURES_PATH = "data/processed/phase5_features.parquet"
OUTPUT_MODEL_DIR = "data/models"
OUTPUT_MODEL_PATH = "data/models/phase5_d_lightgbm.joblib"

# The exact 14 Phase 5 features in exact sequence:
PHASE5_FEATURE_COLS = [
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

THRESHOLD = 0.8000
POS_WEIGHT = 25.0
COST_PCT = 0.0005


def export_phase5_model() -> Path:
    logger.info("Loading Phase 5 dataset from %s...", PHASE5_FEATURES_PATH)
    if not os.path.exists(PHASE5_FEATURES_PATH):
        raise FileNotFoundError(f"Phase 5 features file not found at {PHASE5_FEATURES_PATH}")

    df_all = pd.read_parquet(PHASE5_FEATURES_PATH)
    if "mean_sentiment" in df_all.columns and "sentiment_score" not in df_all.columns:
        df_all["sentiment_score"] = df_all["mean_sentiment"]

    # Chronological split with 240m purging and embargo (exact Phase 5 evaluator split)
    evaluator = Phase5Evaluator(cost_pct=COST_PCT, horizon_minutes=240)
    df_train, df_val, df_test, split_meta = evaluator.split_dataset_chronologically(df_all)

    train_valid = df_train[df_train["label_status"] == "VALID"].copy()
    val_valid = df_val[df_val["label_status"] == "VALID"].copy()
    test_valid = df_test[df_test["label_status"] == "VALID"].copy()

    logger.info("Train rows: %d, Val rows: %d, Test rows: %d", len(train_valid), len(val_valid), len(test_valid))

    # Fit SimpleImputer on Training data sentiment_score only
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_valid[["sentiment_score"]])
    logger.info("Fitted sentiment_score median imputer value: %s", imputer.statistics_[0])

    train_valid["sentiment_score"] = imputer.transform(train_valid[["sentiment_score"]]).ravel()
    val_valid["sentiment_score"] = imputer.transform(val_valid[["sentiment_score"]]).ravel()
    test_valid["sentiment_score"] = imputer.transform(test_valid[["sentiment_score"]]).ravel()

    # Train winning D-LightGBM configuration
    logger.info("Training TradeSignalClassifier(model_family='lightgbm', pos_weight=25, random_state=42)...")
    clf = TradeSignalClassifier(
        model_family="lightgbm",
        pos_weight=POS_WEIGHT,
        feature_cols=PHASE5_FEATURE_COLS,
        random_state=42,
    )
    clf.fit(train_valid, train_valid["label"].values)

    # Validate test predictions
    test_probs = clf.predict_proba(test_valid)
    qualified_trades = (test_probs >= THRESHOLD).sum()
    logger.info("Test set evaluation: %d trades qualified at P >= %.4f", qualified_trades, THRESHOLD)

    # Build artifact bundle
    artifact = {
        "model": clf,
        "imputer": imputer,
        "feature_cols": list(PHASE5_FEATURE_COLS),
        "threshold": THRESHOLD,
        "pos_weight": POS_WEIGHT,
        "model_family": "lightgbm",
        "target_pct": 0.022,
        "stop_loss_pct": 0.009,
        "cost_pct": COST_PCT,
        "split_metadata": split_meta,
        "train_rows": len(train_valid),
        "training_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "calibrator": "none",
    }

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    joblib.dump(artifact, OUTPUT_MODEL_PATH)
    logger.info("Successfully exported Phase 5 D-LightGBM model artifact to %s", OUTPUT_MODEL_PATH)
    return Path(OUTPUT_MODEL_PATH)


if __name__ == "__main__":
    export_phase5_model()
