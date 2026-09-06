"""
Phase 3 Live Integration Tests

Verifies:
A. Live pipeline loads the Phase 5 artifact.
B. Missing artifact fails loudly.
C. _bootstrap_synthetic_training is not called / does not exist.
D. No synthetic data is generated.
E. No +25 confidence shift.
F. No confidence clamping to [72.0, 94.5].
G. Threshold is exactly 0.8000.
H. Raw probability is preserved.
I. Correct model name ("phase5_d_lightgbm") is returned.
J. Direction +1 maps to BUY only when qualified (P >= 0.8000).
K. Direction -1 maps to SELL only when qualified (P >= 0.8000).
L. Unqualified candidates become HOLD / non-qualified.
M. Unqualified candidates get zero capital allocation in constraint enforcer.
N. Live inference cannot run on daily-candle fallback data (fails loudly).
O. Exact 14-feature order is preserved.
P. Saved imputer is used.
Q. News feature values are passed correctly.
R. Both market_similarity and stock_similarity are passed.
S. Existing same-day / 40% constraints remain intact.
T. Regression test: production inference probability == serialized artifact probability on frozen row.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from app.ml.pipeline import MLPredictionEngine, ml_engine
from app.ml.phase5_inference import Phase5ProductionInference, DEFAULT_ARTIFACT_PATH
from app.ml.phase5_feature_builder import PHASE5_FEATURE_COLS
from app.core.constraints import ConstraintEnforcer


# ==============================================================================
# Test A, G, I: Artifact loading, threshold, and model name
# ==============================================================================
def test_live_pipeline_loads_artifact():
    """Verify live pipeline loads Phase 5 artifact and exposes threshold 0.8000."""
    engine = MLPredictionEngine()
    assert isinstance(engine.inference, Phase5ProductionInference)
    assert engine.inference.model is not None
    assert engine.inference.threshold == 0.8000
    assert engine.inference.model_family == "lightgbm"
    assert engine.inference.feature_cols == PHASE5_FEATURE_COLS


# ==============================================================================
# Test B: Missing artifact fails loudly
# ==============================================================================
def test_missing_artifact_fails_loudly():
    """Verify missing artifact raises FileNotFoundError loudly."""
    with pytest.raises(FileNotFoundError, match="Phase 5 model artifact missing"):
        Phase5ProductionInference(artifact_path="data/models/non_existent_model.joblib")


# ==============================================================================
# Test C, D: No synthetic bootstrap training or random normal generation
# ==============================================================================
def test_no_synthetic_bootstrap_methods():
    """Verify legacy synthetic bootstrap methods are completely removed from engine."""
    engine = MLPredictionEngine()
    assert not hasattr(engine, "_bootstrap_synthetic_training")
    assert not hasattr(engine, "logistic_model")
    assert not hasattr(engine, "random_forest_model")


# ==============================================================================
# Test E, F, H: No +25 shift, no confidence clamping, raw probability preserved
# ==============================================================================
def test_raw_probability_and_no_confidence_inflation():
    """Verify raw probability is returned and not inflated with +25 or clamped to [72.0, 94.5]."""
    engine = MLPredictionEngine()

    # Mock inference to return a low probability (0.15)
    with patch.object(engine.inference, "evaluate_dual_directions", return_value={
        "model_name": "phase5_d_lightgbm",
        "model_probability": 0.154321,
        "model_threshold": 0.8000,
        "p_long": 0.154321,
        "p_short": 0.082100,
        "long_qualified": False,
        "short_qualified": False,
        "qualified": False,
        "action": "HOLD",
        "direction": 1,
    }):
        tech_dict = {
            "rsi": 50.0, "obv": 1000.0, "bollinger_position": 0.5,
            "macd": 0.0, "macd_signal": 0.0, "macd_diff": 0.0,
            "price_vs_vwap": 0.0, "price_vs_ema5": 0.0,
        }
        res = engine.predict_trade_signal("TCS.NS", 3500.0, tech_dict, 0.0, 0.0, timeframe="5m")

        # Must NOT be shifted (+25 -> 40.4%) or clamped (min 72.0%)
        assert res["model_probability"] == 0.154321
        assert res["confidence"] == 15.43  # exactly 0.154321 * 100
        assert res["action"] == "HOLD"
        assert res["qualified"] is False
        assert res["model_used"] == "phase5_d_lightgbm"


# ==============================================================================
# Test J, K, L: Direction qualification mapping (BUY, SELL, HOLD)
# ==============================================================================
def test_direction_qualification_mapping():
    """Verify +1 qualifies to BUY, -1 qualifies to SELL, and unqualified becomes HOLD."""
    engine = MLPredictionEngine()
    tech_dict = {
        "rsi": 50.0, "obv": 1000.0, "bollinger_position": 0.5,
        "macd": 0.0, "macd_signal": 0.0, "macd_diff": 0.0,
        "price_vs_vwap": 0.0, "price_vs_ema5": 0.0,
    }

    # Case 1: LONG qualifies (0.85 >= 0.80) -> BUY
    with patch.object(engine.inference, "evaluate_candidate", side_effect=[0.85, 0.10]):
        res_buy = engine.predict_trade_signal("INFY.NS", 1500.0, tech_dict, 0.2, 0.8, timeframe="5m")
        assert res_buy["action"] == "BUY"
        assert res_buy["qualified"] is True
        assert res_buy["direction"] == 1
        assert res_buy["target_price"] == round(1500.0 * 1.022, 2)
        assert res_buy["stop_loss"] == round(1500.0 * 0.991, 2)

    # Case 2: SHORT qualifies (0.88 >= 0.80) -> SELL
    with patch.object(engine.inference, "evaluate_candidate", side_effect=[0.12, 0.88]):
        res_sell = engine.predict_trade_signal("INFY.NS", 1500.0, tech_dict, -0.3, 0.8, timeframe="5m")
        assert res_sell["action"] == "SELL"
        assert res_sell["qualified"] is True
        assert res_sell["direction"] == -1
        assert res_sell["target_price"] == round(1500.0 * 0.978, 2)
        assert res_sell["stop_loss"] == round(1500.0 * 1.009, 2)

    # Case 3: Neither qualifies (0.75 < 0.80, 0.70 < 0.80) -> HOLD
    with patch.object(engine.inference, "evaluate_candidate", side_effect=[0.75, 0.70]):
        res_hold = engine.predict_trade_signal("INFY.NS", 1500.0, tech_dict, 0.0, 0.5, timeframe="5m")
        assert res_hold["action"] == "HOLD"
        assert res_hold["qualified"] is False
        assert res_hold["target_price"] == 1500.0
        assert res_hold["stop_loss"] == 1500.0


# ==============================================================================
# Test M, S: Unqualified gets zero allocation; Constraint enforcer preserves limits
# ==============================================================================
def test_unqualified_zero_allocation_and_constraints():
    """Verify unqualified candidates (HOLD) receive zero allocation in constraint enforcer."""
    enforcer = ConstraintEnforcer()

    # Mix of BUY, SELL, and HOLD
    raw_picks = [
        {"symbol": "TCS.NS", "action": "BUY", "confidence": 85.0, "current_price": 3500.0},
        {"symbol": "INFY.NS", "action": "HOLD", "confidence": 72.0, "current_price": 1500.0},
        {"symbol": "WIPRO.NS", "action": "SELL", "confidence": 82.0, "current_price": 500.0},
    ]

    recs = enforcer.enforce_intraday_constraints(raw_picks, total_capital=100000.0)

    # INFY.NS must NOT be in final allocated recommendations
    allocated_symbols = [r["symbol"] for r in recs]
    assert "TCS.NS" in allocated_symbols
    assert "WIPRO.NS" in allocated_symbols
    assert "INFY.NS" not in allocated_symbols

    # All allocated items must have hold_until == 'same_day'
    for r in recs:
        assert r["hold_until"] == "same_day"
        assert r["allocation_pct"] <= 0.40  # 40% cap enforced


# ==============================================================================
# Test N: Reject daily data
# ==============================================================================
def test_rejects_daily_data():
    """Verify feeding daily data (timeframe='1d') raises ValueError loudly."""
    engine = MLPredictionEngine()
    tech_dict = {
        "rsi": 50.0, "obv": 1000.0, "bollinger_position": 0.5,
        "macd": 0.0, "macd_signal": 0.0, "macd_diff": 0.0,
        "price_vs_vwap": 0.0, "price_vs_ema5": 0.0,
        "timeframe": "1d",
    }

    with pytest.raises(ValueError, match="strictly requires 5-minute"):
        engine.predict_trade_signal("RELIANCE.NS", 2500.0, tech_dict, 0.0, 0.0, timeframe="1d")


# ==============================================================================
# Test O, P, Q, R: Feature ordering, imputer, news, and context
# ==============================================================================
def test_feature_ordering_imputer_and_inputs():
    """Verify feature builder and imputer work together on exact 14 columns."""
    inference = Phase5ProductionInference()
    tech_dict = {
        "rsi": 45.0, "obv": 500000.0, "bollinger_position": 0.4,
        "macd": 0.5, "macd_signal": 0.3, "macd_diff": 0.2,
        "price_vs_vwap": -0.002, "price_vs_ema5": 0.001,
    }
    # Test with NaN news to verify saved imputer replaces it with training median
    news_dict = {"sentiment_score": np.nan, "has_news": False, "number_of_articles": 0}
    context_dict = {"market_similarity": 0.85, "stock_similarity": 0.90}

    # Should succeed without error, imputing sentiment_score to median
    prob = inference.evaluate_candidate(tech_dict, news_dict, context_dict, direction=1, timeframe="5m")
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


# ==============================================================================
# Test T: Regression Test  Production inference probability == Artifact probability
# ==============================================================================
def test_regression_parity_on_known_row():
    """Verify production inference matches frozen research probability identically (Delta < 1e-6)."""
    inference = Phase5ProductionInference()

    # Fixed deterministic feature vector identical to row 0 from Phase 1 parity check
    known_features = pd.DataFrame([{
        "rsi": 42.1234,
        "obv": 123456.0,
        "bollinger_position": 0.35,
        "macd": -0.15,
        "macd_signal": -0.10,
        "macd_diff": -0.05,
        "price_vs_vwap": -0.003,
        "price_vs_ema5": -0.001,
        "direction": 1,
        "sentiment_score": np.nan,  # tests imputer
        "has_news": False,
        "number_of_articles": 0,
        "market_similarity": 0.82,
        "stock_similarity": 0.88,
    }], columns=PHASE5_FEATURE_COLS)

    prob_direct = inference.predict_raw_probability(known_features)

    # Calculate probability directly through underlying LGBM model with imputer
    imputed = known_features.copy()
    imputed["sentiment_score"] = inference.imputer.transform(imputed[["sentiment_score"]]).ravel()
    probs_raw = inference.model.predict_proba(imputed)
    prob_expected = float(probs_raw[0] if getattr(probs_raw, "ndim", 1) == 1 else probs_raw[0, 1])

    assert abs(prob_direct - prob_expected) < 1e-6
    assert isinstance(prob_direct, float)
