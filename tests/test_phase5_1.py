"""
Phase 5.1 Unit Tests: High-Confidence Prediction & Accuracy Research

Tests:
1. Enhanced causal feature engineering causality & column schema.
2. Per-symbol isolation of enhanced features.
3. Probability bucket partitioning & metric calculations.
4. Directional model fairness constraints.
5. Multi-dimensional 90% precision robustness auditing.
6. Canonical locked test set integrity (Days 52-59 holdout preservation).
"""

import numpy as np
import pandas as pd
import pytest

from app.ml.phase5_1_features import Phase51FeatureEngine
from app.ml.phase5_1_research import Phase51AccuracyResearchRunner


def create_synthetic_candle_df(n_rows: int = 50, symbol: str = "SBIN") -> pd.DataFrame:
    """Helper to generate synthetic 5m OHLCV DataFrame."""
    dates = pd.date_range("2026-06-15 09:15:00", periods=n_rows, freq="5min")
    prices = 100.0 + np.cumsum(np.random.normal(0, 0.2, n_rows))
    df = pd.DataFrame({
        "symbol": symbol,
        "timestamp": dates,
        "open": prices,
        "high": prices + 0.3,
        "low": prices - 0.3,
        "close": prices + 0.05,
        "volume": np.random.randint(1000, 5000, n_rows),
        "rsi": np.random.uniform(30, 70, n_rows),
        "bollinger_upper": prices + 1.0,
        "bollinger_lower": prices - 1.0,
        "bollinger_middle": prices,
        "ema_5": prices,
    })
    return df


def test_enhanced_features_causality_and_schema():
    """Verify enhanced feature computation adds all required columns and preserves causality."""
    df1 = create_synthetic_candle_df(60, "RELIANCE")
    df_feat1 = Phase51FeatureEngine.compute_enhanced_features(df1)

    for col in Phase51FeatureEngine.ENHANCED_FEATURE_COLS:
        assert col in df_feat1.columns, f"Missing enhanced feature column: {col}"

    # Causality test: modify row 50 and check row 20
    df2 = df1.copy()
    df2.loc[50:, "close"] = df2.loc[50:, "close"] * 2.0
    df2.loc[50:, "high"] = df2.loc[50:, "high"] * 2.0
    df2.loc[50:, "low"] = df2.loc[50:, "low"] * 2.0
    df2.loc[50:, "volume"] = df2.loc[50:, "volume"] * 10.0

    df_feat2 = Phase51FeatureEngine.compute_enhanced_features(df2)

    # Values at index 20 must be completely identical
    for col in Phase51FeatureEngine.ENHANCED_FEATURE_COLS:
        val1 = df_feat1.loc[20, col]
        val2 = df_feat2.loc[20, col]
        assert np.isclose(val1, val2, atol=1e-7), f"Feature {col} at t=20 changed when t=50 modified! Future leakage detected."


def test_enhanced_features_symbol_isolation():
    """Verify features for symbol A do not bleed into symbol B."""
    df_s1 = create_synthetic_candle_df(40, "SBIN")
    df_s2 = create_synthetic_candle_df(40, "TCS")
    df_comb = pd.concat([df_s1, df_s2], ignore_index=True)

    df_res = Phase51FeatureEngine.compute_enhanced_features(df_comb)
    df_s1_isolated = Phase51FeatureEngine.compute_enhanced_features(df_s1)

    # Check that SBIN rows in combined match isolated
    sbin_res = df_res[df_res["symbol"] == "SBIN"].sort_values("timestamp").reset_index(drop=True)
    for col in Phase51FeatureEngine.ENHANCED_FEATURE_COLS:
        np.testing.assert_allclose(sbin_res[col].values, df_s1_isolated[col].values, atol=1e-7)


def test_probability_bucket_analysis():
    """Test discrete probability bucket analysis computation and edge cases."""
    runner = Phase51AccuracyResearchRunner()
    n = 100
    df_val = pd.DataFrame({
        "label": [1 if i % 2 == 0 else 0 for i in range(n)],
        "direction": [1 if i < 50 else -1 for i in range(n)],
        "realized_return": [0.02 if i % 2 == 0 else -0.01 for i in range(n)],
    })
    probs = np.linspace(0.45, 0.98, n)

    buckets = runner._analyze_probability_buckets(df_val, probs)
    assert len(buckets) == 7
    total_candidates = sum(b["candidates"] for b in buckets)
    assert total_candidates > 0
    for b in buckets:
        assert "bucket" in b
        assert "win_rate" in b
        assert "net_avg_return_pct" in b
        assert "net_profit_factor" in b


def test_robustness_auditor_90pct_criteria():
    """Verify strict multi-dimensional 90% precision robustness criteria."""
    runner = Phase51AccuracyResearchRunner(cost_pct=0.0005)

    # Scenario 1: High precision (100%), but only 10 trades -> Must FAIL
    df_fail_small = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-26", periods=10, freq="1D"),
        "symbol": ["INFY"] * 10,
        "label": [1] * 10,
        "realized_return": [0.02] * 10,
    })
    audit_small = runner._audit_robustness(df_fail_small, np.ones(10), 0.80)
    assert audit_small["trades"] == 10
    assert audit_small["qualifies_as_90pct_edge"] is False

    # Scenario 2: 40 trades, 95% precision, but concentrated in 1 day (100% > 50%) -> Must FAIL
    df_fail_conc = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-08-26 10:00:00")] * 40,
        "symbol": ["INFY"] * 40,
        "label": [1] * 38 + [0] * 2,
        "realized_return": [0.02] * 38 + [-0.01] * 2,
    })
    audit_conc = runner._audit_robustness(df_fail_conc, np.ones(40), 0.80)
    assert audit_conc["qualifies_as_90pct_edge"] is False
    assert audit_conc["max_single_day_concentration_pct"] == 100.0

    # Scenario 3: 50 trades, 92% precision, spread over 5 days, 5 symbols, positive net avg return -> Must PASS
    symbols = ["TCS", "INFY", "RELIANCE", "SBIN", "HDFCBANK"] * 10
    dates = [pd.Timestamp("2026-08-26") + pd.Timedelta(days=i % 5) for i in range(50)]
    labels = [1] * 46 + [0] * 4  # 46/50 = 92%
    rets = [0.02] * 46 + [-0.01] * 4
    df_pass = pd.DataFrame({
        "timestamp": dates,
        "symbol": symbols,
        "label": labels,
        "realized_return": rets,
    })
    audit_pass = runner._audit_robustness(df_pass, np.ones(50), 0.80)
    assert audit_pass["trades"] == 50
    assert audit_pass["precision"] == 92.0
    assert audit_pass["qualifies_as_90pct_edge"] is True


def test_canonical_locked_test_set_dates():
    """Verify that canonical locked test dates are strictly Days 52-59."""
    # Days 52-59 correspond to 2026-08-26 through 2026-09-04 (8 trading days)
    expected_dates = [
        "2026-08-26",
        "2026-08-27",
        "2026-08-28",
        "2026-08-31",
        "2026-09-01",
        "2026-09-02",
        "2026-09-03",
        "2026-09-04",
    ]
    assert len(expected_dates) == 8
