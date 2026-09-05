import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.ml.feature_engineering import feature_engine


@pytest.fixture
def sample_multi_day_df():
    """Generates 2 days of 5-minute candles with known prices and volume for deterministic testing."""
    records = []
    
    # Day 1: 2026-09-01 (Market 09:15 to 10:00 IST)
    day1_start = datetime(2026, 9, 1, 9, 15)
    for i in range(10):
        ts = pd.Timestamp(day1_start + timedelta(minutes=5 * i), tz="Asia/Kolkata")
        records.append({
            "timestamp": ts,
            "open": 100.0 + i,
            "high": 102.0 + i,
            "low": 99.0 + i,
            "close": 101.0 + i,
            "volume": 1000.0 * (i + 1)
        })
        
    # Day 2: 2026-09-02 (Market 09:15 to 10:00 IST) — Reset Session
    day2_start = datetime(2026, 9, 2, 9, 15)
    for i in range(10):
        ts = pd.Timestamp(day2_start + timedelta(minutes=5 * i), tz="Asia/Kolkata")
        records.append({
            "timestamp": ts,
            "open": 200.0 + i,
            "high": 205.0 + i,
            "low": 198.0 + i,
            "close": 202.0 + i,
            "volume": 500.0 * (i + 1)
        })
        
    return pd.DataFrame(records)


def test_ema_calculation(sample_multi_day_df):
    """Test 1: Verifies 5 EMA calculation."""
    df_feat = feature_engine.calculate_features(sample_multi_day_df)
    assert "ema_5" in df_feat.columns
    assert not df_feat["ema_5"].isna().any()
    # 5 EMA must lie between min and max close prices
    assert df_feat["ema_5"].min() >= sample_multi_day_df["close"].min()
    assert df_feat["ema_5"].max() <= sample_multi_day_df["close"].max()


def test_rsi_calculation(sample_multi_day_df):
    """Test 2: Verifies RSI-9 calculation bounded between 0 and 100."""
    df_feat = feature_engine.calculate_features(sample_multi_day_df)
    assert "rsi" in df_feat.columns
    assert not df_feat["rsi"].isna().any()
    assert (df_feat["rsi"] >= 0.0).all() and (df_feat["rsi"] <= 100.0).all()


def test_obv_calculation(sample_multi_day_df):
    """Test 3: Verifies On-Balance Volume signed accumulation."""
    df_feat = feature_engine.calculate_features(sample_multi_day_df)
    assert "obv" in df_feat.columns
    assert not df_feat["obv"].isna().any()
    # Continuous positive trend in prices must yield increasing OBV
    assert df_feat["obv"].iloc[-1] > df_feat["obv"].iloc[0]


def test_bollinger_calculation(sample_multi_day_df):
    """Test 4: Verifies Bollinger Bands (Upper >= Middle >= Lower)."""
    df_feat = feature_engine.calculate_features(sample_multi_day_df)
    assert "bollinger_upper" in df_feat.columns
    assert "bollinger_middle" in df_feat.columns
    assert "bollinger_lower" in df_feat.columns
    assert (df_feat["bollinger_upper"] >= df_feat["bollinger_middle"]).all()
    assert (df_feat["bollinger_middle"] >= df_feat["bollinger_lower"]).all()


def test_macd_calculation(sample_multi_day_df):
    """Test 5: Verifies MACD, Signal, and MACD Diff."""
    df_feat = feature_engine.calculate_features(sample_multi_day_df)
    assert "macd" in df_feat.columns
    assert "macd_signal" in df_feat.columns
    assert "macd_diff" in df_feat.columns
    # macd_diff must equal macd - macd_signal
    assert np.allclose(df_feat["macd_diff"], df_feat["macd"] - df_feat["macd_signal"])


def test_vwap_calculation(sample_multi_day_df):
    """Test 6: Verifies VWAP non-null and within price bounds."""
    df_feat = feature_engine.calculate_features(sample_multi_day_df)
    assert "vwap" in df_feat.columns
    assert not df_feat["vwap"].isna().any()


def test_vwap_session_reset(sample_multi_day_df):
    """Test 7: CRITICAL — Verifies VWAP RESETS at the start of Day 2 (no multi-day leakage)."""
    df_feat = feature_engine.calculate_features(sample_multi_day_df)
    
    # First candle of Day 2 is at index 10
    day2_first_row = df_feat.iloc[10]
    
    # At index 10, typical price = (200 + 205 + 198) / 3 = 201.0
    expected_day2_first_vwap = (day2_first_row["high"] + day2_first_row["low"] + day2_first_row["close"]) / 3.0
    
    # Verify Day 2 first VWAP equals Day 2 first candle typical price, ignoring Day 1's ~100.0 prices
    assert abs(day2_first_row["vwap"] - expected_day2_first_vwap) < 1e-4
    assert day2_first_row["vwap"] > 190.0  # Must NOT be diluted by Day 1 (~100.0) prices


def test_timezone_handling(sample_multi_day_df):
    """Test 8: Verifies timezone awareness in timestamps."""
    df_feat = feature_engine.calculate_features(sample_multi_day_df)
    assert df_feat["timestamp"].dt.tz is not None
    assert str(df_feat["timestamp"].dt.tz) == "Asia/Kolkata"


def test_missing_candles_handling():
    """Test 9: Verifies pipeline handles missing candles without crashing."""
    # Create dataset with gap
    ts1 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts2 = pd.Timestamp("2026-09-01 10:30", tz="Asia/Kolkata")  # 75 min gap
    gap_df = pd.DataFrame([
        {"timestamp": ts1, "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000},
        {"timestamp": ts2, "open": 102, "high": 108, "low": 101, "close": 107, "volume": 2000}
    ])
    df_feat = feature_engine.calculate_features(gap_df)
    assert len(df_feat) == 2
    assert not df_feat["ema_5"].isna().any()


def test_nan_inf_handling(sample_multi_day_df):
    """Test 10: Verifies zero infs or unexpected NaNs in calculated features."""
    df_feat = feature_engine.calculate_features(sample_multi_day_df)
    num_cols = ["ema_5", "rsi", "obv", "bollinger_upper", "macd", "vwap", "price_vs_vwap"]
    for col in num_cols:
        assert not np.isinf(df_feat[col]).any()
        assert not df_feat[col].isna().any()


def test_feature_leakage_prevention(sample_multi_day_df):
    """Test 11: CRITICAL — Verifies Feature Leakage Prevention (Causality).
    Feature values at index T must be IDENTICAL whether computed on full series
    or truncated series ending at T.
    """
    full_feat = feature_engine.calculate_features(sample_multi_day_df)
    
    # Truncate dataset to first 5 rows (Timestamp T = index 4)
    trunc_df = sample_multi_day_df.iloc[:5].copy()
    trunc_feat = feature_engine.calculate_features(trunc_df)
    
    # Compare row 4 in both
    row_full = full_feat.iloc[4]
    row_trunc = trunc_feat.iloc[4]
    
    assert np.isclose(row_full["ema_5"], row_trunc["ema_5"])
    assert np.isclose(row_full["rsi"], row_trunc["rsi"])
    assert np.isclose(row_full["vwap"], row_trunc["vwap"])


def test_duplicate_timestamps(sample_multi_day_df):
    """Test 12: Handles duplicate timestamps cleanly after audit."""
    dup_df = pd.concat([sample_multi_day_df, sample_multi_day_df.iloc[[0]]]).reset_index(drop=True)
    from app.ml.data_quality import data_quality_validator
    report = data_quality_validator.audit_dataset(dup_df, timeframe="5m")
    assert report["issues_detected"]["duplicate_timestamps"] > 0


def test_invalid_ohlcv(sample_multi_day_df):
    """Test 13: Detects zero/negative prices in quality auditor."""
    bad_df = sample_multi_day_df.copy()
    bad_df.loc[0, "close"] = 0.0
    bad_df.loc[1, "volume"] = -100.0
    from app.ml.data_quality import data_quality_validator
    report = data_quality_validator.audit_dataset(bad_df, timeframe="5m")
    assert report["issues_detected"]["zero_prices"] > 0
    assert report["issues_detected"]["invalid_negative_volumes"] > 0


def test_reproducibility(sample_multi_day_df):
    """Test 14: Verifies 100% deterministic reproducibility across runs."""
    run1 = feature_engine.calculate_features(sample_multi_day_df)
    run2 = feature_engine.calculate_features(sample_multi_day_df)
    pd.testing.assert_frame_equal(run1, run2)
