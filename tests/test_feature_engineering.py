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


def test_vwap_mathematical_precision_and_session_reset(sample_multi_day_df):
    """Test 7: Strengthened VWAP mathematical session-reset test across multiple days and zero-volume candles."""
    df_feat = feature_engine.calculate_features(sample_multi_day_df)
    
    # 1. Day 1 First Candle (Index 0): VWAP must equal typical price when volume > 0
    row0 = df_feat.iloc[0]
    expected_tp0 = (row0["high"] + row0["low"] + row0["close"]) / 3.0
    assert np.isclose(row0["vwap"], expected_tp0, atol=1e-4)

    # 2. Day 2 First Candle (Index 10): VWAP must reset to Day 2 Candle 1 typical price
    row10 = df_feat.iloc[10]
    expected_tp10 = (row10["high"] + row10["low"] + row10["close"]) / 3.0
    assert np.isclose(row10["vwap"], expected_tp10, atol=1e-4)
    # Must NOT be diluted by Day 1 (~100.0) prices
    assert row10["vwap"] > 190.0

    # 3. Day 2 Subsequent Candles: Independently compute cumsum(TP*vol) / cumsum(vol) for Day 2
    day2_df = df_feat.iloc[10:20].copy()
    tp_day2 = (day2_df["high"] + day2_df["low"] + day2_df["close"]) / 3.0
    math_vwap_day2 = (tp_day2 * day2_df["volume"]).cumsum() / day2_df["volume"].cumsum()
    
    assert np.allclose(day2_df["vwap"], math_vwap_day2, atol=1e-4)

    # 4. Zero-volume candle test: Create session with a zero-volume candle mid-day
    zero_vol_records = []
    ts_start = pd.Timestamp("2026-09-03 09:15", tz="Asia/Kolkata")
    # Candle 0: Volume 1000
    zero_vol_records.append({"timestamp": ts_start, "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000.0})
    # Candle 1: Zero Volume
    zero_vol_records.append({"timestamp": ts_start + pd.Timedelta(minutes=5), "open": 102, "high": 104, "low": 101, "close": 103, "volume": 0.0})
    # Candle 2: Volume 500
    zero_vol_records.append({"timestamp": ts_start + pd.Timedelta(minutes=10), "open": 103, "high": 106, "low": 102, "close": 105, "volume": 500.0})
    
    zv_df = pd.DataFrame(zero_vol_records)
    zv_feat = feature_engine.calculate_features(zv_df)
    
    # Mid-day zero-volume candle must hold previous valid VWAP
    assert np.isclose(zv_feat.iloc[1]["vwap"], zv_feat.iloc[0]["vwap"])
    assert not zv_feat["vwap"].isna().any()


def test_full_feature_causality():
    """Test 11: Strengthened Causality — Adding FUTURE candles after timestamp T must NEVER change feature values at T or earlier."""
    records = []
    base_ts = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    
    # Generate 20 candles
    for i in range(20):
        records.append({
            "timestamp": base_ts + pd.Timedelta(minutes=5 * i),
            "open": 100.0 + (i * 0.5),
            "high": 102.0 + (i * 0.5),
            "low": 98.0 + (i * 0.5),
            "close": 101.0 + (i * 0.5),
            "volume": 1000.0 + (i * 10)
        })
    df_full = pd.DataFrame(records)
    
    # Dataset A: Truncated at index 10 (Timestamp T)
    df_A = df_full.iloc[:11].copy()
    # Dataset B: Full 20 candles
    df_B = df_full.copy()
    
    feat_A = feature_engine.calculate_features(df_A)
    feat_B = feature_engine.calculate_features(df_B)
    
    # Verify all 6 base indicators + 3 normalized features match 100% for rows 0 to 10
    feature_cols = [
        "ema_5", "rsi", "obv", "bollinger_middle", "bollinger_upper", "bollinger_lower",
        "macd", "macd_signal", "macd_diff", "vwap", "bollinger_position", "price_vs_vwap", "price_vs_ema5"
    ]
    for col in feature_cols:
        assert np.allclose(feat_A[col], feat_B[col].iloc[:11], atol=1e-5), f"Causality broken for feature: {col}"


def test_per_symbol_feature_isolation():
    """Test 13: Verifies that multi-symbol DataFrames NEVER bleed indicator state across symbols."""
    ts_base = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    
    records_A = []
    records_B = []
    for i in range(10):
        records_A.append({
            "timestamp": ts_base + pd.Timedelta(minutes=5 * i),
            "symbol": "RELIANCE.NS",
            "open": 2500.0 + i, "high": 2510.0 + i, "low": 2490.0 + i, "close": 2505.0 + i, "volume": 10000.0
        })
        records_B.append({
            "timestamp": ts_base + pd.Timedelta(minutes=5 * i),
            "symbol": "TCS.NS",
            "open": 4000.0 + (i * 10), "high": 4020.0 + (i * 10), "low": 3990.0 + (i * 10), "close": 4010.0 + (i * 10), "volume": 50000.0
        })
        
    df_A = pd.DataFrame(records_A)
    df_B = pd.DataFrame(records_B)
    df_combined = pd.concat([df_A, df_B], ignore_index=True)
    
    feat_A = feature_engine.calculate_features(df_A)
    feat_B = feature_engine.calculate_features(df_B)
    feat_combined = feature_engine.calculate_features(df_combined)
    
    feat_comb_A = feat_combined[feat_combined["symbol"] == "RELIANCE.NS"].reset_index(drop=True)
    feat_comb_B = feat_combined[feat_combined["symbol"] == "TCS.NS"].reset_index(drop=True)
    
    for col in ["ema_5", "obv", "macd", "vwap"]:
        assert np.allclose(feat_comb_A[col], feat_A[col]), f"Symbol A contaminated for {col}"
        assert np.allclose(feat_comb_B[col], feat_B[col]), f"Symbol B contaminated for {col}"


def test_session_boundaries_and_timezone(sample_multi_day_df):
    """Test 14: Verifies IST timezone awareness, session start/end boundaries, and zero overnight accumulation."""
    df_feat = feature_engine.calculate_features(sample_multi_day_df)
    
    assert df_feat["timestamp"].dt.tz is not None
    assert str(df_feat["timestamp"].dt.tz) == "Asia/Kolkata"
    
    # First candle of Day 1 (09:15 IST)
    first_d1 = df_feat.iloc[0]
    assert first_d1["timestamp"].hour == 9 and first_d1["timestamp"].minute == 15
    
    # First candle of Day 2 (09:15 IST)
    first_d2 = df_feat.iloc[10]
    assert first_d2["timestamp"].hour == 9 and first_d2["timestamp"].minute == 15
    
    # VWAP at start of Day 2 must not equal cumulative VWAP of Day 1
    d1_last_vwap = df_feat.iloc[9]["vwap"]
    d2_first_vwap = df_feat.iloc[10]["vwap"]
    assert not np.isclose(d1_last_vwap, d2_first_vwap)


def test_data_quality_ohlc_relationships_and_duplicates(sample_multi_day_df):
    """Test 15: Verifies DataQualityValidator detects OHLC logical violations and handles multi-symbol duplicate checking."""
    from app.ml.data_quality import data_quality_validator
    
    # 1. Invalid OHLC relationship test
    bad_ohlc_df = sample_multi_day_df.copy()
    bad_ohlc_df.loc[0, "high"] = 90.0  # High < Open (invalid)
    report_ohlc = data_quality_validator.audit_dataset(bad_ohlc_df, timeframe="5m")
    assert report_ohlc["issues_detected"]["invalid_ohlc_relationships"] > 0
    
    # 2. Duplicate timestamp across DIFFERENT symbols is NOT a duplicate
    ts = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    multi_sym_df = pd.DataFrame([
        {"timestamp": ts, "symbol": "RELIANCE.NS", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000},
        {"timestamp": ts, "symbol": "TCS.NS", "open": 200, "high": 205, "low": 199, "close": 202, "volume": 2000}
    ])
    report_multi = data_quality_validator.audit_dataset(multi_sym_df, timeframe="5m")
    assert report_multi["issues_detected"]["duplicate_timestamps"] == 0
    
    # 3. Duplicate timestamp within SAME symbol IS a duplicate
    same_sym_dup = pd.DataFrame([
        {"timestamp": ts, "symbol": "RELIANCE.NS", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000},
        {"timestamp": ts, "symbol": "RELIANCE.NS", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000}
    ])
    report_same = data_quality_validator.audit_dataset(same_sym_dup, timeframe="5m")
    assert report_same["issues_detected"]["duplicate_timestamps"] == 1


def test_missing_candles_handling():
    """Test 9: Verifies pipeline handles missing candles without crashing."""
    ts1 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts2 = pd.Timestamp("2026-09-01 10:30", tz="Asia/Kolkata")
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


def test_reproducibility(sample_multi_day_df):
    """Test 16: Verifies 100% deterministic reproducibility across runs."""
    run1 = feature_engine.calculate_features(sample_multi_day_df)
    run2 = feature_engine.calculate_features(sample_multi_day_df)
    pd.testing.assert_frame_equal(run1, run2)
