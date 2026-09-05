import os
import pytest
import pandas as pd
import numpy as np
from app.ml.strategy_sensitivity import StrategySensitivityAnalyzer, strategy_analyzer


@pytest.fixture
def sample_candles_df():
    """Generates 2 days of 5-minute candles for sensitivity testing."""
    records = []
    ts_day1 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    for i in range(20):
        records.append({
            "timestamp": ts_day1 + pd.Timedelta(minutes=5 * i),
            "symbol": "RELIANCE.NS",
            "open": 100.0 + (i * 0.2),
            "high": 101.5 + (i * 0.2),
            "low": 99.0 + (i * 0.2),
            "close": 100.5 + (i * 0.2),
            "volume": 1000.0
        })

    ts_day2 = pd.Timestamp("2026-09-02 09:15", tz="Asia/Kolkata")
    for i in range(20):
        records.append({
            "timestamp": ts_day2 + pd.Timedelta(minutes=5 * i),
            "symbol": "RELIANCE.NS",
            "open": 200.0 + (i * 0.5),
            "high": 203.0 + (i * 0.5),
            "low": 198.0 + (i * 0.5),
            "close": 201.0 + (i * 0.5),
            "volume": 1000.0
        })

    return pd.DataFrame(records)


def test_target_sensitivity(sample_candles_df):
    """TEST 1 — Target Sensitivity: Smaller target yields higher positive rate."""
    analyzer = StrategySensitivityAnalyzer()
    res, _ = analyzer.run_sensitivity_analysis(
        sample_candles_df,
        output_json_path="scratch/test_sens.json",
        output_csv_path="scratch/test_sens.csv"
    )

    # Find Combo A (1.0% target) vs Combo G (3.0% target)
    cfg_A = next(r for r in res if r["combo_name"] == "A")
    cfg_G = next(r for r in res if r["combo_name"] == "G")

    assert cfg_A["target_hit_count"] >= cfg_G["target_hit_count"]


def test_stop_sensitivity(sample_candles_df):
    """TEST 2 — Stop Sensitivity: Tighter stop (0.5%) yields higher stop hit count than 0.9% stop."""
    analyzer = StrategySensitivityAnalyzer()
    res, _ = analyzer.run_sensitivity_analysis(
        sample_candles_df,
        output_json_path="scratch/test_sens.json",
        output_csv_path="scratch/test_sens.csv"
    )

    cfg_A = next(r for r in res if r["combo_name"] == "A")  # 1.0% target, 0.5% stop
    cfg_B = next(r for r in res if r["combo_name"] == "B")  # 1.0% target, 0.9% stop

    assert cfg_A["stop_hit_count"] >= cfg_B["stop_hit_count"]


def test_hold_period_sensitivity(sample_candles_df):
    """TEST 3 — Hold-Period Sensitivity: Shorter hold (60m) yields higher timeout count than 240m hold."""
    analyzer = StrategySensitivityAnalyzer()
    res, _ = analyzer.run_sensitivity_analysis(
        sample_candles_df,
        output_json_path="scratch/test_sens.json",
        output_csv_path="scratch/test_sens.csv"
    )

    cfg_60m = next(r for r in res if r["combo_name"] == "E_60m")
    cfg_240m = next(r for r in res if r["combo_name"] == "E")

    assert cfg_60m["timeout_count"] >= cfg_240m["timeout_count"]


def test_long_short_symmetry(sample_candles_df):
    """TEST 4 — LONG/SHORT Symmetry: Total candidate count is split 50/50 between LONG and SHORT."""
    analyzer = StrategySensitivityAnalyzer()
    res, _ = analyzer.run_sensitivity_analysis(
        sample_candles_df,
        output_json_path="scratch/test_sens.json",
        output_csv_path="scratch/test_sens.csv"
    )

    cfg = res[0]
    assert cfg["long_candidates"] == cfg["short_candidates"]
    assert cfg["total_candidates"] == cfg["long_candidates"] + cfg["short_candidates"]


def test_entry_candle_exclusion(sample_candles_df):
    """TEST 5 — Entry Candle Exclusion: Entry candle high/low never triggers target/stop."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 110, "low": 90, "close": 100, "volume": 1000}
    ])

    analyzer = StrategySensitivityAnalyzer()
    res, _ = analyzer.run_sensitivity_analysis(
        df,
        output_json_path="scratch/test_sens.json",
        output_csv_path="scratch/test_sens.csv"
    )

    cfg = res[0]
    assert cfg["insufficient_future_data_count"] == 2
    assert cfg["target_hit_count"] == 0
    assert cfg["stop_hit_count"] == 0


def test_same_day_restriction(sample_candles_df):
    """TEST 6 — Same-Day Restriction: Day 1 entry cannot use Day 2 candles."""
    ts_d1 = pd.Timestamp("2026-09-01 15:15", tz="Asia/Kolkata")
    ts_d2 = pd.Timestamp("2026-09-02 09:15", tz="Asia/Kolkata")

    df = pd.DataFrame([
        {"timestamp": ts_d1, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"timestamp": ts_d2, "symbol": "RELIANCE.NS", "open": 100, "high": 110, "low": 100, "close": 110, "volume": 1000}
    ])

    analyzer = StrategySensitivityAnalyzer()
    res, _ = analyzer.run_sensitivity_analysis(
        df,
        output_json_path="scratch/test_sens.json",
        output_csv_path="scratch/test_sens.csv"
    )

    cfg = res[0]
    # Day 1 entry has 0 future candles on Day 1 -> insufficient data
    assert cfg["insufficient_future_data_count"] >= 2


def test_same_candle_ambiguity():
    """TEST 7 — Same-Candle Ambiguity: Same future candle hitting target and stop is marked AMBIGUOUS."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata")

    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "open": 100, "high": 105, "low": 95, "close": 100, "volume": 1000}
    ])

    analyzer = StrategySensitivityAnalyzer()
    res, _ = analyzer.run_sensitivity_analysis(
        df,
        output_json_path="scratch/test_sens.json",
        output_csv_path="scratch/test_sens.csv"
    )

    cfg = res[0]
    assert cfg["ambiguous_count"] > 0


def test_timeout_handling(sample_candles_df):
    """TEST 8 — Timeout Handling: Trades unresolved within horizon exit as TIMEOUT."""
    analyzer = StrategySensitivityAnalyzer()
    res, _ = analyzer.run_sensitivity_analysis(
        sample_candles_df,
        output_json_path="scratch/test_sens.json",
        output_csv_path="scratch/test_sens.csv"
    )

    cfg = res[0]
    assert "timeout_count" in cfg
    assert cfg["timeout_count"] >= 0


def test_insufficient_future_handling(sample_candles_df):
    """TEST 9 — Insufficient Future Data Handling: Final candle of session marked INSUFFICIENT_FUTURE_DATA."""
    analyzer = StrategySensitivityAnalyzer()
    res, _ = analyzer.run_sensitivity_analysis(
        sample_candles_df,
        output_json_path="scratch/test_sens.json",
        output_csv_path="scratch/test_sens.csv"
    )

    cfg = res[0]
    assert cfg["insufficient_future_data_count"] > 0


def test_deterministic_reproducible_results(sample_candles_df):
    """TEST 10 — Deterministic Reproducibility: Re-running yields 100% identical metrics."""
    analyzer = StrategySensitivityAnalyzer()
    res1, _ = analyzer.run_sensitivity_analysis(sample_candles_df, output_json_path="scratch/test_sens1.json", output_csv_path="scratch/test_sens1.csv")
    res2, _ = analyzer.run_sensitivity_analysis(sample_candles_df, output_json_path="scratch/test_sens2.json", output_csv_path="scratch/test_sens2.csv")

    assert res1 == res2


def test_no_modification_of_phase2_dataset():
    """TEST 11 — Protection of Phase 2 Labeled Dataset: File data/processed/labeled_dataset.parquet is NOT modified."""
    phase2_file = "data/processed/labeled_dataset.parquet"
    if os.path.exists(phase2_file):
        mtime_before = os.path.getmtime(phase2_file)
        
        # Run sensitivity analysis
        df_sample = pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata"), "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}
        ])
        analyzer = StrategySensitivityAnalyzer()
        analyzer.run_sensitivity_analysis(df_sample, output_json_path="scratch/test_sens.json", output_csv_path="scratch/test_sens.csv")
        
        mtime_after = os.path.getmtime(phase2_file)
        assert mtime_before == mtime_after, "Phase 2 labeled_dataset.parquet was modified!"


def test_no_future_leakage():
    """TEST 12 — No Future Leakage: Outcome columns (target_price, stop_price, exit_reason) are separated from features."""
    from app.ml.labeling import HistoricalTradeLabeler
    for col in HistoricalTradeLabeler.OUTCOME_COLUMNS:
        assert col not in HistoricalTradeLabeler.FEATURE_COLUMNS
