import pytest
import pandas as pd
import numpy as np
from app.ml.labeling import HistoricalTradeLabeler, trade_labeler


@pytest.fixture
def controlled_labeler():
    """Returns labeler with 2.0% target, 1.0% stop, and 60m max hold for easy deterministic math."""
    return HistoricalTradeLabeler(target_pct=0.02, stop_loss_pct=0.01, max_hold_minutes=60)


def test_long_target_first(controlled_labeler):
    """TEST 1 — LONG TARGET FIRST: Future high reaches LONG target before stop."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata")

    # Entry close = 100.0. Target = 102.0, Stop = 99.0
    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 105, "low": 95, "close": 100, "volume": 1000},  # Entry
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "open": 100, "high": 103, "low": 99.5, "close": 102.5, "volume": 1000}  # High 103 hits target 102
    ])

    labeled, _ = controlled_labeler.label_dataset(df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")
    long_res = labeled[(labeled["direction"] == 1) & (labeled["timestamp"] == ts0)].iloc[0]

    assert long_res["label"] == 1
    assert long_res["label_status"] == "VALID"
    assert long_res["exit_reason"] == "TARGET"
    assert np.isclose(long_res["exit_price"], 102.0)


def test_long_stop_first(controlled_labeler):
    """TEST 2 — LONG STOP FIRST: Stop is reached before target."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata")

    # Entry close = 100.0. Target = 102.0, Stop = 99.0
    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "open": 100, "high": 101, "low": 98.5, "close": 99.0, "volume": 1000}  # Low 98.5 hits stop 99
    ])

    labeled, _ = controlled_labeler.label_dataset(df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")
    long_res = labeled[(labeled["direction"] == 1) & (labeled["timestamp"] == ts0)].iloc[0]

    assert long_res["label"] == 0
    assert long_res["label_status"] == "VALID"
    assert long_res["exit_reason"] == "STOP"
    assert np.isclose(long_res["exit_price"], 99.0)


def test_short_target_first(controlled_labeler):
    """TEST 3 — SHORT TARGET FIRST: Future low reaches SHORT target before stop."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata")

    # Entry close = 100.0. SHORT Target = 98.0, SHORT Stop = 101.0
    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "open": 100, "high": 100.5, "low": 97.5, "close": 98.0, "volume": 1000}  # Low 97.5 hits target 98
    ])

    labeled, _ = controlled_labeler.label_dataset(df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")
    short_res = labeled[(labeled["direction"] == -1) & (labeled["timestamp"] == ts0)].iloc[0]

    assert short_res["label"] == 1
    assert short_res["label_status"] == "VALID"
    assert short_res["exit_reason"] == "TARGET"
    assert np.isclose(short_res["exit_price"], 98.0)


def test_short_stop_first(controlled_labeler):
    """TEST 4 — SHORT STOP FIRST: High reaches SHORT stop before target."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata")

    # Entry close = 100.0. SHORT Target = 98.0, SHORT Stop = 101.0
    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "open": 100, "high": 101.5, "low": 99.0, "close": 101.0, "volume": 1000}  # High 101.5 hits stop 101
    ])

    labeled, _ = controlled_labeler.label_dataset(df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")
    short_res = labeled[(labeled["direction"] == -1) & (labeled["timestamp"] == ts0)].iloc[0]

    assert short_res["label"] == 0
    assert short_res["label_status"] == "VALID"
    assert short_res["exit_reason"] == "STOP"
    assert np.isclose(short_res["exit_price"], 101.0)


def test_same_candle_ambiguity(controlled_labeler):
    """TEST 5 — SAME-CANDLE AMBIGUITY: Future candle hits BOTH target and stop simultaneously."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata")

    # Entry close = 100.0. LONG Target = 102.0, Stop = 99.0. Candle 1 high 103, low 98 (both hit!)
    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "open": 100, "high": 103, "low": 98, "close": 100, "volume": 1000}
    ])

    labeled, _ = controlled_labeler.label_dataset(df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")
    long_res = labeled[(labeled["direction"] == 1) & (labeled["timestamp"] == ts0)].iloc[0]

    assert long_res["label_status"] == "AMBIGUOUS"
    assert pd.isna(long_res["label"]) or long_res["label"] is None
    assert long_res["exit_reason"] == "AMBIGUOUS"


def test_same_day_exit(controlled_labeler):
    """TEST 6 & 12 — SAME-DAY EXIT / NO OVERNIGHT LABELING: Day 1 entry cannot use Day 2 candles."""
    ts0 = pd.Timestamp("2026-09-01 15:15", tz="Asia/Kolkata")  # End of Day 1
    ts_day2 = pd.Timestamp("2026-09-02 09:15", tz="Asia/Kolkata")  # Day 2

    # Entry close = 100.0. Day 2 candle has high 105 (target 102). Must NEVER be used!
    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"timestamp": ts_day2, "symbol": "RELIANCE.NS", "open": 100, "high": 105, "low": 100, "close": 105, "volume": 1000}
    ])

    labeled, _ = controlled_labeler.label_dataset(df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")
    res = labeled[(labeled["direction"] == 1) & (labeled["timestamp"] == ts0)].iloc[0]

    # Day 1 entry has no future candles on Day 1 -> INSUFFICIENT_FUTURE_DATA
    assert res["label_status"] == "INSUFFICIENT_FUTURE_DATA"
    assert res["exit_reason"] == "INSUFFICIENT_FUTURE_DATA"


def test_max_hold_timeout(controlled_labeler):
    """TEST 7 — MAX HOLD: Target hit beyond max_hold_minutes is ignored -> TIMEOUT."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:30", tz="Asia/Kolkata")  # 15 mins (within 60m horizon)
    ts2 = pd.Timestamp("2026-09-01 10:30", tz="Asia/Kolkata")  # 75 mins (beyond 60m horizon!)

    # Entry close = 100.0. Target = 102.0. ts1 stays 100.5. ts2 hits 105 (beyond horizon)
    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "open": 100, "high": 100.5, "low": 99.5, "close": 100.5, "volume": 1000},
        {"timestamp": ts2, "symbol": "RELIANCE.NS", "open": 100, "high": 105, "low": 100, "close": 105, "volume": 1000}
    ])

    labeled, _ = controlled_labeler.label_dataset(df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")
    res = labeled[(labeled["direction"] == 1) & (labeled["timestamp"] == ts0)].iloc[0]

    assert res["label_status"] == "VALID"
    assert res["exit_reason"] == "TIMEOUT"
    assert res["exit_timestamp"] == ts1  # Final candle within horizon
    assert np.isclose(res["exit_price"], 100.5)


def test_insufficient_future_data(controlled_labeler):
    """TEST 8 — INSUFFICIENT FUTURE DATA: Entry on final candle of dataset."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}
    ])

    labeled, _ = controlled_labeler.label_dataset(df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")
    res = labeled[labeled["timestamp"] == ts0].iloc[0]

    assert res["label_status"] == "INSUFFICIENT_FUTURE_DATA"
    assert res["exit_reason"] == "INSUFFICIENT_FUTURE_DATA"


def test_entry_candle_exclusion(controlled_labeler):
    """TEST 9 — ENTRY CANDLE EXCLUSION: Entry candle high/low must NOT determine outcome."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata")

    # Entry candle (ts0) has high 110 (target 102), but CLOSE is 100. ts1 stays 100.0.
    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 110, "low": 90, "close": 100, "volume": 1000},
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "open": 100, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}
    ])

    labeled, _ = controlled_labeler.label_dataset(df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")
    res = labeled[(labeled["direction"] == 1) & (labeled["timestamp"] == ts0)].iloc[0]

    # Entry candle high 110 must NOT trigger TARGET. Trade times out on ts1.
    assert res["exit_reason"] == "TIMEOUT"
    assert np.isclose(res["exit_price"], 100.0)


def test_long_short_symmetry(controlled_labeler):
    """TEST 10 — LONG/SHORT SYMMETRY: Symmetric price movements yield symmetric labels."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata")

    # Up move +3%: LONG hits target (+2%), SHORT hits stop (-0.9%)
    up_df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "open": 100, "high": 103, "low": 100, "close": 103, "volume": 1000}
    ])
    lbl_up, _ = controlled_labeler.label_dataset(up_df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")

    long_up = lbl_up[(lbl_up["direction"] == 1) & (lbl_up["timestamp"] == ts0)].iloc[0]
    short_up = lbl_up[(lbl_up["direction"] == -1) & (lbl_up["timestamp"] == ts0)].iloc[0]

    assert long_up["label"] == 1 and long_up["exit_reason"] == "TARGET"
    assert short_up["label"] == 0 and short_up["exit_reason"] == "STOP"


def test_multi_symbol_isolation(controlled_labeler):
    """TEST 11 — MULTI-SYMBOL ISOLATION: Symbol A future candles never label Symbol B."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata")

    # Symbol A has entry at ts0 and future at ts1. Symbol B ONLY has entry at ts0.
    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "open": 100, "high": 105, "low": 100, "close": 105, "volume": 1000},
        {"timestamp": ts0, "symbol": "TCS.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}
    ])

    labeled, _ = controlled_labeler.label_dataset(df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")

    tcs_res = labeled[(labeled["symbol"] == "TCS.NS") & (labeled["direction"] == 1)].iloc[0]
    # TCS has no future candles -> INSUFFICIENT_FUTURE_DATA. Must NOT borrow RELIANCE's ts1 candle!
    assert tcs_res["label_status"] == "INSUFFICIENT_FUTURE_DATA"


def test_dataset_integrity(controlled_labeler):
    """TEST 13 — DATASET INTEGRITY: Checks duplicates, timezone awareness, and non-negative entry prices."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata")

    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000},
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "open": 100, "high": 103, "low": 99.5, "close": 102.5, "volume": 1000}
    ])

    labeled, report = controlled_labeler.label_dataset(df, output_parquet_path="scratch/test_lbl.parquet", output_json_path="scratch/test_lbl.json")

    # 1. Check no duplicate (symbol, timestamp, direction)
    dups = labeled.duplicated(subset=["symbol", "timestamp", "direction"]).sum()
    assert dups == 0

    # 2. Check timezone awareness
    assert labeled["timestamp"].dt.tz is not None
    assert str(labeled["timestamp"].dt.tz) == "Asia/Kolkata"

    # 3. Check positive entry prices
    assert (labeled["entry_price"] > 0).all()

    # 4. Check feature columns versus outcome columns separation
    for col in HistoricalTradeLabeler.OUTCOME_COLUMNS:
        assert col in labeled.columns
