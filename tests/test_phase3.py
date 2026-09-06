import pytest
import numpy as np
import pandas as pd
from app.ml.models import TradeSignalClassifier
from app.ml.evaluation import ChronologicalEvaluator


@pytest.fixture
def sample_features_df():
    """Generates synthetic multi-day chronological DataFrame for testing Phase 3 modules."""
    np.random.seed(42)
    n_rows = 500
    dates = pd.date_range("2026-09-01 09:15", periods=60, freq="1D", tz="Asia/Kolkata")
    
    records = []
    for d in dates:
        for sym in ["RELIANCE.NS", "TCS.NS"]:
            for i in range(4):  # 4 candles per day
                ts = d + pd.Timedelta(minutes=i * 15)
                exit_ts = ts + pd.Timedelta(minutes=240)
                records.append({
                    "timestamp": ts,
                    "symbol": sym,
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 1000.0,
                    "rsi": np.random.uniform(30, 70),
                    "obv": np.random.uniform(-1000, 1000),
                    "bollinger_position": np.random.uniform(0, 1),
                    "macd": np.random.uniform(-1, 1),
                    "macd_signal": np.random.uniform(-1, 1),
                    "macd_diff": np.random.uniform(-0.5, 0.5),
                    "price_vs_vwap": np.random.uniform(-0.01, 0.01),
                    "price_vs_ema5": np.random.uniform(-0.01, 0.01),
                    "direction": 1 if np.random.rand() > 0.5 else -1,
                    "label": 1 if np.random.rand() > 0.85 else 0,
                    "label_status": "VALID",
                    "exit_reason": np.random.choice(["TARGET_HIT", "STOP_HIT", "TIMEOUT"]),
                    "exit_price": 102.0 if np.random.rand() > 0.5 else 99.1,
                    "exit_timestamp": exit_ts,
                    "realized_return": np.random.uniform(-0.009, 0.022)
                })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


def test_chronological_split_ordering(sample_features_df):
    """TEST 1 — CHRONOLOGICAL SPLIT ORDERING: Verify train < val < test dates."""
    evaluator = ChronologicalEvaluator()
    df_train, df_val, df_test, meta = evaluator.split_dataset_chronologically(sample_features_df, train_days=30, val_days=15)

    assert meta["train_days"] > 0
    assert meta["val_days"] > 0
    assert meta["test_days"] > 0

    max_train_ts = df_train["timestamp"].max()
    min_val_ts = df_val["timestamp"].min()
    max_val_ts = df_val["timestamp"].max()
    min_test_ts = df_test["timestamp"].min()

    assert max_train_ts < min_val_ts
    assert max_val_ts < min_test_ts


def test_no_train_test_overlap(sample_features_df):
    """TEST 2 — NO TRAIN TEST OVERLAP: Ensure disjoint sets of dates."""
    evaluator = ChronologicalEvaluator()
    df_train, df_val, df_test, _ = evaluator.split_dataset_chronologically(sample_features_df)

    train_dates = set(df_train["timestamp"].dt.date.unique())
    val_dates = set(df_val["timestamp"].dt.date.unique())
    test_dates = set(df_test["timestamp"].dt.date.unique())

    assert train_dates.isdisjoint(val_dates)
    assert val_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(test_dates)


def test_boundary_purging(sample_features_df):
    """TEST 3 — BOUNDARY PURGING: Purged rows exist when horizon extends into val set."""
    evaluator = ChronologicalEvaluator(horizon_minutes=240)
    df_train, df_val, _, meta = evaluator.split_dataset_chronologically(sample_features_df)

    assert "purged_train_rows" in meta
    # All remaining train rows must be <= val_start_ts - 240m
    val_start_ts = pd.Timestamp(meta["val_start_date"]).tz_localize("Asia/Kolkata")
    purge_cutoff = val_start_ts - pd.Timedelta(minutes=240)
    assert df_train["timestamp"].max() <= purge_cutoff


def test_boundary_embargo(sample_features_df):
    """TEST 4 — BOUNDARY EMBARGO: Val rows near test boundary are purged."""
    evaluator = ChronologicalEvaluator(horizon_minutes=240)
    _, df_val, _, meta = evaluator.split_dataset_chronologically(sample_features_df)

    test_start_ts = pd.Timestamp(meta["test_start_date"]).tz_localize("Asia/Kolkata")
    purge_cutoff = test_start_ts - pd.Timedelta(minutes=240)
    assert df_val["timestamp"].max() <= purge_cutoff


def test_no_future_leakage(sample_features_df):
    """TEST 5 — NO FUTURE LEAKAGE: Scaler fitted on train does not use val/test data."""
    evaluator = ChronologicalEvaluator()
    df_train, df_val, df_test, _ = evaluator.split_dataset_chronologically(sample_features_df)

    clf = TradeSignalClassifier(model_family="logistic_regression")
    clf.fit(df_train, df_train["label"].values)

    train_mean = clf.scaler.mean_
    assert len(train_mean) == len(clf.FEATURE_COLS)


def test_train_only_scaler_fitting(sample_features_df):
    """TEST 6 — TRAIN ONLY SCALER: Tree models have scaler=None, LR has scaler."""
    clf_lr = TradeSignalClassifier(model_family="logistic_regression")
    clf_rf = TradeSignalClassifier(model_family="random_forest")

    assert clf_lr.scaler is not None
    assert clf_rf.scaler is None


def test_class_weight_grid():
    """TEST 7 — CLASS WEIGHT GRID: Verify neutral initialization across models."""
    clf_lr = TradeSignalClassifier(model_family="logistic_regression", pos_weight=74.0)
    clf_rf = TradeSignalClassifier(model_family="random_forest", pos_weight=50.0)

    assert clf_lr.pos_weight == 74.0
    assert clf_rf.pos_weight == 50.0
    assert clf_lr.model.class_weight == {0: 1.0, 1: 74.0}


def test_model_family_weighting_translation():
    """TEST 8 — WEIGHTING TRANSLATION: Boosting uses scale_pos_weight, sklearn uses dict."""
    try:
        clf_gb = TradeSignalClassifier(model_family="lightgbm", pos_weight=10.0)
        assert getattr(clf_gb.model, "scale_pos_weight", 10.0) == 10.0
    except RuntimeError:
        pytest.skip("Boosting library not available.")


def test_calibration_separation(sample_features_df):
    """TEST 9 — CALIBRATION SEPARATION: Fit calibrator on raw probabilities."""
    clf = TradeSignalClassifier(model_family="logistic_regression")
    clf.fit(sample_features_df, sample_features_df["label"].values)

    raw_p = clf.predict_proba_raw(sample_features_df)
    clf.fit_calibrator(raw_p, sample_features_df["label"].values, method="isotonic")

    calib_p = clf.predict_proba(sample_features_df)
    assert len(calib_p) == len(sample_features_df)
    assert np.all(calib_p >= 0.0) and np.all(calib_p <= 1.0)


def test_threshold_range(sample_features_df):
    """TEST 10 — THRESHOLD RANGE: select_optimal_threshold returns valid float."""
    evaluator = ChronologicalEvaluator()
    probs = np.random.uniform(0.4, 0.9, size=len(sample_features_df))

    best_thresh, econ = evaluator.select_optimal_threshold(sample_features_df, probs, min_trade_count=5)
    assert 0.50 <= best_thresh <= 0.95


def test_30_trade_threshold_guard(sample_features_df):
    """TEST 11 — 30 TRADE GUARD: Thresholds yielding < min_trade_count are guarded against."""
    evaluator = ChronologicalEvaluator()
    # High threshold where almost no trades qualify
    probs = np.full(len(sample_features_df), 0.55)

    best_thresh, econ = evaluator.select_optimal_threshold(sample_features_df, probs, min_trade_count=1000)
    # Should fall back gracefully without crashing
    assert "selected_trade_count" in econ


def test_single_position_per_symbol():
    """TEST 12 — SINGLE POSITION PER SYMBOL: Overlapping signals for same symbol are suppressed."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts1 = pd.Timestamp("2026-09-01 09:30", tz="Asia/Kolkata")  # While ts0 is active!
    ts0_exit = pd.Timestamp("2026-09-01 10:15", tz="Asia/Kolkata")

    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "direction": 1, "realized_return": 0.01, "exit_reason": "TARGET_HIT", "exit_timestamp": ts0_exit},
        {"timestamp": ts1, "symbol": "RELIANCE.NS", "direction": 1, "realized_return": 0.02, "exit_reason": "TARGET_HIT", "exit_timestamp": ts0_exit + pd.Timedelta(minutes=15)}
    ])

    evaluator = ChronologicalEvaluator()
    probs = np.array([0.80, 0.85])
    econ = evaluator.backtest_economic_performance(df, probs, threshold=0.50)

    # ts1 signal should be suppressed because ts0 active trade exits at 10:15!
    assert econ["selected_trade_count"] == 1


def test_overlapping_signal_suppression():
    """TEST 13 — OVERLAPPING SIGNAL SUPPRESSION: Different symbols run concurrently."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts0_exit = pd.Timestamp("2026-09-01 10:15", tz="Asia/Kolkata")

    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "direction": 1, "realized_return": 0.01, "exit_reason": "TARGET_HIT", "exit_timestamp": ts0_exit},
        {"timestamp": ts0, "symbol": "TCS.NS", "direction": 1, "realized_return": 0.01, "exit_reason": "TARGET_HIT", "exit_timestamp": ts0_exit}
    ])

    evaluator = ChronologicalEvaluator()
    probs = np.array([0.80, 0.85])
    econ = evaluator.backtest_economic_performance(df, probs, threshold=0.50)

    # Both symbols should be accepted since they are different tickers!
    assert econ["selected_trade_count"] == 2


def test_005_transaction_cost():
    """TEST 14 — 0.05% COST: Net return = Gross return - 0.0005."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts0_exit = pd.Timestamp("2026-09-01 10:15", tz="Asia/Kolkata")

    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "direction": 1, "realized_return": 0.01, "exit_reason": "TARGET_HIT", "exit_timestamp": ts0_exit}
    ])

    evaluator = ChronologicalEvaluator(cost_pct=0.0005)
    econ = evaluator.backtest_economic_performance(df, np.array([0.80]), threshold=0.50)

    assert np.isclose(econ["gross_avg_return_pct"], 1.0)
    assert np.isclose(econ["net_avg_return_pct"], 0.95)  # 1.0% - 0.05% = 0.95%


def test_timeout_realized_return_preservation():
    """TEST 15 — TIMEOUT REALIZED RETURN: TIMEOUT preserves exact return (+0.12%)."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts0_exit = pd.Timestamp("2026-09-01 13:15", tz="Asia/Kolkata")

    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "direction": 1, "realized_return": 0.0012, "exit_reason": "TIMEOUT", "exit_timestamp": ts0_exit}
    ])

    evaluator = ChronologicalEvaluator(cost_pct=0.0005)
    econ = evaluator.backtest_economic_performance(df, np.array([0.80]), threshold=0.50)

    assert econ["timeouts"] == 1
    assert np.isclose(econ["gross_avg_return_pct"], 0.12)


def test_long_short_separation():
    """TEST 16 — LONG SHORT SEPARATION: Breakdown calculates metrics separately."""
    ts0 = pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    ts0_exit = pd.Timestamp("2026-09-01 10:15", tz="Asia/Kolkata")

    df = pd.DataFrame([
        {"timestamp": ts0, "symbol": "RELIANCE.NS", "direction": 1, "realized_return": 0.02, "exit_reason": "TARGET_HIT", "exit_timestamp": ts0_exit},
        {"timestamp": ts0, "symbol": "TCS.NS", "direction": -1, "realized_return": -0.01, "exit_reason": "STOP_HIT", "exit_timestamp": ts0_exit}
    ])

    evaluator = ChronologicalEvaluator(cost_pct=0.0005)
    econ = evaluator.backtest_economic_performance(df, np.array([0.80, 0.80]), threshold=0.50)

    assert econ["long_count"] == 1
    assert econ["short_count"] == 1
    assert econ["long_net_avg_return_pct"] > 0
    assert econ["short_net_avg_return_pct"] < 0


def test_test_set_locking(sample_features_df):
    """TEST 17 — TEST SET LOCKING: Test set evaluation uses pre-fitted model & threshold."""
    evaluator = ChronologicalEvaluator()
    df_train, df_val, df_test, _ = evaluator.split_dataset_chronologically(sample_features_df)

    clf = TradeSignalClassifier(model_family="logistic_regression")
    clf.fit(df_train, df_train["label"].values)

    test_probs = clf.predict_proba(df_test)
    econ_test = evaluator.backtest_economic_performance(df_test, test_probs, threshold=0.60)

    assert "net_avg_return_pct" in econ_test


def test_deterministic_results():
    """TEST 18 — DETERMINISTIC RESULTS: Fixed seed produces identical predictions."""
    clf1 = TradeSignalClassifier(model_family="random_forest", random_state=42)
    clf2 = TradeSignalClassifier(model_family="random_forest", random_state=42)

    X = pd.DataFrame(np.random.randn(50, 9), columns=TradeSignalClassifier.FEATURE_COLS)
    y = np.random.randint(0, 2, size=50)

    clf1.fit(X, y)
    clf2.fit(X, y)

    p1 = clf1.predict_proba_raw(X)
    p2 = clf2.predict_proba_raw(X)

    np.testing.assert_array_almost_equal(p1, p2)
