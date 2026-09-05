import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, brier_score_loss

logger = logging.getLogger(__name__)


class ChronologicalEvaluator:
    """
    Evaluation framework enforcing:
    1. Deterministic chronological train (days 1-42), val (days 43-51), test (days 52-60) splitting.
    2. Timestamp-based 240m purging and embargo at temporal boundaries.
    3. Classification metric computation (PR-AUC, Precision, Recall, ECE, Brier score).
    4. Economic backtesting enforcing single-position-per-symbol overlap rule and 0.05% friction.
    5. Threshold optimization with a mandatory minimum trade-count guard (min 30 validation trades).
    """

    def __init__(self, cost_pct: float = 0.0005, horizon_minutes: int = 240):
        """
        cost_pct: Round-trip transaction cost (default 0.05% / 5 bps).
        horizon_minutes: Maximum trade holding horizon for purging / embargo (default 240m).
        """
        self.cost_pct = cost_pct
        self.horizon_minutes = horizon_minutes

    def split_dataset_chronologically(
        self,
        df: pd.DataFrame,
        train_days: int = 42,
        val_days: int = 9
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Splits dataset chronologically into Train (first 42 days), Val (next 9 days), Test (remaining ~9 days).
        Applies timestamp-based purging (240m horizon) at boundary cuts.

        Returns (df_train, df_val, df_test, split_metadata).
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Extract unique dates sorted
        unique_dates = np.sort(df["timestamp"].dt.date.unique())
        total_days = len(unique_dates)

        if total_days < (train_days + val_days + 1):
            # Dynamic proportional split if dataset has fewer days than expected
            train_end_idx = int(total_days * 0.70)
            val_end_idx = int(total_days * 0.85)
        else:
            train_end_idx = train_days
            val_end_idx = train_days + val_days

        train_dates = set(unique_dates[:train_end_idx])
        val_dates = set(unique_dates[train_end_idx:val_end_idx])
        test_dates = set(unique_dates[val_end_idx:])

        val_start_date = unique_dates[train_end_idx]
        test_start_date = unique_dates[val_end_idx]

        # Convert boundary start dates to timestamp cutoffs
        val_start_ts = pd.Timestamp(val_start_date)
        test_start_ts = pd.Timestamp(test_start_date)

        if hasattr(df["timestamp"].dt, "tz") and df["timestamp"].dt.tz is not None:
            tz = df["timestamp"].dt.tz
            val_start_ts = val_start_ts.tz_localize(tz)
            test_start_ts = test_start_ts.tz_localize(tz)

        # Purging cutoff: 240 minutes before val_start_ts and test_start_ts
        train_purge_cutoff = val_start_ts - pd.Timedelta(minutes=self.horizon_minutes)
        val_purge_cutoff = test_start_ts - pd.Timedelta(minutes=self.horizon_minutes)

        # Define raw masks based on dates
        raw_train_mask = df["timestamp"].dt.date.isin(train_dates)
        raw_val_mask = df["timestamp"].dt.date.isin(val_dates)
        raw_test_mask = df["timestamp"].dt.date.isin(test_dates)

        # Apply purging: exclude train rows extending into val, and val rows extending into test
        clean_train_mask = raw_train_mask & (df["timestamp"] <= train_purge_cutoff)
        clean_val_mask = raw_val_mask & (df["timestamp"] <= val_purge_cutoff)
        clean_test_mask = raw_test_mask

        df_train = df[clean_train_mask].copy().reset_index(drop=True)
        df_val = df[clean_val_mask].copy().reset_index(drop=True)
        df_test = df[clean_test_mask].copy().reset_index(drop=True)

        purged_train_count = int((raw_train_mask & ~clean_train_mask).sum())
        purged_val_count = int((raw_val_mask & ~clean_val_mask).sum())

        metadata = {
            "total_rows": len(df),
            "total_days": total_days,
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "test_days": len(test_dates),
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "test_rows": len(df_test),
            "purged_train_rows": purged_train_count,
            "purged_val_rows": purged_val_count,
            "val_start_date": str(val_start_date),
            "test_start_date": str(test_start_date),
        }

        logger.info(
            "Chronological Split: Train=%d rows (%d purged), Val=%d rows (%d purged), Test=%d rows",
            len(df_train), purged_train_count, len(df_val), purged_val_count, len(df_test)
        )

        return df_train, df_val, df_test, metadata

    @staticmethod
    def calculate_statistical_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Calculates classification metrics: Precision, Recall, PR-AUC, Brier score, ECE, Confusion Matrix.
        """
        y_pred = (y_prob >= threshold).astype(int)

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(recall_curve, precision_curve))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        brier = float(brier_score_loss(y_true, y_prob))

        # Expected Calibration Error (ECE) calculation with 10 bins
        bin_boundaries = np.linspace(0, 1, 11)
        ece = 0.0
        n_samples = len(y_true)

        for i in range(10):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < 9 else y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return {
            "pr_auc": round(pr_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "brier_score": round(brier, 6),
            "ece": round(float(ece), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "threshold_used": round(threshold, 4),
        }

    def backtest_economic_performance(
        self,
        df: pd.DataFrame,
        probabilities: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Executes economic backtest on dataset filtered by probability >= threshold.

        Enforces:
        - SINGLE POSITION PER SYMBOL: If symbol S is active, subsequent signals for S are ignored until exit.
        - 0.05% ROUND-TRIP FRICTION applied to every selected trade.
        - CONTINUOUS REALIZED_RETURN preserved for TIMEOUT trades (not treated as -0.9% loss).

        Returns comprehensive economic results dictionary.
        """
        df_eval = df.copy().reset_index(drop=True)
        df_eval["prob"] = probabilities

        # Filter by threshold
        selected_mask = df_eval["prob"] >= threshold
        df_selected = df_eval[selected_mask].copy()

        if df_selected.empty:
            return self._empty_economic_result(threshold)

        # Sort chronologically
        df_selected = df_selected.sort_values("timestamp").reset_index(drop=True)

        # Active positions dict: symbol -> exit_timestamp
        active_positions: Dict[str, pd.Timestamp] = {}

        accepted_trades = []

        for idx, row in df_selected.iterrows():
            sym = row["symbol"]
            ts = pd.Timestamp(row["timestamp"])
            exit_ts = pd.Timestamp(row["exit_timestamp"])

            if ts.tzinfo is not None and exit_ts.tzinfo is None:
                exit_ts = exit_ts.tz_localize(ts.tzinfo)
            elif ts.tzinfo is None and exit_ts.tzinfo is not None:
                ts = ts.tz_localize(exit_ts.tzinfo)

            # Check if symbol has an active open trade
            if sym in active_positions:
                active_exit_ts = active_positions[sym]
                if active_exit_ts.tzinfo is None and ts.tzinfo is not None:
                    active_exit_ts = active_exit_ts.tz_localize(ts.tzinfo)
                elif active_exit_ts.tzinfo is not None and ts.tzinfo is None:
                    ts = ts.tz_localize(active_exit_ts.tzinfo)

                if ts < active_exit_ts:
                    # OVERLAP SUPPRESSION: Ignore signal while trade is active
                    continue

            # Open trade and record exit timestamp
            active_positions[sym] = exit_ts
            accepted_trades.append(row)

        if not accepted_trades:
            return self._empty_economic_result(threshold)

        df_trades = pd.DataFrame(accepted_trades).reset_index(drop=True)

        # Calculate gross and net returns
        gross_returns = df_trades["realized_return"].values
        net_returns = gross_returns - self.cost_pct

        total_selected = len(df_trades)
        unique_days = len(df_trades["timestamp"].dt.date.unique())
        trade_frequency_per_day = round(total_selected / max(unique_days, 1), 2)

        gross_avg_return_pct = float(np.mean(gross_returns)) * 100.0
        net_avg_return_pct = float(np.mean(net_returns)) * 100.0

        gross_total_return_pct = float(np.sum(gross_returns)) * 100.0
        net_total_return_pct = float(np.sum(net_returns)) * 100.0

        # Profit Factor calculations
        gross_gains = np.sum(gross_returns[gross_returns > 0])
        gross_losses = np.abs(np.sum(gross_returns[gross_returns < 0]))
        gross_profit_factor = float(gross_gains / gross_losses) if gross_losses > 0 else 999.0

        net_gains = np.sum(net_returns[net_returns > 0])
        net_losses = np.abs(np.sum(net_returns[net_returns < 0]))
        net_profit_factor = float(net_gains / net_losses) if net_losses > 0 else 999.0

        # Cumulative equity curve & Maximum Drawdown (in net return %)
        cum_net_returns = np.cumsum(net_returns) * 100.0
        peak = np.maximum.accumulate(cum_net_returns)
        drawdowns = peak - cum_net_returns
        max_drawdown_pct = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Directional breakdown (LONG vs SHORT)
        long_trades = df_trades[df_trades["direction"] == 1]
        short_trades = df_trades[df_trades["direction"] == -1]

        long_count = len(long_trades)
        short_count = len(short_trades)

        long_net_avg_pct = float(np.mean(long_trades["realized_return"] - self.cost_pct)) * 100.0 if long_count > 0 else 0.0
        short_net_avg_pct = float(np.mean(short_trades["realized_return"] - self.cost_pct)) * 100.0 if short_count > 0 else 0.0

        # Exit Reason Breakdown
        target_hits = int((df_trades["exit_reason"] == "TARGET_HIT").sum())
        stop_hits = int((df_trades["exit_reason"] == "STOP_HIT").sum())
        timeouts = int((df_trades["exit_reason"] == "TIMEOUT").sum())

        return {
            "threshold_used": round(threshold, 4),
            "selected_trade_count": total_selected,
            "trade_frequency_per_day": trade_frequency_per_day,
            "gross_avg_return_pct": round(gross_avg_return_pct, 4),
            "net_avg_return_pct": round(net_avg_return_pct, 4),
            "gross_total_return_pct": round(gross_total_return_pct, 4),
            "net_total_return_pct": round(net_total_return_pct, 4),
            "gross_profit_factor": round(gross_profit_factor, 4),
            "net_profit_factor": round(net_profit_factor, 4),
            "max_drawdown_pct": round(max_drawdown_pct, 4),
            "target_hits": target_hits,
            "stop_hits": stop_hits,
            "timeouts": timeouts,
            "long_count": long_count,
            "short_count": short_count,
            "long_net_avg_return_pct": round(long_net_avg_pct, 4),
            "short_net_avg_return_pct": round(short_net_avg_pct, 4),
        }

    def select_optimal_threshold(
        self,
        df_val: pd.DataFrame,
        probabilities: np.ndarray,
        min_trade_count: int = 30,
        threshold_range: Tuple[float, float, float] = (0.50, 0.95, 0.02)
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluates thresholds P* on Validation set and deterministically selects the optimal threshold.

        Guards:
        - ENFORCES MINIMUM 30 SELECTED TRADES GUARD on Validation set.
        - Primary Objective: Maximizes Net Average Return per Selected Trade.
        - Tie-breaker: Prefers higher trade count when net returns are within 0.01 bps.

        Returns (optimal_threshold, best_economic_result).
        """
        start, stop, step = threshold_range
        p_max = float(np.max(probabilities)) if len(probabilities) > 0 else 0.0

        if p_max < start and p_max > 0.001:
            # Calibrated probability range adaptation
            p_min = float(np.percentile(probabilities, 50))
            p_top = float(np.percentile(probabilities, 99.5))
            if p_top > p_min:
                candidate_thresholds = np.linspace(p_min, p_top, 23)
            else:
                candidate_thresholds = np.arange(start, stop + 1e-5, step)
        else:
            candidate_thresholds = np.arange(start, stop + 1e-5, step)

        best_thresh = 0.50
        best_net_avg = -999.0
        best_count = 0
        best_result = self._empty_economic_result(0.50)

        results = []

        for thresh in candidate_thresholds:
            thresh = float(round(thresh, 4))
            econ = self.backtest_economic_performance(df_val, probabilities, thresh)
            results.append(econ)

            count = econ["selected_trade_count"]
            net_avg = econ["net_avg_return_pct"]

            # Enforce minimum trade count guard
            if count >= min_trade_count:
                # Compare net average return
                if net_avg > best_net_avg + 0.0001:
                    best_net_avg = net_avg
                    best_thresh = thresh
                    best_count = count
                    best_result = econ
                elif abs(net_avg - best_net_avg) <= 0.0001:
                    # Tie breaker: prefer higher trade count
                    if count > best_count:
                        best_net_avg = net_avg
                        best_thresh = thresh
                        best_count = count
                        best_result = econ

        # Fallback if no threshold met the min_trade_count guard: select threshold with max trades above 0
        if best_count == 0:
            logger.warning(
                "No threshold satisfied the min_trade_count guard (%d trades). Falling back to threshold with max trades.",
                min_trade_count
            )
            for econ in results:
                if econ["selected_trade_count"] > best_count:
                    best_count = econ["selected_trade_count"]
                    best_thresh = econ["threshold_used"]
                    best_result = econ

        logger.info(
            "Validation Threshold Selected: P* = %.2f (Net Avg Return = %.4f%%, Selected Trades = %d)",
            best_thresh, best_result["net_avg_return_pct"], best_result["selected_trade_count"]
        )

        return best_thresh, best_result

    @staticmethod
    def _empty_economic_result(threshold: float) -> Dict[str, Any]:
        return {
            "threshold_used": round(threshold, 4),
            "selected_trade_count": 0,
            "trade_frequency_per_day": 0.0,
            "gross_avg_return_pct": 0.0,
            "net_avg_return_pct": 0.0,
            "gross_total_return_pct": 0.0,
            "net_total_return_pct": 0.0,
            "gross_profit_factor": 0.0,
            "net_profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "target_hits": 0,
            "stop_hits": 0,
            "timeouts": 0,
            "long_count": 0,
            "short_count": 0,
            "long_net_avg_return_pct": 0.0,
            "short_net_avg_return_pct": 0.0,
        }
