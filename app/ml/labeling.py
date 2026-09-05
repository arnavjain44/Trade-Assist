import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from app.config import settings

logger = logging.getLogger(__name__)


class HistoricalTradeLabeler:
    """
    Real Historical Outcome Labeling Pipeline (Phase 2).

    Constructs hypothetical intraday LONG and SHORT trades for every eligible
    5-minute feature row, evaluates future intraday price action chronologically
    within the same trading session, and labels historical outcomes.

    Strict Constraints & Principles:
    - The label represents a historical hypothetical trading outcome. IT IS NOT A PREDICTION.
    - Zero future-data leakage into feature columns.
    - Entry candle itself is strictly EXCLUDED from outcome determination.
    - Same-day intraday rule: only candles belonging to the SAME trading date (Asia/Kolkata) are inspected.
    - Same-candle target + stop ambiguity is assigned label_status = 'AMBIGUOUS' and excluded from ML training.
    """

    FEATURE_COLUMNS = [
        "open", "high", "low", "close", "volume",
        "ema_5", "rsi", "obv",
        "bollinger_middle", "bollinger_upper", "bollinger_lower",
        "macd", "macd_signal", "macd_diff", "vwap",
        "bollinger_position", "price_vs_vwap", "price_vs_ema5",
        "sentiment_score", "market_similarity", "stock_similarity"
    ]

    OUTCOME_COLUMNS = [
        "entry_price", "target_price", "stop_price", "direction",
        "label", "label_status", "exit_timestamp", "exit_price",
        "exit_reason", "holding_period_minutes", "realized_return"
    ]

    def __init__(
        self,
        target_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        max_hold_minutes: Optional[int] = None
    ):
        self.target_pct = target_pct if target_pct is not None else settings.LABEL_TARGET_PCT
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else settings.LABEL_STOP_LOSS_PCT
        self.max_hold_minutes = max_hold_minutes if max_hold_minutes is not None else settings.LABEL_MAX_HOLD_MINUTES

    def label_dataset(
        self,
        df: pd.DataFrame,
        output_parquet_path: str = "data/processed/labeled_dataset.parquet",
        output_json_path: str = "data/processed/label_quality.json"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generates hypothetical LONG & SHORT trade outcomes for all eligible rows in df.

        Returns (labeled_df, quality_report_dict).
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Ensure Asia/Kolkata timezone
        if hasattr(df["timestamp"].dt, "tz") and df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")

        symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else ["UNKNOWN"]
        labeled_rows = []

        logger.info(
            "Starting Historical Outcome Labeling (Target: %.3f, Stop: %.3f, MaxHold: %dm) across %d symbols...",
            self.target_pct, self.stop_loss_pct, self.max_hold_minutes, len(symbols)
        )

        for sym in symbols:
            sym_df = df[df["symbol"] == sym].sort_values("timestamp").reset_index(drop=True) if "symbol" in df.columns else df.sort_values("timestamp").reset_index(drop=True)
            sym_labeled = self._label_single_symbol(sym_df)
            labeled_rows.append(sym_labeled)

        non_empty_rows = [r for r in labeled_rows if not r.empty]
        combined_labeled = pd.concat(non_empty_rows, ignore_index=True) if non_empty_rows else pd.DataFrame()

        # Generate Quality Audit Report
        quality_report = self._generate_quality_report(combined_labeled, len(df))

        # Save Parquet and JSON artifacts
        os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
        combined_labeled.to_parquet(os.path.abspath(output_parquet_path), index=False)
        logger.info("Labeled dataset exported to %s (%d candidate rows)", output_parquet_path, len(combined_labeled))

        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(os.path.abspath(output_json_path), "w", encoding="utf-8") as f:
            json.dump(quality_report, f, indent=2)
        logger.info("Label quality report exported to %s", output_json_path)

        return combined_labeled, quality_report

    def _label_single_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Labels all 5-minute bars for a single ticker chronologically."""
        df = df.sort_values("timestamp").reset_index(drop=True)
        n_rows = len(df)
        
        # Extract session dates for same-day restriction
        session_dates = df["timestamp"].dt.date

        output_records = []

        for i in range(n_rows):
            entry_row = df.iloc[i]
            entry_ts = entry_row["timestamp"]
            entry_date = session_dates.iloc[i]
            entry_close = float(entry_row["close"])

            # Identify future candidates strictly AFTER entry index i, within same session date & max holding window
            future_mask = (
                (df.index > i) &
                (session_dates == entry_date) &
                (df["timestamp"] <= entry_ts + pd.Timedelta(minutes=self.max_hold_minutes))
            )
            future_df = df.loc[future_mask]

            # Evaluate LONG (direction = 1) and SHORT (direction = -1) independently
            for direction in [1, -1]:
                record = entry_row.to_dict()

                if direction == 1:
                    target_price = entry_close * (1.0 + self.target_pct)
                    stop_price = entry_close * (1.0 - self.stop_loss_pct)
                else:
                    target_price = entry_close * (1.0 - self.target_pct)
                    stop_price = entry_close * (1.0 + self.stop_loss_pct)

                record["direction"] = direction
                record["entry_price"] = entry_close
                record["target_price"] = target_price
                record["stop_price"] = stop_price

                # Outcome Determination
                if future_df.empty:
                    record["label"] = None
                    record["label_status"] = "INSUFFICIENT_FUTURE_DATA"
                    record["exit_timestamp"] = None
                    record["exit_price"] = None
                    record["exit_reason"] = "INSUFFICIENT_FUTURE_DATA"
                    record["holding_period_minutes"] = 0.0
                    record["realized_return"] = 0.0
                else:
                    outcome = self._evaluate_trade_outcome(
                        future_df, direction, entry_close, target_price, stop_price, entry_ts
                    )
                    record.update(outcome)

                output_records.append(record)

        return pd.DataFrame(output_records)

    def _evaluate_trade_outcome(
        self,
        future_df: pd.DataFrame,
        direction: int,
        entry_price: float,
        target_price: float,
        stop_price: float,
        entry_ts: pd.Timestamp
    ) -> Dict[str, Any]:
        """Evaluates future candles chronologically to determine trade outcome."""
        for _, fut_row in future_df.iterrows():
            fut_ts = fut_row["timestamp"]
            fut_high = float(fut_row["high"])
            fut_low = float(fut_row["low"])

            if direction == 1:  # LONG
                target_hit = (fut_high >= target_price)
                stop_hit = (fut_low <= stop_price)
            else:  # SHORT
                target_hit = (fut_low <= target_price)
                stop_hit = (fut_high >= stop_price)

            # Same-candle Ambiguity
            if target_hit and stop_hit:
                hold_mins = (fut_ts - entry_ts).total_seconds() / 60.0
                return {
                    "label": None,
                    "label_status": "AMBIGUOUS",
                    "exit_timestamp": fut_ts,
                    "exit_price": None,
                    "exit_reason": "AMBIGUOUS",
                    "holding_period_minutes": hold_mins,
                    "realized_return": 0.0
                }

            if target_hit:
                hold_mins = (fut_ts - entry_ts).total_seconds() / 60.0
                realized_ret = (target_price - entry_price) / entry_price if direction == 1 else (entry_price - target_price) / entry_price
                return {
                    "label": 1,
                    "label_status": "VALID",
                    "exit_timestamp": fut_ts,
                    "exit_price": target_price,
                    "exit_reason": "TARGET",
                    "holding_period_minutes": hold_mins,
                    "realized_return": round(realized_ret, 6)
                }

            if stop_hit:
                hold_mins = (fut_ts - entry_ts).total_seconds() / 60.0
                realized_ret = (stop_price - entry_price) / entry_price if direction == 1 else (entry_price - stop_price) / entry_price
                return {
                    "label": 0,
                    "label_status": "VALID",
                    "exit_timestamp": fut_ts,
                    "exit_price": stop_price,
                    "exit_reason": "STOP",
                    "holding_period_minutes": hold_mins,
                    "realized_return": round(realized_ret, 6)
                }

        # Timeout: Exit at final valid candle in same-session horizon
        last_row = future_df.iloc[-1]
        last_ts = last_row["timestamp"]
        last_close = float(last_row["close"])
        hold_mins = (last_ts - entry_ts).total_seconds() / 60.0
        realized_ret = (last_close - entry_price) / entry_price if direction == 1 else (entry_price - last_close) / entry_price

        return {
            "label": 0,  # Trade failed to hit target before stop objective
            "label_status": "VALID",
            "exit_timestamp": last_ts,
            "exit_price": last_close,
            "exit_reason": "TIMEOUT",
            "holding_period_minutes": hold_mins,
            "realized_return": round(realized_ret, 6)
        }

    def _generate_quality_report(self, df: pd.DataFrame, source_rows: int) -> Dict[str, Any]:
        """Generates comprehensive label quality audit JSON object."""
        total_candidates = len(df)
        long_candidates = len(df[df["direction"] == 1])
        short_candidates = len(df[df["direction"] == -1])

        valid_df = df[df["label_status"] == "VALID"]
        valid_count = len(valid_df)

        pos_count = int((valid_df["label"] == 1).sum())
        neg_count = int((valid_df["label"] == 0).sum())

        ambiguous_count = int((df["label_status"] == "AMBIGUOUS").sum())
        timeout_count = int((df["exit_reason"] == "TIMEOUT").sum())
        insufficient_count = int((df["label_status"] == "INSUFFICIENT_FUTURE_DATA").sum())

        pos_rate = round((pos_count / valid_count) * 100.0, 2) if valid_count > 0 else 0.0
        neg_rate = round((neg_count / valid_count) * 100.0, 2) if valid_count > 0 else 0.0

        symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []

        return {
            "labeling_status": "SUCCESS",
            "labeling_version": "1.0.0",
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "target_pct": self.target_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "max_hold_minutes": self.max_hold_minutes,
            "source_dataset_information": {
                "source_rows": source_rows,
                "symbols_count": len(symbols),
                "date_range": {
                    "start": str(df["timestamp"].min()) if "timestamp" in df.columns else "N/A",
                    "end": str(df["timestamp"].max()) if "timestamp" in df.columns else "N/A"
                }
            },
            "total_candidate_rows": total_candidates,
            "long_candidate_rows": long_candidates,
            "short_candidate_rows": short_candidates,
            "valid_labeled_rows": valid_count,
            "positive_labels": pos_count,
            "negative_labels": neg_count,
            "positive_rate_pct": pos_rate,
            "negative_rate_pct": neg_rate,
            "ambiguous_rows": ambiguous_count,
            "timeout_rows": timeout_count,
            "insufficient_future_data_rows": insufficient_count,
            "symbols_count": len(symbols),
            "symbols": symbols
        }


trade_labeler = HistoricalTradeLabeler()
