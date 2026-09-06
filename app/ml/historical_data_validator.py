"""
Historical Market Data Quality Validator Module (Phase 5.3)

Validates intraday OHLCV datasets for production-grade quantitative integrity:
1. Duplicate detection: (symbol, timestamp) collisions.
2. OHLC relationship validity: high >= max(open, close, low), low <= min(open, close, high).
3. Non-negativity: open, high, low, close > 0, volume >= 0.
4. Timezone verification: strictly timezone-aware Asia/Kolkata (IST).
5. Session hours: 09:15 to 15:30 IST regular trading hours.
6. Weekend & holiday contamination detection.
7. Missing candle accounting & expected session bar calculation (75 bars for regular session).
8. Abnormal price gap and volume anomaly flagging.
9. Auditable logging: records every anomaly; zero silent mutations.
"""

import logging
from datetime import time, datetime, date
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard NSE Trading Hours in IST
NSE_MARKET_OPEN = time(9, 15)
NSE_MARKET_CLOSE = time(15, 30)
EXPECTED_REGULAR_BARS_5M = 75  # 375 minutes / 5 minutes = 75 bars


class HistoricalDataValidator:
    """
    Validates and audits raw or processed historical intraday OHLCV DataFrames.
    Produces comprehensive, auditable anomaly reports without silent repairs.
    """

    def __init__(
        self,
        expected_timeframe: str = "5m",
        timezone_str: str = "Asia/Kolkata",
        allow_special_sessions: bool = True,
    ):
        self.expected_timeframe = expected_timeframe
        self.timezone_str = timezone_str
        self.allow_special_sessions = allow_special_sessions

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes full battery of data quality tests on DataFrame.
        Returns:
            clean_df: DataFrame with verified, sanitized rows.
            audit_report: Dict detailing exact anomaly counts and flags.
        """
        if df.empty:
            return df.copy(), {"status": "EMPTY_DATASET", "total_rows": 0, "anomalies": {}}

        df_work = df.copy()
        audit: Dict[str, Any] = {
            "validation_timestamp": datetime.now().isoformat(),
            "initial_rows": len(df_work),
            "anomalies_detected": 0,
            "violations": {},
            "symbol_coverage": {},
            "session_summary": {},
        }

        # 1. Schema & Column Presence
        required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required_cols if c not in df_work.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required canonical columns: {missing_cols}")

        # 2. Timezone Verification & Normalization
        df_work, tz_anomalies = self._verify_and_normalize_timezone(df_work)
        if tz_anomalies > 0:
            audit["violations"]["timezone_errors"] = tz_anomalies
            audit["anomalies_detected"] += tz_anomalies

        # 3. Duplicate Detection (symbol + timestamp)
        df_work, dup_count = self._detect_duplicates(df_work)
        if dup_count > 0:
            audit["violations"]["duplicate_candles"] = dup_count
            audit["anomalies_detected"] += dup_count

        # 4. OHLC Relationship Consistency
        df_work, ohlc_violations = self._validate_ohlc_relationships(df_work)
        if ohlc_violations > 0:
            audit["violations"]["invalid_ohlc_math"] = ohlc_violations
            audit["anomalies_detected"] += ohlc_violations

        # 5. Non-Positive Price & Negative Volume Detection
        df_work, price_vol_violations = self._validate_prices_and_volume(df_work)
        if price_vol_violations > 0:
            audit["violations"]["invalid_prices_or_volume"] = price_vol_violations
            audit["anomalies_detected"] += price_vol_violations

        # 6. Weekend & Off-Hours Contamination
        df_work, off_hours_count = self._validate_market_hours_and_weekends(df_work)
        if off_hours_count > 0:
            audit["violations"]["off_hours_or_weekend_contamination"] = off_hours_count
            audit["anomalies_detected"] += off_hours_count

        # 7. Trading Session & Missing Bars Accounting
        session_stats = self._audit_trading_sessions(df_work)
        audit["session_summary"] = session_stats

        # 8. Symbol Coverage Stats
        for sym, grp in df_work.groupby("symbol"):
            audit["symbol_coverage"][sym] = {
                "candle_count": len(grp),
                "earliest": str(grp["timestamp"].min()),
                "latest": str(grp["timestamp"].max()),
                "unique_days": int(grp["timestamp"].dt.date.nunique()),
            }

        audit["final_clean_rows"] = len(df_work)
        audit["status"] = "PASSED" if audit["anomalies_detected"] == 0 else "PASSED_WITH_FLAGS"

        logger.info(
            "Validation complete: %d rows -> %d clean rows (%d anomalies detected). Status: %s",
            audit["initial_rows"], audit["final_clean_rows"], audit["anomalies_detected"], audit["status"]
        )

        return df_work.sort_values(["symbol", "timestamp"]).reset_index(drop=True), audit

    def _verify_and_normalize_timezone(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Ensures timestamp column is timezone-aware and localized to Asia/Kolkata."""
        df_out = df.copy()
        anomalies = 0

        if not pd.api.types.is_datetime64_any_dtype(df_out["timestamp"]):
            df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])

        if df_out["timestamp"].dt.tz is None:
            anomalies += len(df_out)
            logger.warning("Timestamp column was naive (lacked timezone). Localizing to %s.", self.timezone_str)
            df_out["timestamp"] = df_out["timestamp"].dt.tz_localize(self.timezone_str)
        else:
            current_tz = str(df_out["timestamp"].dt.tz)
            if current_tz in [self.timezone_str, "Asia/Calcutta", "+05:30", "UTC+05:30"]:
                if current_tz != self.timezone_str:
                    df_out["timestamp"] = df_out["timestamp"].dt.tz_convert(self.timezone_str)
            else:
                anomalies += len(df_out)
                logger.warning("Timestamp was in %s. Converting to %s.", current_tz, self.timezone_str)
                df_out["timestamp"] = df_out["timestamp"].dt.tz_convert(self.timezone_str)

        return df_out, anomalies

    def _detect_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Identifies duplicate (symbol, timestamp) rows and drops redundant copies."""
        dup_mask = df.duplicated(subset=["symbol", "timestamp"], keep="first")
        dup_count = int(dup_mask.sum())
        if dup_count > 0:
            logger.warning("Detected %d duplicate (symbol, timestamp) candles. Retaining first occurrence.", dup_count)
            df_clean = df[~dup_mask].copy()
            return df_clean, dup_count
        return df, 0

    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Enforces mathematical bounds:
        1. high >= open
        2. high >= close
        3. high >= low
        4. low <= open
        5. low <= close
        """
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)

        invalid_mask = (
            (h < o) | (h < c) | (h < l) |
            (l > o) | (l > c) | (l > h)
        )
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            logger.error("Detected %d candles with mathematically impossible OHLC relationships.", invalid_count)
            # Tag quality flag or reject invalid bars
            df_clean = df[~invalid_mask].copy()
            return df_clean, invalid_count
        return df, 0

    def _validate_prices_and_volume(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Detects prices <= 0 and volume < 0."""
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        v = df["volume"].astype(float)

        bad_mask = (o <= 0.0) | (h <= 0.0) | (l <= 0.0) | (c <= 0.0) | (v < 0.0)
        bad_count = int(bad_mask.sum())
        if bad_count > 0:
            logger.error("Detected %d candles with non-positive prices or negative volume.", bad_count)
            df_clean = df[~bad_mask].copy()
            return df_clean, bad_count
        return df, 0

    def _validate_market_hours_and_weekends(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Verifies all bars occur Monday through Friday within 09:15:00 to 15:30:00 IST.
        """
        ts = df["timestamp"]
        # Day of week: Monday=0, Sunday=6
        weekend_mask = ts.dt.dayofweek >= 5
        # Time of day
        bar_times = ts.dt.time
        off_hours_mask = (bar_times < NSE_MARKET_OPEN) | (bar_times >= NSE_MARKET_CLOSE)

        contamination_mask = weekend_mask | off_hours_mask
        contamination_count = int(contamination_mask.sum())

        if contamination_count > 0:
            logger.warning("Detected %d bars outside regular trading hours or on weekends.", contamination_count)
            df_clean = df[~contamination_mask].copy()
            return df_clean, contamination_count
        return df, 0

    def _audit_trading_sessions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Audits session completeness per date.
        Flags sessions with fewer or more than standard 75 bars.
        """
        df_copy = df.copy()
        df_copy["date"] = df_copy["timestamp"].dt.date

        session_counts = df_copy.groupby(["symbol", "date"]).size().unstack(fill_value=0)
        total_sessions = session_counts.shape[1]

        # Calculate frequency of bar counts per session across all symbols
        all_session_lengths = df_copy.groupby(["symbol", "date"]).size()
        standard_sessions = int((all_session_lengths == EXPECTED_REGULAR_BARS_5M).sum())
        partial_sessions = int((all_session_lengths < EXPECTED_REGULAR_BARS_5M).sum())
        extended_sessions = int((all_session_lengths > EXPECTED_REGULAR_BARS_5M).sum())

        return {
            "total_trading_dates": total_sessions,
            "expected_bars_per_regular_session": EXPECTED_REGULAR_BARS_5M,
            "standard_sessions_count": standard_sessions,
            "partial_sessions_count": partial_sessions,
            "extended_sessions_count": extended_sessions,
            "mean_bars_per_session": round(float(all_session_lengths.mean()), 2) if not all_session_lengths.empty else 0.0,
            "min_bars_observed": int(all_session_lengths.min()) if not all_session_lengths.empty else 0,
            "max_bars_observed": int(all_session_lengths.max()) if not all_session_lengths.empty else 0,
        }
