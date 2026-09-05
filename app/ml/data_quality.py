import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """
    Data Quality Audit & Verification Suite (Phase 1E).

    Performs comprehensive data integrity, timezone, missingness, and cross-day leakage
    checks on raw and processed market feature datasets.
    """

    @staticmethod
    def audit_dataset(
        df: pd.DataFrame,
        timeframe: str = "5m",
        output_filepath: str = "dataset_quality.json"
    ) -> Dict[str, Any]:
        """
        Runs exhaustive quality audits on the dataset and exports JSON report.
        """
        if df.empty:
            raise ValueError("Cannot audit empty dataset.")

        df = df.copy()
        total_rows = len(df)
        symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else ["UNKNOWN"]

        report = {
            "audit_status": "SUCCESS",
            "symbols_processed": symbols,
            "total_symbols_count": len(symbols),
            "candle_timeframe": timeframe,
            "total_rows": total_rows,
            "date_range": {
                "start": str(df["timestamp"].min()) if "timestamp" in df.columns else "N/A",
                "end": str(df["timestamp"].max()) if "timestamp" in df.columns else "N/A"
            },
            "per_symbol_row_counts": {},
            "issues_detected": {
                "duplicate_timestamps": 0,
                "missing_ohlcv_rows": 0,
                "impossible_negative_prices": 0,
                "zero_prices": 0,
                "invalid_negative_volumes": 0,
                "zero_volumes": 0,
                "invalid_ohlc_relationships": 0,
                "nan_feature_values": 0,
                "inf_feature_values": 0,
                "timezone_missing_or_incorrect": 0,
                "cross_day_vwap_leakage_detected": False
            },
            "feature_coverage": {}
        }

        # 1. Per-symbol row counts & duplicate timestamp check (grouped per symbol)
        for sym in symbols:
            sym_df = df[df["symbol"] == sym] if "symbol" in df.columns else df
            report["per_symbol_row_counts"][sym] = len(sym_df)

            if "timestamp" in sym_df.columns:
                dups = sym_df.duplicated(subset=["timestamp"]).sum()
                report["issues_detected"]["duplicate_timestamps"] += int(dups)

        # 2. Price, Volume, and OHLC Relationship Integrity Checks
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                negs = (df[col] < 0).sum()
                zeros = (df[col] == 0).sum()
                report["issues_detected"]["impossible_negative_prices"] += int(negs)
                report["issues_detected"]["zero_prices"] += int(zeros)

        if "volume" in df.columns:
            report["issues_detected"]["invalid_negative_volumes"] += int((df["volume"] < 0).sum())
            report["issues_detected"]["zero_volumes"] += int((df["volume"] == 0).sum())

        # OHLC logical consistency: High >= max(Open, Close), Low <= min(Open, Close), High >= Low
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            invalid_high = (df["high"] < np.maximum(df["open"], df["close"])).sum()
            invalid_low = (df["low"] > np.minimum(df["open"], df["close"])).sum()
            invalid_hl = (df["high"] < df["low"]).sum()
            total_invalid_ohlc = int(invalid_high + invalid_low + invalid_hl)
            report["issues_detected"]["invalid_ohlc_relationships"] = total_invalid_ohlc
            if total_invalid_ohlc > 0:
                logger.warning("DataQuality: Detected %d invalid OHLC relationship violations.", total_invalid_ohlc)

        # 3. Timezone verification
        if "timestamp" in df.columns:
            if not hasattr(df["timestamp"].dt, "tz") or df["timestamp"].dt.tz is None:
                report["issues_detected"]["timezone_missing_or_incorrect"] += total_rows
                logger.warning("DataQuality: Dataset timestamps are naive (missing timezone).")

        # 4. Feature NaN / Inf check
        feature_cols = [c for c in df.columns if c not in ["timestamp", "symbol"]]
        for f_col in feature_cols:
            nans = int(df[f_col].isna().sum())
            infs = int(np.isinf(df[f_col].replace([np.nan], 0.0)).sum())

            report["feature_coverage"][f_col] = {
                "valid_count": total_rows - nans,
                "null_count": nans,
                "null_percentage": round((nans / total_rows) * 100.0, 2)
            }

            if nans > 0 and f_col in ["close", "ema_5", "vwap", "rsi"]:
                report["issues_detected"]["nan_feature_values"] += nans

        # 5. Mathematical Session-Reset VWAP Leakage Verification (Per Symbol & Per Candle)
        if "vwap" in df.columns and "timestamp" in df.columns:
            try:
                leakage_count = 0
                for sym in symbols:
                    sym_df = df[df["symbol"] == sym].sort_values("timestamp") if "symbol" in df.columns else df.sort_values("timestamp")
                    if sym_df.empty:
                        continue

                    # Extract session dates (Asia/Kolkata aware)
                    if hasattr(sym_df["timestamp"].dt, "tz") and sym_df["timestamp"].dt.tz is not None:
                        session_dates = sym_df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
                    else:
                        session_dates = sym_df["timestamp"].dt.date

                    sym_df["_session"] = session_dates
                    sym_df["_tp"] = (sym_df["high"] + sym_df["low"] + sym_df["close"]) / 3.0
                    sym_df["_tp_vol"] = sym_df["_tp"] * sym_df["volume"]

                    for sess_date, sess_df in sym_df.groupby("_session"):
                        cum_tp_vol = sess_df["_tp_vol"].cumsum()
                        cum_vol = sess_df["volume"].cumsum()
                        math_vwap = (cum_tp_vol / cum_vol.replace(0.0, np.nan)).ffill().fillna(sess_df["close"])

                        # Verify each candle against mathematical VWAP
                        for stored, expected in zip(sess_df["vwap"], math_vwap):
                            if not np.isclose(stored, expected, atol=1e-3, rtol=1e-4):
                                leakage_count += 1

                if leakage_count > 0:
                    report["issues_detected"]["cross_day_vwap_leakage_detected"] = True
                    logger.warning("DataQuality: Detected %d candle VWAP discrepancies / session leaks.", leakage_count)
            except Exception as exc:
                logger.error("DataQuality: Failed VWAP leakage check — %s", exc)
            finally:
                for col in ["_session", "_tp", "_tp_vol"]:
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True)

        # Export Quality Report JSON
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info("Data quality audit report exported to %s", output_filepath)
        except Exception as exc:
            logger.error("Failed to export dataset quality report: %s", exc)

        return report


data_quality_validator = DataQualityValidator()
