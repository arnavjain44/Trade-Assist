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
                "nan_feature_values": 0,
                "inf_feature_values": 0,
                "timezone_missing_or_incorrect": 0,
                "cross_day_vwap_leakage_detected": False
            },
            "feature_coverage": {}
        }

        # 1. Per-symbol row counts & duplicate timestamp check
        for sym in symbols:
            sym_df = df[df["symbol"] == sym] if "symbol" in df.columns else df
            report["per_symbol_row_counts"][sym] = len(sym_df)

            if "timestamp" in sym_df.columns:
                dups = sym_df.duplicated(subset=["timestamp"]).sum()
                report["issues_detected"]["duplicate_timestamps"] += int(dups)

        # 2. Price and Volume Integrity Checks
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                negs = (df[col] < 0).sum()
                zeros = (df[col] == 0).sum()
                report["issues_detected"]["impossible_negative_prices"] += int(negs)
                report["issues_detected"]["zero_prices"] += int(zeros)

        if "volume" in df.columns:
            report["issues_detected"]["invalid_negative_volumes"] += int((df["volume"] < 0).sum())
            report["issues_detected"]["zero_volumes"] += int((df["volume"] == 0).sum())

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

        # 5. Cross-day VWAP Leakage Verification
        # Check that VWAP at the start of a new trading day equals the typical price of that candle
        if "vwap" in df.columns and "timestamp" in df.columns:
            try:
                df["_date"] = df["timestamp"].dt.date
                session_starts = df.groupby(["symbol", "_date"]).first().reset_index()

                leakage_count = 0
                for _, row in session_starts.iterrows():
                    typical_price = (row["high"] + row["low"] + row["close"]) / 3.0
                    # At the start of a day, VWAP must equal typical price (within tolerance)
                    if abs(row["vwap"] - typical_price) > 0.5 and row["volume"] > 0:
                        leakage_count += 1

                if leakage_count > 0:
                    report["issues_detected"]["cross_day_vwap_leakage_detected"] = True
                    logger.warning("DataQuality: Detected %d session starts with potential VWAP leakage.", leakage_count)
            except Exception as exc:
                logger.error("DataQuality: Failed VWAP leakage check — %s", exc)
            finally:
                if "_date" in df.columns:
                    df.drop(columns=["_date"], inplace=True)

        # Export Quality Report JSON
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info("Data quality audit report exported to %s", output_filepath)
        except Exception as exc:
            logger.error("Failed to export dataset quality report: %s", exc)

        return report


data_quality_validator = DataQualityValidator()
