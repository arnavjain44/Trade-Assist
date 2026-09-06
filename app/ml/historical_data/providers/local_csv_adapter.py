"""
Local CSV & Parquet Archive Historical Adapter (Phase 5.4)

Implements BaseHistoricalProvider for offline research archives (e.g. Kaggle/GitHub datasets).
"""

import os
import glob
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd

from app.ml.historical_data.base import BaseHistoricalProvider
from app.ml.historical_data.models import HistoricalDataError

logger = logging.getLogger(__name__)


class LocalCsvHistoricalProvider(BaseHistoricalProvider):
    """
    Adapter for loading multi-year historical intraday datasets from local CSV or Parquet files.
    Enables free, offline quantitative research on pre-acquired archives.
    """

    def __init__(self, archive_dir: str = "data/raw"):
        self.archive_dir = os.path.abspath(archive_dir)

    @property
    def provider_name(self) -> str:
        return "local_csv"

    @property
    def max_chunk_days(self) -> int:
        """Local file ingestion does not have pagination limits; can ingest years at once."""
        return 3650

    def is_authenticated(self) -> bool:
        """Local file ingestion requires no credentials."""
        return True

    def validate_credentials(self) -> None:
        """No credentials needed."""
        pass

    def fetch_chunk(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """
        Loads matching file for symbol from archive directory and filters by date range.
        """
        clean_sym = symbol.replace(".NS", "").replace(".BO", "").replace("&", "_")

        # Look for matching parquet or csv files
        candidates = [
            os.path.join(self.archive_dir, f"{clean_sym}_raw_{interval}.parquet"),
            os.path.join(self.archive_dir, f"{clean_sym}_{interval}.csv"),
            os.path.join(self.archive_dir, f"{clean_sym}.csv"),
            os.path.join(self.archive_dir, f"{symbol}_raw_{interval}.parquet"),
        ]

        # Also search for any file containing the symbol
        matched_file = None
        for candidate in candidates:
            if os.path.exists(candidate):
                matched_file = candidate
                break

        if not matched_file:
            # Fallback search in directory
            pattern = os.path.join(self.archive_dir, f"*{clean_sym}*")
            matches = glob.glob(pattern)
            if matches:
                matched_file = matches[0]

        if not matched_file:
            logger.warning("No local archive file found for symbol %s in %s", symbol, self.archive_dir)
            return pd.DataFrame()

        # Load file
        if matched_file.endswith(".parquet"):
            df = pd.read_parquet(matched_file)
        else:
            df = pd.read_csv(matched_file)

        return df

    def normalize_to_canonical(self, raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalizes diverse column naming conventions into canonical schema."""
        if raw_df.empty:
            return pd.DataFrame(columns=[
                "timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"
            ])

        df = raw_df.copy()

        # Map column names (case-insensitive)
        cols_lower = {c.lower(): c for c in df.columns}

        # 1. Timestamp resolution
        ts_col = None
        for cand in ["timestamp", "datetime", "date_time", "date"]:
            if cand in cols_lower:
                ts_col = cols_lower[cand]
                break

        if "date" in cols_lower and "time" in cols_lower:
            # Split date and time columns
            df["timestamp"] = pd.to_datetime(df[cols_lower["date"]].astype(str) + " " + df[cols_lower["time"]].astype(str))
        elif ts_col:
            df["timestamp"] = pd.to_datetime(df[ts_col])
        else:
            raise HistoricalDataError(f"Cannot identify timestamp column in local file. Columns: {list(df.columns)}")

        # Timezone localization to Asia/Kolkata (IST)
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

        # 2. OHLCV resolution
        for std_name in ["open", "high", "low", "close", "volume"]:
            if std_name in cols_lower:
                df[std_name] = df[cols_lower[std_name]].astype(float)
            else:
                raise HistoricalDataError(f"Missing required price column '{std_name}' in local file.")

        df["symbol"] = symbol
        df["trading_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        df["source"] = self.provider_name

        canonical_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"]
        return df[canonical_cols].sort_values("timestamp").reset_index(drop=True)
