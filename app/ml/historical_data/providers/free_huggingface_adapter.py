"""
Free Hugging Face Historical NSE Intraday Provider Adapter (Phase 5.4a)

Implements BaseHistoricalProvider using open-source, permissive (MIT licensed)
NSE minute-level historical data from Hugging Face Hub (xxparthparekhxx/indian-stock-market-minute-data).
Provides 100% free (₹0 cost), unauthenticated access to 2022–2026 intraday data across 2,500+ NSE equities.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import fsspec

from app.ml.historical_data.base import BaseHistoricalProvider
from app.ml.historical_data.models import HistoricalDataError

logger = logging.getLogger(__name__)

# Shard alphabetical boundaries on Hugging Face Hub for xxparthparekhxx/indian-stock-market-minute-data
SHARD_BOUNDARIES: List[Tuple[str, str, int]] = [
    ("20MICRONS", "BOMDYEING", 0),
    ("BOMDYEING", "GANECOS", 1),
    ("GANECOS", "IVZINGOLD", 2),
    ("IVZINGOLD", "MOKSH", 3),
    ("MOKSH", "PHARMABEES", 4),
    ("PHARMABEES", "SMARTLINK", 5),
    ("SMARTLINK", "WELENT", 6),
    ("WELENT", "ZYDUSWELL", 7),
]

BASE_HF_URL = "https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data/resolve/main/minute/train-{:05d}.parquet"


class FreeHuggingFaceHistoricalProvider(BaseHistoricalProvider):
    """
    Adapter for free, open-source historical NSE 1m data hosted on Hugging Face Hub.
    Resamples high-frequency 1m candles into verified 5m canonical bars.
    Requires ZERO API keys, ZERO paid subscriptions, and ZERO authentication (100% ₹0 cost).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._fs = None

    @property
    def provider_name(self) -> str:
        return "free_huggingface"

    @property
    def max_chunk_days(self) -> int:
        """Can fetch multi-month or annual blocks efficiently."""
        return 365

    def is_authenticated(self) -> bool:
        """100% Free / Public dataset on Hugging Face Hub — no authentication required."""
        return True

    def validate_credentials(self) -> None:
        """No credentials needed for public MIT-licensed open dataset."""
        pass

    @staticmethod
    def get_shard_index(symbol: str) -> int:
        """Identifies which parquet shard (0-7) contains the symbol based on alphabetical sort."""
        clean_sym = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
        for s_min, s_max, shard_idx in SHARD_BOUNDARIES:
            if s_min <= clean_sym <= s_max:
                return shard_idx
        # Fallback to boundary edges
        if clean_sym < SHARD_BOUNDARIES[0][0]:
            return 0
        return 7

    def fetch_chunk(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """
        Fetches matching 1m candle rows from the appropriate Hugging Face shard via HTTP range requests,
        and resamples them into canonical 5-minute OHLCV bars.
        """
        clean_sym = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
        shard_idx = self.get_shard_index(clean_sym)
        url = BASE_HF_URL.format(shard_idx)

        logger.info("Connecting to Hugging Face Shard %d for %s (%s to %s)...", shard_idx, clean_sym, start_date, end_date)

        fs, path = fsspec.core.url_to_fs(url)
        pf = pq.ParquetFile(fs.open(path, "rb"))

        # Find row groups containing the symbol
        matched_rgs = []
        for i in range(pf.num_row_groups):
            rg = pf.metadata.row_group(i)
            s_min = rg.column(0).statistics.min
            s_max = rg.column(0).statistics.max
            if s_min <= clean_sym <= s_max:
                matched_rgs.append(i)

        if not matched_rgs:
            logger.warning("Symbol %s not found in Shard %d row groups.", clean_sym, shard_idx)
            return pd.DataFrame()

        dfs: List[pd.DataFrame] = []
        for rg_idx in matched_rgs:
            tbl = pf.read_row_group(rg_idx)
            mask = pc.equal(tbl["symbol"], clean_sym)
            sub = tbl.filter(mask)
            if len(sub) > 0:
                dfs.append(sub.to_pandas())

        if not dfs:
            logger.warning("No rows extracted for %s from matched row groups %s.", clean_sym, matched_rgs)
            return pd.DataFrame()

        df_1m = pd.concat(dfs, ignore_index=True)

        # Convert timestamp from UTC to Asia/Kolkata (IST)
        df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"])
        if df_1m["timestamp"].dt.tz is None:
            df_1m["timestamp"] = df_1m["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
        else:
            df_1m["timestamp"] = df_1m["timestamp"].dt.tz_convert("Asia/Kolkata")

        # Filter by requested date range (localized to IST)
        # Ensure start_date and end_date have timezone
        s_tz = start_date if start_date.tzinfo else start_date.replace(tzinfo=df_1m["timestamp"].dt.tz)
        e_tz = end_date if end_date.tzinfo else end_date.replace(tzinfo=df_1m["timestamp"].dt.tz)

        df_1m = df_1m[(df_1m["timestamp"] >= s_tz) & (df_1m["timestamp"] <= e_tz)]
        if df_1m.empty:
            logger.warning("Zero 1m rows match requested range %s to %s for %s.", start_date, end_date, clean_sym)
            return pd.DataFrame()

        df_1m = df_1m.sort_values("timestamp").reset_index(drop=True)

        # Resample 1m candles into 5m candles
        if interval in ["5m", "5minute"]:
            df_indexed = df_1m.set_index("timestamp")
            resampled = df_indexed.resample("5min", closed="left", label="right").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna().reset_index()
            resampled["symbol"] = f"{clean_sym}.NS"
            return resampled
        else:
            df_1m["symbol"] = f"{clean_sym}.NS"
            return df_1m

    def normalize_to_canonical(self, raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Transforms resampled DataFrame into canonical schema:
        - timestamp: IST (Asia/Kolkata / UTC+05:30) with candle close semantics
        - symbol: standardized ticker with .NS suffix
        - open, high, low, close: positive floats
        - volume: non-negative float
        - trading_date: YYYY-MM-DD
        - source: free_huggingface
        """
        if raw_df.empty:
            return pd.DataFrame(columns=[
                "timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"
            ])

        df = raw_df.copy()

        # Ensure timestamp is timezone-aware Asia/Kolkata
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

        clean_sym = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        df["symbol"] = clean_sym
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["trading_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        df["source"] = self.provider_name

        canonical_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"]
        return df[canonical_cols].sort_values("timestamp").reset_index(drop=True)
