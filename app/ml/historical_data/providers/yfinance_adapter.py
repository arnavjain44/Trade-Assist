"""
Yahoo Finance Historical Data Adapter (Phase 5.4)

Implements BaseHistoricalProvider for yfinance.
Explicitly enforces the verified 60-day lookback limit for 5m intraday data.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import pandas as pd
import yfinance as yf

from app.ml.historical_data.base import BaseHistoricalProvider
from app.ml.historical_data.models import (
    HistoricalRangeExceededError,
    HistoricalDataError,
)

logger = logging.getLogger(__name__)


class YFinanceHistoricalProvider(BaseHistoricalProvider):
    """
    Adapter for Yahoo Finance (yfinance).
    Fails with HistoricalRangeExceededError if requests exceed the 60-day intraday limit.
    """

    @property
    def provider_name(self) -> str:
        return "yfinance"

    @property
    def max_chunk_days(self) -> int:
        """yfinance caps 5m intraday data at 60 calendar days maximum."""
        return 60

    def is_authenticated(self) -> bool:
        """yfinance does not require authentication."""
        return True

    def validate_credentials(self) -> None:
        """No credentials needed for yfinance."""
        pass

    def fetch_chunk(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """
        Fetches historical data chunk via yfinance.
        Enforces 60-day limit check.
        """
        # Calculate lookback age
        now = datetime.now(timezone.utc) if start_date.tzinfo else datetime.now()
        age_days = (now - start_date).days

        if interval in ["1m", "2m", "5m", "15m", "30m"] and age_days > 59:
            raise HistoricalRangeExceededError(
                f"yfinance cannot supply {interval} data older than 60 days (requested start {start_date.isoformat()} is {age_days} days old). "
                "Verified Yahoo Finance server-side rejection. An authorized multi-year provider (Kite/TrueData/Archive) is required."
            )

        try:
            ticker = yf.Ticker(symbol)
            # Use start/end or period
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
            )

            if df is None or df.empty:
                logger.warning("yfinance returned empty DataFrame for %s (%s to %s)", symbol, start_date, end_date)
                return pd.DataFrame()

            df = df.reset_index()
            return df

        except Exception as exc:
            raise HistoricalDataError(f"yfinance fetch failed for {symbol}: {exc}")

    def normalize_to_canonical(self, raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalizes yfinance DataFrame into canonical schema."""
        if raw_df.empty:
            return pd.DataFrame(columns=[
                "timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"
            ])

        df = raw_df.copy()

        # Identify timestamp column
        ts_col = None
        for candidate in ["Datetime", "datetime", "Date", "date"]:
            if candidate in df.columns:
                ts_col = candidate
                break

        if not ts_col:
            raise HistoricalDataError(f"No timestamp column found in yfinance output: {list(df.columns)}")

        df["timestamp"] = pd.to_datetime(df[ts_col])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

        df["open"] = df["Open"].astype(float)
        df["high"] = df["High"].astype(float)
        df["low"] = df["Low"].astype(float)
        df["close"] = df["Close"].astype(float)
        df["volume"] = df["Volume"].astype(float)

        df["symbol"] = symbol
        df["trading_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        df["source"] = self.provider_name

        canonical_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"]
        return df[canonical_cols].sort_values("timestamp").reset_index(drop=True)
