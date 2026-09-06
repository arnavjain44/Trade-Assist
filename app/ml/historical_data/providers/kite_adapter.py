"""
Zerodha Kite Connect Historical Data Adapter (Phase 5.4)

Implements BaseHistoricalProvider for Zerodha Kite Connect Historical API.
API Reference: https://kite.trade/docs/connect/v3/historical/
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd

from app.ml.historical_data.base import BaseHistoricalProvider
from app.ml.historical_data.models import (
    AuthenticationRequiredError,
    HistoricalDataError,
    RateLimitExceededError,
)

logger = logging.getLogger(__name__)

# NSE Instrument token mapping cache for Nifty 50 equities (Kite requires integer instrument_tokens)
# These are the official Zerodha NSE instrument tokens for major constituents
DEFAULT_KITE_INSTRUMENT_TOKENS: Dict[str, int] = {
    "RELIANCE.NS": 738561,
    "TCS.NS": 2953217,
    "INFY.NS": 408065,
    "HDFCBANK.NS": 341249,
    "ICICIBANK.NS": 1270529,
    "SBIN.NS": 779521,
    "BHARTIARTL.NS": 2714625,
    "ITC.NS": 424961,
    "KOTAKBANK.NS": 492033,
    "LT.NS": 2939649,
    "AXISBANK.NS": 1510401,
    "HINDUNILVR.NS": 356865,
    "BAJFINANCE.NS": 81153,
    "MARUTI.NS": 2815745,
    "TATASTEEL.NS": 895745,
    "ASIANPAINT.NS": 60417,
    "TITAN.NS": 897281,
    "SUNPHARMA.NS": 857857,
    "WIPRO.NS": 969473,
    "HCLTECH.NS": 1850625,
}


class KiteHistoricalProvider(BaseHistoricalProvider):
    """
    Adapter for Zerodha Kite Connect Historical Data API.
    Supports continuous 5-minute historical data with 100-day chunked pagination.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        instrument_tokens: Optional[Dict[str, int]] = None,
    ):
        self.api_key = api_key or os.getenv("HISTORICAL_DATA_API_KEY")
        self.access_token = access_token or os.getenv("HISTORICAL_DATA_ACCESS_TOKEN")
        self.instrument_tokens = instrument_tokens or DEFAULT_KITE_INSTRUMENT_TOKENS
        self._kite_client = None

    @property
    def provider_name(self) -> str:
        return "kite"

    @property
    def max_chunk_days(self) -> int:
        """Kite Historical API strictly caps 5-minute interval requests at 100 days per call."""
        return 100

    def is_authenticated(self) -> bool:
        """Checks if both api_key and active access_token are present."""
        return bool(self.api_key and self.access_token)

    def validate_credentials(self) -> None:
        """Fails loudly and safely if credentials are not configured."""
        if not self.is_authenticated():
            raise AuthenticationRequiredError(
                "Zerodha Kite Connect credentials not configured. "
                "Please set HISTORICAL_DATA_API_KEY and HISTORICAL_DATA_ACCESS_TOKEN in .env. "
                "Zero synthetic data will be fabricated."
            )

    def _get_instrument_token(self, symbol: str) -> int:
        """Resolves symbol to Kite instrument token."""
        clean_sym = symbol.strip().upper()
        if not clean_sym.endswith(".NS"):
            clean_sym = f"{clean_sym}.NS"

        if clean_sym in self.instrument_tokens:
            return self.instrument_tokens[clean_sym]
        raise HistoricalDataError(f"No Kite instrument token mapped for symbol '{symbol}'.")

    def fetch_chunk(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5minute",
    ) -> pd.DataFrame:
        """
        Fetches historical data chunk from Kite Connect API.
        """
        self.validate_credentials()

        # Map interval string to Kite format ('5m' -> '5minute')
        kite_interval = "5minute" if interval in ["5m", "5minute"] else interval
        instrument_token = self._get_instrument_token(symbol)

        # Lazy initialize client
        if self._kite_client is None:
            try:
                from kiteconnect import KiteConnect
                self._kite_client = KiteConnect(api_key=self.api_key)
                self._kite_client.set_access_token(self.access_token)
            except ImportError:
                raise HistoricalDataError(
                    "kiteconnect package is not installed. Install via 'pip install kiteconnect'."
                )

        try:
            records = self._kite_client.historical_data(
                instrument_token=instrument_token,
                from_date=start_date.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=end_date.strftime("%Y-%m-%d %H:%M:%S"),
                interval=kite_interval,
                continuous=False,
                oi=False,
            )

            if not records:
                logger.warning("Kite returned 0 records for %s between %s and %s", symbol, start_date, end_date)
                return pd.DataFrame()

            df = pd.DataFrame(records)
            return df

        except Exception as exc:
            err_msg = str(exc)
            if "TokenException" in err_msg or "403" in err_msg:
                raise AuthenticationRequiredError(f"Kite session expired or invalid: {err_msg}")
            elif "NetworkException" in err_msg or "Too many requests" in err_msg:
                raise RateLimitExceededError(f"Kite rate limit hit: {err_msg}")
            else:
                raise HistoricalDataError(f"Kite historical fetch error for {symbol}: {err_msg}")

    def normalize_to_canonical(self, raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalizes Kite raw output into Trade-Assist canonical schema.

        CRITICAL TEMPORAL CONVERSION:
        - Kite returns the candle START timestamp (e.g. 09:15:00 for the 09:15-09:20 bar).
        - Trade-Assist canonical schema standardizes on the candle CLOSE/END timestamp (09:20:00).
        - We explicitly shift the timestamp forward by 5 minutes:
          canonical_timestamp = raw_timestamp + 5 minutes
        """
        if raw_df.empty:
            return pd.DataFrame(columns=[
                "timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"
            ])

        df = raw_df.copy()

        # Identify timestamp column
        ts_col = "date" if "date" in df.columns else "timestamp"
        if ts_col not in df.columns:
            raise HistoricalDataError(f"Kite DataFrame missing timestamp column. Columns: {list(df.columns)}")

        # Ensure timezone-aware IST (UTC+05:30)
        df["raw_ts"] = pd.to_datetime(df[ts_col])
        if df["raw_ts"].dt.tz is None:
            df["raw_ts"] = df["raw_ts"].dt.tz_localize("Asia/Kolkata")
        else:
            df["raw_ts"] = df["raw_ts"].dt.tz_convert("Asia/Kolkata")

        # Causal temporal conversion: start time -> close/end time (+5 minutes for 5m candles)
        df["timestamp"] = df["raw_ts"] + pd.Timedelta(minutes=5)

        # Standardize OHLCV
        col_map = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                col_map[col.lower()] = col

        df["open"] = df["open" if "open" in df.columns else "Open"].astype(float)
        df["high"] = df["high" if "high" in df.columns else "High"].astype(float)
        df["low"] = df["low" if "low" in df.columns else "Low"].astype(float)
        df["close"] = df["close" if "close" in df.columns else "Close"].astype(float)
        df["volume"] = df["volume" if "volume" in df.columns else "Volume"].astype(float)

        df["symbol"] = symbol
        df["trading_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        df["source"] = self.provider_name

        canonical_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "trading_date", "source"]
        return df[canonical_cols].sort_values("timestamp").reset_index(drop=True)
