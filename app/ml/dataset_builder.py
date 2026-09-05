import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Any, Optional, Tuple
from app.config import settings
from app.ml.feature_engineering import feature_engine

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Configurable historical market data ingestion and dataset processing pipeline (Phase 1A).

    Capable of fetching multi-day intraday candles (e.g. 5m, 15m, 1h) or daily candles,
    validating timezone-aware timestamps, enforcing zero lookahead bias, and applying
    the unified FeatureEngine pipeline.
    """

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def verify_provider_capabilities(self, timeframe: str = "5m", period: str = "60d") -> bool:
        """
        Verifies if the configured provider (yfinance) can supply intraday candles.
        yfinance supports max 60 days of 5m/15m intraday data.
        """
        if timeframe in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
            logger.info("Provider capability check: yfinance supports intraday %s for period up to %s.", timeframe, period)
            return True
        elif timeframe in ["1d", "5d", "1wk", "1mo"]:
            logger.info("Provider capability check: yfinance supports daily %s.", timeframe)
            return True
        else:
            logger.warning("Unrecognized timeframe %s. Defaulting to 5m capability check.", timeframe)
            return False

    def fetch_symbol_raw_history(
        self,
        symbol: str,
        timeframe: str = "5m",
        period: str = "60d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical raw OHLCV series for a single ticker via yfinance.
        Ensures timezone-aware timestamps (Asia/Kolkata / IST).
        Does NOT fabricate missing prices.
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=timeframe)

            if df is None or df.empty:
                logger.warning("No historical data returned for symbol %s (timeframe=%s, period=%s).", symbol, timeframe, period)
                return None

            # Reset index to extract timestamp
            df = df.reset_index()

            # Identify timestamp column
            ts_col = None
            for candidate in ["Datetime", "datetime", "Date", "date"]:
                if candidate in df.columns:
                    ts_col = candidate
                    break

            if not ts_col:
                logger.error("No timestamp column identified for %s. Available: %s", symbol, list(df.columns))
                return None

            # Standardize column naming
            df["timestamp"] = pd.to_datetime(df[ts_col])

            # Localize / convert to Asia/Kolkata timezone
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

            df["symbol"] = symbol
            df["open"] = df["Open"].astype(float)
            df["high"] = df["High"].astype(float)
            df["low"] = df["Low"].astype(float)
            df["close"] = df["Close"].astype(float)
            df["volume"] = df["Volume"].astype(float)

            # Keep canonical raw columns
            raw_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            df_raw = df[raw_cols].sort_values("timestamp").reset_index(drop=True)

            return df_raw

        except Exception as exc:
            logger.error("Failed to fetch history for %s: %s", symbol, exc)
            return None

    def build_dataset(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "5m",
        period: str = "60d"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds both raw and processed feature datasets across the configured universe of symbols.

        Returns Tuple[raw_combined_df, processed_combined_df].
        """
        target_symbols = symbols or settings.DEFAULT_NSE_TICKERS[:10]  # Default to sample universe
        self.verify_provider_capabilities(timeframe, period)

        raw_dfs = []
        processed_dfs = []

        logger.info("Starting historical dataset build for %d symbols...", len(target_symbols))

        for sym in target_symbols:
            logger.info("Ingesting historical OHLCV for %s (%s, %s)...", sym, timeframe, period)
            raw_df = self.fetch_symbol_raw_history(sym, timeframe=timeframe, period=period)

            if raw_df is None or raw_df.empty:
                logger.warning("Skipping %s — no historical market candles available.", sym)
                continue

            raw_dfs.append(raw_df)

            # Save raw ticker file to disk
            sym_clean = sym.replace(".NS", "")
            raw_filepath = os.path.join(self.raw_dir, f"{sym_clean}_raw_{timeframe}.parquet")
            raw_df.to_parquet(raw_filepath, index=False)

            # Process features using unified FeatureEngine
            try:
                proc_df = feature_engine.calculate_features(raw_df)
                processed_dfs.append(proc_df)

                proc_filepath = os.path.join(self.processed_dir, f"{sym_clean}_processed_{timeframe}.parquet")
                proc_df.to_parquet(proc_filepath, index=False)
            except Exception as exc:
                logger.error("Feature calculation failed for %s: %s", sym, exc)

        if not raw_dfs:
            raise RuntimeError("Failed to build dataset: No symbols produced valid raw market data.")

        combined_raw = pd.concat(raw_dfs, ignore_index=True)
        combined_processed = pd.concat(processed_dfs, ignore_index=True)

        # Save combined datasets
        combined_raw.to_parquet(os.path.join(self.raw_dir, "combined_raw.parquet"), index=False)
        combined_processed.to_parquet(os.path.join(self.processed_dir, "combined_processed.parquet"), index=False)

        logger.info(
            "Dataset build complete. Raw rows: %d | Processed rows: %d | Total symbols: %d",
            len(combined_raw), len(combined_processed), len(processed_dfs)
        )

        return combined_raw, combined_processed


dataset_builder = DatasetBuilder()
