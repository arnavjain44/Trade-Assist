import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Unified, reusable feature engineering pipeline for BOTH historical dataset building
    and live inference. Guaranteed parity across training and inference.

    All feature calculations are strictly causal: for any row at timestamp T,
    only information available at or before timestamp T is utilized.

    Per-symbol isolation is strictly enforced: calculations never bleed across tickers.
    """

    @classmethod
    def calculate_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the required 6 technical indicators + normalized features.

        Input DataFrame must contain:
        - timestamp (datetime, timezone-aware IST/UTC or naive)
        - open, high, low, close, volume (floats)

        Returns DataFrame with added indicator and normalized feature columns.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        df = df.copy()

        # Ensure correct column casing
        df.columns = [c.lower() for c in df.columns]

        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for feature calculation: {missing}")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # If multiple symbols exist in the DataFrame, process each independently
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            results = []
            for _, group in df.groupby("symbol", sort=False):
                results.append(cls._calculate_single_symbol_features(group))
            return pd.concat(results, ignore_index=True)
        else:
            return cls._calculate_single_symbol_features(df)

    @staticmethod
    def _calculate_single_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Sort chronologically to prevent lookahead / ordering bugs
        df = df.sort_values("timestamp").reset_index(drop=True)

        # ----------------------------------------------------------------------
        # 1. EMA-5 (Short-term trend)
        # Formula: Exponential Moving Average over 5 periods on Close price.
        # Range: [0, +inf)
        # Causality: Causal (uses exponentially weighted past close prices).
        # ----------------------------------------------------------------------
        df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()

        # ----------------------------------------------------------------------
        # 2. RSI-9 (Intraday Momentum)
        # Formula: 100 - (100 / (1 + RS)), where RS = Avg Gain / Avg Loss over 9 periods.
        # Range: [0, 100]
        # Causality: Causal (uses rolling 9 past price diffs).
        # ----------------------------------------------------------------------
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=9, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=9, min_periods=1).mean()
        rs = gain / (loss.replace(0.0, np.nan))
        df["rsi"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

        # ----------------------------------------------------------------------
        # 3. OBV (On-Balance Volume)
        # Formula: Cumulative volume addition on up-candles, subtraction on down-candles.
        # Range: (-inf, +inf)
        # Causality: Causal (cumulative sum of past signed volume).
        # ----------------------------------------------------------------------
        sign = np.sign(df["close"].diff().fillna(0.0))
        df["obv"] = (sign * df["volume"]).cumsum()

        # ----------------------------------------------------------------------
        # 4. Bollinger Bands (20-period, 2 StdDev)
        # Formula: Middle = 20 SMA, Upper/Lower = Middle +/- 2 * 20 StdDev.
        # Range: [0, +inf)
        # Causality: Causal (rolling 20 past close prices).
        # ----------------------------------------------------------------------
        bb_middle = df["close"].rolling(window=20, min_periods=1).mean()
        bb_std = df["close"].rolling(window=20, min_periods=1).std().fillna(0.0)
        df["bollinger_middle"] = bb_middle
        df["bollinger_upper"] = bb_middle + (2.0 * bb_std)
        df["bollinger_lower"] = bb_middle - (2.0 * bb_std)

        # ----------------------------------------------------------------------
        # 5. MACD (12, 26, 9)
        # Formula: MACD Line = 12 EMA - 26 EMA; Signal = 9 EMA of MACD; Diff = MACD - Signal.
        # Range: (-inf, +inf)
        # Causality: Causal (uses past EMAs).
        # ----------------------------------------------------------------------
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_diff"] = df["macd"] - df["macd_signal"]

        # ----------------------------------------------------------------------
        # 6. VWAP (Intraday Volume Weighted Average Price — SESSION RESET)
        # Formula: Sum(Typical Price * Volume) / Sum(Volume) per session date.
        # RESETS AT THE START OF EVERY TRADING DAY (Asia/Kolkata IST).
        # Range: [0, +inf)
        # Causality: Causal (resets per session date; strictly intraday).
        # ----------------------------------------------------------------------
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        tp_volume = typical_price * df["volume"]

        # Extract session date (converting timezone-aware timestamps to Asia/Kolkata)
        if hasattr(df["timestamp"].dt, "tz") and df["timestamp"].dt.tz is not None:
            session_dates = df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
        else:
            session_dates = df["timestamp"].dt.date

        df["_session_date"] = session_dates
        df["_tp_vol"] = tp_volume

        cum_tp_vol = df.groupby("_session_date")["_tp_vol"].cumsum()
        cum_vol = df.groupby("_session_date")["volume"].cumsum()

        raw_vwap = cum_tp_vol / cum_vol.replace(0.0, np.nan)
        df["_raw_vwap"] = raw_vwap
        df["vwap"] = df.groupby("_session_date")["_raw_vwap"].ffill().fillna(df["close"])

        # Clean up temporary helper columns
        df.drop(columns=["_session_date", "_tp_vol", "_raw_vwap"], inplace=True)

        # ----------------------------------------------------------------------
        # Normalized Features (Derived from the 6 base indicators)
        # ----------------------------------------------------------------------
        # Bollinger Position: 0.0 = at lower band, 1.0 = at upper band
        bb_range = (df["bollinger_upper"] - df["bollinger_lower"]).replace(0.0, np.nan)
        df["bollinger_position"] = ((df["close"] - df["bollinger_lower"]) / bb_range).fillna(0.5)

        # Price vs VWAP % difference
        df["price_vs_vwap"] = ((df["close"] - df["vwap"]) / df["close"].replace(0.0, np.nan)).fillna(0.0)

        # Price vs EMA-5 % difference
        df["price_vs_ema5"] = ((df["close"] - df["ema_5"]) / df["close"].replace(0.0, np.nan)).fillna(0.0)

        # Placeholder schema columns (FinBERT sentiment & Historical vector similarity)
        # Explicitly set to None / NaN until real historical models ingest them in future phases.
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = np.nan
        if "market_similarity" not in df.columns:
            df["market_similarity"] = np.nan
        if "stock_similarity" not in df.columns:
            df["stock_similarity"] = np.nan

        return df


feature_engine = FeatureEngine()
