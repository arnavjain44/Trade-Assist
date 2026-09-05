"""
Phase 5.1 Enhanced Causal Feature Engineering Module

Computes domain-informed causal features strictly using historical candle data (<= T)
and strictly prior news (< T):
1. Multi-timeframe momentum (return_5m, return_15m, return_60m)
2. Volatility & range dynamics (normalized_atr, bollinger_bandwidth)
3. Trend & moving average dynamics (ema5_slope, price_vs_ema20)
4. Momentum acceleration (rsi_delta_3)
5. Volume acceleration (relative_volume)
6. Intraday session timing (time_of_day_fraction, is_opening_session)
7. Context & news signals (sentiment_score, has_news, market_similarity, stock_similarity)

Guarantees zero future lookahead.
"""

import logging
from typing import List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Phase51FeatureEngine:
    """Computes strictly causal enhanced features on chronological 5m candle DataFrame."""

    BASE_TECHNICAL_FEATURES = [
        "rsi",
        "obv",
        "bollinger_position",
        "macd",
        "macd_signal",
        "macd_diff",
        "price_vs_vwap",
        "price_vs_ema5",
        "direction",
    ]

    BASE_NEWS_CONTEXT_FEATURES = [
        "sentiment_score",
        "has_news",
        "number_of_articles",
        "market_similarity",
        "stock_similarity",
    ]

    ENHANCED_FEATURE_COLS = [
        "return_5m",
        "return_15m",
        "return_60m",
        "normalized_atr",
        "bollinger_bandwidth",
        "ema5_slope",
        "price_vs_ema20",
        "rsi_delta_3",
        "relative_volume",
        "time_of_day_fraction",
        "is_opening_session",
    ]

    @classmethod
    def get_all_feature_cols(cls) -> List[str]:
        """Returns the full combined feature list for Phase 5.1 models."""
        return cls.BASE_TECHNICAL_FEATURES + cls.BASE_NEWS_CONTEXT_FEATURES + cls.ENHANCED_FEATURE_COLS

    @classmethod
    def compute_enhanced_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes enhanced causal features per symbol chronologically.
        Expects df to contain OHLCV and timestamps.
        """
        if df.empty:
            return df.copy()

        df_out = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_out["timestamp"]):
            df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])

        # Sort chronologically
        df_out = df_out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        # Process per symbol to maintain isolation
        enhanced_dfs = []
        for sym, group in df_out.groupby("symbol", sort=False):
            g = group.copy().sort_values("timestamp")

            # 1. Multi-timeframe returns
            c = g["close"].astype(float)
            g["return_5m"] = (c / c.shift(1) - 1.0).fillna(0.0)
            g["return_15m"] = (c / c.shift(3) - 1.0).fillna(0.0)
            g["return_60m"] = (c / c.shift(12) - 1.0).fillna(0.0)

            # 2. Normalized ATR (14-period)
            h = g["high"].astype(float)
            l = g["low"].astype(float)
            c_prev = c.shift(1).fillna(c)
            tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
            atr_14 = tr.rolling(14, min_periods=1).mean()
            g["normalized_atr"] = (atr_14 / np.maximum(c, 1e-6)).fillna(0.0)

            # 3. Bollinger Bandwidth
            if "bollinger_upper" in g.columns and "bollinger_lower" in g.columns and "bollinger_middle" in g.columns:
                bw = (g["bollinger_upper"] - g["bollinger_lower"]) / np.maximum(g["bollinger_middle"], 1e-6)
                g["bollinger_bandwidth"] = bw.fillna(0.0)
            else:
                g["bollinger_bandwidth"] = 0.0

            # 4. Trend Dynamics: EMA5 slope & EMA20
            if "ema_5" in g.columns:
                ema5 = g["ema_5"].astype(float)
                g["ema5_slope"] = (ema5 / ema5.shift(3).fillna(ema5) - 1.0).fillna(0.0)
            else:
                ema5 = c.ewm(span=5, adjust=False).mean()
                g["ema5_slope"] = (ema5 / ema5.shift(3).fillna(ema5) - 1.0).fillna(0.0)

            ema20 = c.ewm(span=20, adjust=False).mean()
            g["price_vs_ema20"] = ((c - ema20) / np.maximum(ema20, 1e-6)).fillna(0.0)

            # 5. RSI Momentum (3-candle change)
            if "rsi" in g.columns:
                rsi = g["rsi"].astype(float)
                g["rsi_delta_3"] = (rsi - rsi.shift(3)).fillna(0.0)
            else:
                g["rsi_delta_3"] = 0.0

            # 6. Relative Volume (volume / 20-period rolling mean)
            v = g["volume"].astype(float)
            v_mean_20 = v.rolling(20, min_periods=1).mean()
            g["relative_volume"] = (v / np.maximum(v_mean_20, 1.0)).fillna(1.0)

            # 7. Session Timing (09:15 to 15:30 -> 375 minutes total session)
            ts = g["timestamp"]
            session_minutes = (ts.dt.hour - 9) * 60 + (ts.dt.minute - 15)
            g["time_of_day_fraction"] = np.clip(session_minutes / 375.0, 0.0, 1.0)
            g["is_opening_session"] = (session_minutes <= 45).astype(float)

            enhanced_dfs.append(g)

        result_df = pd.concat(enhanced_dfs, ignore_index=True)
        # Restore sort order
        result_df = result_df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Computed %d enhanced causal features across %d rows.", len(cls.ENHANCED_FEATURE_COLS), len(result_df))
        return result_df
