import pandas as pd
import numpy as np
from typing import Dict, Any
from app.ml.feature_engineering import feature_engine

class TechnicalIndicators:
    """Calculates the 6 curated technical indicators required for intraday analysis."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Applies 5 EMA, RSI, OBV, Bollinger Bands, MACD, and VWAP (with session reset)."""
        df_copy = df.copy()
        col_map = {str(c).lower(): c for c in df_copy.columns}

        if "timestamp" not in col_map:
            if "date" in col_map:
                df_copy["timestamp"] = pd.to_datetime(df_copy[col_map["date"]])
            elif "datetime" in col_map:
                df_copy["timestamp"] = pd.to_datetime(df_copy[col_map["datetime"]])
            elif "date_str" in col_map and not str(df_copy[col_map["date_str"]].iloc[0]).isdigit():
                df_copy["timestamp"] = pd.to_datetime(df_copy[col_map["date_str"]])
            else:
                df_copy["timestamp"] = pd.to_datetime(df_copy.index)

        if "date_str" not in col_map or str(df_copy[col_map.get("date_str", "")].iloc[0] if "date_str" in col_map else "").isdigit():
            df_copy["date_str"] = pd.to_datetime(df_copy["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

        df_feat = feature_engine.calculate_features(df_copy)
        # Ensure backward compatibility aliases
        df_feat["bb_middle"] = df_feat["bollinger_middle"]
        df_feat["bb_upper"] = df_feat["bollinger_upper"]
        df_feat["bb_lower"] = df_feat["bollinger_lower"]
        return df_feat

    @staticmethod
    def extract_latest_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Extracts the latest numeric values and signals from the indicators dataframe."""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        rsi_signal = "NEUTRAL"
        if latest['rsi'] > 70:
            rsi_signal = "OVERBOUGHT"
        elif latest['rsi'] < 30:
            rsi_signal = "OVERSOLD"

        ema_signal = "BULLISH" if latest['close'] > latest['ema_5'] else "BEARISH"
        macd_signal = "BULLISH" if latest['macd'] > latest['macd_signal'] else "BEARISH"
        vwap_signal = "BULLISH" if latest['close'] > latest['vwap'] else "BEARISH"

        # Deterministic 4-indicator majority vote calculation
        ema_score  = 1 if latest['close'] > latest['ema_5'] else -1
        vwap_score = 1 if latest['close'] > latest['vwap'] else -1
        macd_score = 1 if latest['macd'] > latest['macd_signal'] else -1
        rsi_score  = 1 if latest['rsi'] >= 50 else -1

        total_score = ema_score + vwap_score + macd_score + rsi_score
        if total_score >= 2:
            overall_bias = "BULLISH"
        elif total_score <= -2:
            overall_bias = "BEARISH"
        else:
            overall_bias = "NEUTRAL"

        return {
            "close_price": round(float(latest['close']), 2),
            "ema_5": round(float(latest['ema_5']), 2),
            "rsi": round(float(latest['rsi']), 2),
            "rsi_signal": rsi_signal,
            "obv": float(latest['obv']),
            "bb_upper": round(float(latest['bb_upper']), 2),
            "bb_lower": round(float(latest['bb_lower']), 2),
            "bb_middle": round(float(latest['bb_middle']), 2),
            "macd": round(float(latest['macd']), 2),
            "macd_signal": round(float(latest['macd_signal']), 2),
            "vwap": round(float(latest['vwap']), 2),
            "overall_technical_bias": overall_bias
        }


indicators_calculator = TechnicalIndicators()
