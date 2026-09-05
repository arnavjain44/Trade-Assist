import pandas as pd
import numpy as np
from typing import Dict, Any

class TechnicalIndicators:
    """Calculates the 6 curated technical indicators required for intraday analysis."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Applies 5 EMA, RSI, OBV, Bollinger Bands, MACD, and VWAP to the dataframe."""
        df = df.copy()

        # 1. 5 EMA (Trend)
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()

        # 2. RSI (Momentum - 9 period for intraday)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
        rs = gain / (loss.replace(0, np.nan))
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50.0)

        # 3. OBV (Volume)
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv

        # 4. Bollinger Bands (Volatility)
        bb_middle = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std().fillna(0)
        df['bb_middle'] = bb_middle
        df['bb_upper'] = bb_middle + (2 * bb_std)
        df['bb_lower'] = bb_middle - (2 * bb_std)

        # 5. MACD (Trend/Momentum - 12, 26, 9)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # 6. VWAP (Intraday Volume Weighted Average Price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        tp_volume = typical_price * df['volume']
        cum_tp_volume = tp_volume.cumsum()
        cum_volume = df['volume'].cumsum().replace(0, np.nan)
        df['vwap'] = (cum_tp_volume / cum_volume).fillna(df['close'])

        return df

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
