"""
Phase 5 Feature Builder Module

Constructs the exact 14-dimensional feature vector for Phase 5 LightGBM inference
with strict mathematical, causal, and ordering parity with the Phase 5 research dataset.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

# The canonical 14 Phase 5 features in exact sequence
PHASE5_FEATURE_COLS: List[str] = [
    "rsi",
    "obv",
    "bollinger_position",
    "macd",
    "macd_signal",
    "macd_diff",
    "price_vs_vwap",
    "price_vs_ema5",
    "direction",
    "sentiment_score",
    "has_news",
    "number_of_articles",
    "market_similarity",
    "stock_similarity",
]


def build_phase5_feature_row(
    technical_indicators: Dict[str, Any],
    news_features: Dict[str, Any],
    context_features: Dict[str, Any],
    direction: int,
) -> Dict[str, Any]:
    """
    Constructs a single feature dictionary conforming strictly to the Phase 5 schema.

    Args:
        technical_indicators: Dict with rsi, obv, bollinger_position, macd, macd_signal,
                              macd_diff, price_vs_vwap, price_vs_ema5.
        news_features: Dict with sentiment_score (float or NaN), has_news (bool),
                       number_of_articles (int).
        context_features: Dict with market_similarity (float), stock_similarity (float).
        direction: Integer direction (+1 for LONG, -1 for SHORT).

    Returns:
        Dict with exactly the 14 Phase 5 features in canonical sequence.
    """
    if direction not in (1, -1):
        raise ValueError(f"Invalid direction: {direction}. Must be exactly +1 (LONG) or -1 (SHORT).")

    raw_sentiment = news_features.get("sentiment_score")
    if raw_sentiment is None or (isinstance(raw_sentiment, float) and np.isnan(raw_sentiment)):
        sentiment_val = float("nan")
    else:
        sentiment_val = float(raw_sentiment)

    has_news_val = bool(news_features.get("has_news", False))
    num_articles_val = int(news_features.get("number_of_articles", 0))

    row = {
        "rsi": float(technical_indicators["rsi"]),
        "obv": float(technical_indicators["obv"]),
        "bollinger_position": float(technical_indicators["bollinger_position"]),
        "macd": float(technical_indicators["macd"]),
        "macd_signal": float(technical_indicators["macd_signal"]),
        "macd_diff": float(technical_indicators["macd_diff"]),
        "price_vs_vwap": float(technical_indicators["price_vs_vwap"]),
        "price_vs_ema5": float(technical_indicators["price_vs_ema5"]),
        "direction": int(direction),
        "sentiment_score": sentiment_val,
        "has_news": has_news_val,
        "number_of_articles": num_articles_val,
        "market_similarity": float(context_features.get("market_similarity", 0.0)),
        "stock_similarity": float(context_features.get("stock_similarity", 0.0)),
    }

    return row


def build_phase5_feature_dataframe(
    technical_indicators: Dict[str, Any],
    news_features: Dict[str, Any],
    context_features: Dict[str, Any],
    direction: int,
) -> pd.DataFrame:
    """Constructs a 1-row DataFrame with the 14 Phase 5 features in canonical column order."""
    row = build_phase5_feature_row(technical_indicators, news_features, context_features, direction)
    df = pd.DataFrame([row], columns=PHASE5_FEATURE_COLS)
    return df
