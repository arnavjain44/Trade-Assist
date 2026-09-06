"""
ML Prediction Engine for Live Inference

Connects the serialized Phase 5 D-LightGBM model artifact to live trading inference.
Strict requirements:
- Zero synthetic data, zero heuristic BUY/SELL rules, zero confidence inflating.
- Strictly evaluates 5-minute candles (rejects daily data).
- Evaluates direction=+1 and direction=-1 independently against threshold 0.8000.
- Unqualified candidates are assigned action="HOLD" with zero capital allocation.
"""

import logging
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from app.ml.phase5_inference import Phase5ProductionInference

logger = logging.getLogger(__name__)


class MLPredictionEngine:
    """
    Live production prediction engine backed strictly by the frozen Phase 5 D-LightGBM model.
    """

    def __init__(self, artifact_path: Optional[str] = None):
        if artifact_path:
            self.inference = Phase5ProductionInference(artifact_path)
        else:
            self.inference = Phase5ProductionInference()
        self.is_trained = True

    def predict_trade_signal(
        self,
        symbol: str,
        current_price: float,
        indicators: Dict[str, Any],
        sentiment_score_or_features: Union[float, Dict[str, Any]],
        vector_similarity_or_features: Union[float, Dict[str, Any]],
        timeframe: str = "5m",
    ) -> Dict[str, Any]:
        """
        Runs live 5-minute features through the serialized Phase 5 D-LightGBM model.

        Evaluates direction=+1 (LONG) and direction=-1 (SHORT) independently.
        Only candidates with P >= 0.8000 qualify for BUY or SELL.
        Unqualified candidates receive HOLD.

        Fails loudly if daily data is passed.
        """
        # Validate timeframe
        tf = str(indicators.get("timeframe", timeframe)).lower()
        if tf != "5m":
            raise ValueError(
                f"Phase 5 model strictly requires 5-minute intraday data. "
                f"Received timeframe='{tf}'. Daily candles cannot be fed to this model."
            )

        # Standardize technical indicators
        tech_dict = {
            "rsi": float(indicators["rsi"]),
            "obv": float(indicators["obv"]),
            "bollinger_position": float(indicators.get("bollinger_position", 0.5)),
            "macd": float(indicators["macd"]),
            "macd_signal": float(indicators["macd_signal"]),
            "macd_diff": float(indicators.get("macd_diff", indicators["macd"] - indicators["macd_signal"])),
            "price_vs_vwap": float(indicators.get("price_vs_vwap", 0.0)),
            "price_vs_ema5": float(indicators.get("price_vs_ema5", 0.0)),
        }

        # Standardize news features
        if isinstance(sentiment_score_or_features, dict):
            news_dict = sentiment_score_or_features
        else:
            s_val = float(sentiment_score_or_features) if sentiment_score_or_features is not None else float("nan")
            has_news = not np.isnan(s_val)
            news_dict = {
                "sentiment_score": s_val,
                "has_news": has_news,
                "number_of_articles": 1 if has_news else 0,
            }

        # Standardize context similarity features
        if isinstance(vector_similarity_or_features, dict):
            context_dict = vector_similarity_or_features
        else:
            v_val = float(vector_similarity_or_features) if vector_similarity_or_features is not None else 0.0
            context_dict = {
                "market_similarity": v_val,
                "stock_similarity": v_val,
            }

        # Evaluate both directions independently
        eval_res = self.inference.evaluate_dual_directions(
            technical_indicators=tech_dict,
            news_features=news_dict,
            context_features=context_dict,
            timeframe="5m",
        )

        action = eval_res["action"]
        qualified = eval_res["qualified"]
        raw_prob = eval_res["model_probability"]
        p_long = eval_res["p_long"]
        p_short = eval_res["p_short"]
        chosen_dir = eval_res["direction"]

        # Deterministic intraday targets (+2.2% / -0.9%) only for qualified BUY/SELL
        if action == "BUY":
            target_price = round(current_price * 1.022, 2)  # +2.2% profit target
            stop_loss = round(current_price * 0.991, 2)     # -0.9% stop loss
        elif action == "SELL":
            target_price = round(current_price * 0.978, 2)  # -2.2% profit target
            stop_loss = round(current_price * 1.009, 2)     # +0.9% stop loss
        else:
            # HOLD / Unqualified: targets equal current price
            target_price = round(current_price, 2)
            stop_loss = round(current_price, 2)

        # Confidence percentage is strictly the raw probability scaled to % (0.0 to 100.0)
        confidence_pct = round(raw_prob * 100.0, 2)

        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence_pct,
            "current_price": current_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "model_used": "phase5_d_lightgbm",
            "model_name": "phase5_d_lightgbm",
            "model_probability": raw_prob,
            "model_threshold": self.inference.threshold,
            "qualified": qualified,
            "direction": chosen_dir,
            "p_long": p_long,
            "p_short": p_short,
            "feature_summary": {
                "rsi": tech_dict["rsi"],
                "obv": tech_dict["obv"],
                "bollinger_position": tech_dict["bollinger_position"],
                "macd": tech_dict["macd"],
                "macd_signal": tech_dict["macd_signal"],
                "macd_diff": tech_dict["macd_diff"],
                "price_vs_vwap": tech_dict["price_vs_vwap"],
                "price_vs_ema5": tech_dict["price_vs_ema5"],
                "sentiment_score": news_dict.get("sentiment_score"),
                "market_similarity": context_dict.get("market_similarity"),
                "stock_similarity": context_dict.get("stock_similarity"),
            },
        }


ml_engine = MLPredictionEngine()
