"""
Phase 5 Production Inference Engine

Dedicated production inference module for the serialized Phase 5 D-LightGBM model.
Guarantees:
1. Loads frozen artifact strictly from disk; fails loudly if missing or invalid.
2. Never retrains, bootstraps, or fits preprocessing at startup or runtime.
3. Preserves exact 14-feature sequence matching the Phase 5 research dataset.
4. Uses saved SimpleImputer (median fitted strictly on training data).
5. Outputs raw LightGBM probability without shift, clamp, or heuristic scaling.
6. Strictly rejects daily bars (requires 5-minute intraday data).
"""

import os
import logging
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd

from app.ml.phase5_feature_builder import (
    PHASE5_FEATURE_COLS,
    build_phase5_feature_dataframe,
)

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH = "data/models/phase5_d_lightgbm.joblib"


class Phase5ProductionInference:
    """Production inference wrapper around the frozen Phase 5 D-LightGBM artifact."""

    def __init__(self, artifact_path: str = DEFAULT_ARTIFACT_PATH):
        self.artifact_path = artifact_path
        self.model = None
        self.imputer = None
        self.feature_cols = None
        self.threshold: float = 0.8000
        self.model_family: str = "lightgbm"
        self._load_artifact()

    def _load_artifact(self):
        """Loads and validates the serialized model artifact. Fails loudly if missing."""
        if not os.path.exists(self.artifact_path):
            raise FileNotFoundError(
                f"Phase 5 model artifact missing at '{self.artifact_path}'. "
                f"Cannot perform live inference without the frozen research artifact."
            )

        try:
            artifact = joblib.load(self.artifact_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to deserialize Phase 5 model artifact at '{self.artifact_path}': {exc}"
            ) from exc

        required_keys = ["model", "imputer", "feature_cols", "threshold"]
        missing = [k for k in required_keys if k not in artifact]
        if missing:
            raise ValueError(
                f"Phase 5 model artifact at '{self.artifact_path}' is corrupted or incomplete; missing keys: {missing}"
            )

        self.model = artifact["model"]
        self.imputer = artifact["imputer"]
        self.feature_cols = artifact["feature_cols"]
        self.threshold = float(artifact.get("threshold", 0.8000))
        self.model_family = str(artifact.get("model_family", "lightgbm"))

        # Verify feature columns match canonical sequence
        if self.feature_cols != PHASE5_FEATURE_COLS:
            raise ValueError(
                f"Feature columns in artifact do not match canonical Phase 5 schema.\n"
                f"Artifact: {self.feature_cols}\n"
                f"Canonical: {PHASE5_FEATURE_COLS}"
            )

        logger.info(
            "Phase5ProductionInference loaded artifact '%s' successfully (family=%s, threshold=%.4f).",
            self.artifact_path,
            self.model_family,
            self.threshold,
        )

    def predict_raw_probability(self, df_features: pd.DataFrame) -> float:
        """
        Runs a 1-row DataFrame through imputer and model, returning raw P(WIN | X).
        No confidence clamping, no +25 shift, no heuristic adjustments.
        """
        if list(df_features.columns) != self.feature_cols:
            raise ValueError(
                f"Feature columns mismatch for Phase 5 inference.\n"
                f"Expected: {self.feature_cols}\n"
                f"Got:      {list(df_features.columns)}"
            )

        # Apply saved training imputer strictly to sentiment_score
        df_imputed = df_features.copy()
        df_imputed["sentiment_score"] = self.imputer.transform(df_imputed[["sentiment_score"]]).ravel()

        # Compute raw probability
        probs = self.model.predict_proba(df_imputed)
        # Handle 1D or 2D return from LGBMClassifier
        if hasattr(probs, "ndim") and probs.ndim == 2:
            raw_p = float(probs[0, 1])
        elif isinstance(probs, (list, np.ndarray)) and len(probs) > 0:
            raw_p = float(probs[0])
        else:
            raw_p = float(probs)

        return raw_p

    def evaluate_candidate(
        self,
        technical_indicators: Dict[str, Any],
        news_features: Dict[str, Any],
        context_features: Dict[str, Any],
        direction: int,
        timeframe: str = "5m",
    ) -> float:
        """
        Evaluates a single direction candidate (+1 or -1) on 5-minute data.
        Fails loudly if daily data is passed.
        """
        if str(timeframe).lower() != "5m":
            raise ValueError(
                f"Phase 5 model strictly requires 5-minute data. "
                f"Received timeframe='{timeframe}'. Daily candles cannot be fed to this model."
            )

        df = build_phase5_feature_dataframe(
            technical_indicators,
            news_features,
            context_features,
            direction=direction,
        )
        return self.predict_raw_probability(df)

    def evaluate_dual_directions(
        self,
        technical_indicators: Dict[str, Any],
        news_features: Dict[str, Any],
        context_features: Dict[str, Any],
        timeframe: str = "5m",
    ) -> Dict[str, Any]:
        """
        Evaluates BOTH direction=+1 (LONG) and direction=-1 (SHORT) independently.

        Qualification:
            long_qualified  = (P_LONG >= 0.8000)
            short_qualified = (P_SHORT >= 0.8000)

        Decision policy:
            - If long_qualified and not short_qualified:  BUY
            - If short_qualified and not long_qualified:  SELL
            - If both qualify:                            Choose higher probability signal
            - If neither qualifies:                       HOLD (unqualified, zero allocation)
        """
        if str(timeframe).lower() != "5m":
            raise ValueError(
                f"Phase 5 model strictly requires 5-minute data. "
                f"Received timeframe='{timeframe}'. Daily candles cannot be fed to this model."
            )

        p_long = self.evaluate_candidate(
            technical_indicators, news_features, context_features, direction=1, timeframe=timeframe
        )
        p_short = self.evaluate_candidate(
            technical_indicators, news_features, context_features, direction=-1, timeframe=timeframe
        )

        long_qualified = bool(p_long >= self.threshold)
        short_qualified = bool(p_short >= self.threshold)

        if long_qualified and not short_qualified:
            action = "BUY"
            chosen_p = p_long
            chosen_dir = 1
            qualified = True
        elif short_qualified and not long_qualified:
            action = "SELL"
            chosen_p = p_short
            chosen_dir = -1
            qualified = True
        elif long_qualified and short_qualified:
            # Both directions qualify at this exact candle
            if p_long >= p_short:
                action = "BUY"
                chosen_p = p_long
                chosen_dir = 1
            else:
                action = "SELL"
                chosen_p = p_short
                chosen_dir = -1
            qualified = True
        else:
            action = "HOLD"
            chosen_p = max(p_long, p_short)
            chosen_dir = 1 if p_long >= p_short else -1
            qualified = False

        return {
            "model_name": "phase5_d_lightgbm",
            "model_probability": round(float(chosen_p), 6),
            "model_threshold": self.threshold,
            "p_long": round(float(p_long), 6),
            "p_short": round(float(p_short), 6),
            "long_qualified": long_qualified,
            "short_qualified": short_qualified,
            "qualified": qualified,
            "action": action,
            "direction": chosen_dir,
        }
