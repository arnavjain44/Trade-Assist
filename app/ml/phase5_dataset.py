"""
Phase 5 Dataset Builder Module

Constructs `data/processed/phase5_features.parquet` and `data/processed/phase5_data_quality.json`
by combining:
1. Phase 4 features: 9 Technical + FinBERT news sentiment + Chroma historical context.
2. Phase 2 candidate trade labels: Model A (+2.2% target, -0.9% stop loss, 240-min max hold).
3. Trade execution metadata: entry_price, target_price, stop_price, exit_timestamp,
   exit_price, exit_reason, holding_period_minutes, realized_return.

Strictly causal: zero future news, zero current-day context, zero label leakage into features.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from app.ml.labeling import HistoricalTradeLabeler

logger = logging.getLogger(__name__)


class Phase5DatasetBuilder:
    """Builds the unified Phase 5 feature and labeled trade candidate dataset."""

    def __init__(
        self,
        phase4_features_path: str = "data/processed/phase4_features.parquet",
        output_parquet_path: str = "data/processed/phase5_features.parquet",
        output_quality_path: str = "data/processed/phase5_data_quality.json",
        target_pct: float = 0.022,
        stop_loss_pct: float = 0.009,
        max_hold_minutes: int = 240,
    ):
        self.phase4_features_path = Path(phase4_features_path)
        self.output_parquet_path = Path(output_parquet_path)
        self.output_quality_path = Path(output_quality_path)
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_minutes = max_hold_minutes

    def build_dataset(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads Phase 4 features, applies frozen Model A labeling, and saves Phase 5 dataset."""
        logger.info("--- Starting Phase 5 Dataset Construction ---")

        if not self.phase4_features_path.exists():
            raise FileNotFoundError(f"Phase 4 features file not found at {self.phase4_features_path}")

        logger.info("Loading Phase 4 features from %s...", self.phase4_features_path)
        df_p4 = pd.read_parquet(self.phase4_features_path)
        if "mean_sentiment" in df_p4.columns:
            df_p4["sentiment_score"] = df_p4["mean_sentiment"]
        logger.info("Loaded %d rows with %d columns from Phase 4.", len(df_p4), len(df_p4.columns))

        # Run HistoricalTradeLabeler
        labeler = HistoricalTradeLabeler(
            target_pct=self.target_pct,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_minutes=self.max_hold_minutes,
        )

        df_labeled, label_meta = labeler.label_dataset(
            df_p4,
            output_parquet_path=str(self.output_parquet_path),
            output_json_path="scratch/temp_p5_label_quality.json",
        )

        # Generate Phase 5 Data Quality Report
        quality_report = self._generate_data_quality_report(df_labeled, df_p4)

        self.output_quality_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_quality_path, "w", encoding="utf-8") as f:
            json.dump(quality_report, f, indent=2)

        logger.info(
            "Phase 5 Dataset exported to %s (%d rows, %d columns). Data quality saved to %s.",
            self.output_parquet_path,
            len(df_labeled),
            len(df_labeled.columns),
            self.output_quality_path,
        )

        return df_labeled, quality_report

    def _generate_data_quality_report(self, df_labeled: pd.DataFrame, df_raw: pd.DataFrame) -> Dict[str, Any]:
        """Generates comprehensive audit metrics for Phase 5 dataset."""
        n_candidates = len(df_labeled)
        n_symbols = int(df_labeled["symbol"].nunique()) if "symbol" in df_labeled.columns else 0
        valid_candidates = int((df_labeled["label_status"] == "VALID").sum())
        positive_labels = int(((df_labeled["label_status"] == "VALID") & (df_labeled["label"] == 1)).sum())

        has_news_cnt = int(df_labeled["has_news"].sum()) if "has_news" in df_labeled.columns else 0
        missing_news_pct = round((1.0 - has_news_cnt / n_candidates) * 100.0, 2) if n_candidates > 0 else 100.0

        if "trading_date" in df_labeled.columns:
            n_trading_days = int(df_labeled["trading_date"].nunique())
            earliest_date = str(df_labeled["trading_date"].min())
            latest_date = str(df_labeled["trading_date"].max())
        else:
            n_trading_days = 0
            earliest_date = "UNKNOWN"
            latest_date = "UNKNOWN"

        report = {
            "dataset_phase": "Phase 5 Real-Context ML",
            "number_of_candles": len(df_raw),
            "number_of_candidate_trades": n_candidates,
            "number_of_symbols": n_symbols,
            "number_of_trading_days": n_trading_days,
            "date_range": {"start": earliest_date, "end": latest_date},
            "valid_labeled_candidates": valid_candidates,
            "positive_label_count": positive_labels,
            "positive_label_prevalence_pct": round((positive_labels / valid_candidates) * 100.0, 4) if valid_candidates > 0 else 0.0,
            "has_news_count": has_news_cnt,
            "missing_news_pct": missing_news_pct,
            "chroma_context_present": "market_similarity" in df_labeled.columns and "stock_similarity" in df_labeled.columns,
            "neo4j_context_present": False,
            "neo4j_status": "UNAVAILABLE_NO_URI",
            "temporal_provenance": {
                "technical_features": "Contemporaneous (<= T)",
                "finbert_news": "Strictly prior (< T)",
                "chroma_fingerprints": "Strictly prior completed days (< current date)",
                "labels": "Intraday future outcome window (T to T+240m) strictly isolated to label column",
            },
        }
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = Phase5DatasetBuilder()
    builder.build_dataset()
