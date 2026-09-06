"""
Phase 5 Chroma Population Script

Populates the two project ChromaDB collections:
1. whole_market_daily_fingerprints (59 market daily fingerprints)
2. per_stock_daily_fingerprints (2,832 stock daily fingerprints)

Uses real 5-minute historical candles from data/processed/phase5_features.parquet.
Deterministic document IDs ensure idempotent re-execution.
"""

import sys
import os
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ml.context_store import HistoricalContextStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PARQUET_PATH = "data/processed/phase5_features.parquet"

def populate():
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Historical feature parquet missing at '{PARQUET_PATH}'")

    logger.info("Loading historical feature parquet from %s...", PARQUET_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    logger.info("Loaded %d rows across %d trading days.", len(df), df["trading_date"].nunique())

    store = HistoricalContextStore(persist_directory="./chroma_db")
    m_cnt, s_cnt = store.populate_from_historical_candles(df)
    logger.info("Successfully populated ChromaDB: %d market fingerprints, %d stock fingerprints.", m_cnt, s_cnt)
    return m_cnt, s_cnt

if __name__ == "__main__":
    populate()