"""
Phase 4 Historical Context Store & Chroma Vector Database Manager

Provides:
1. Computation of real whole-market and per-stock daily fingerprints from OHLCV candles.
2. Population of the TWO project ChromaDB collections:
   - whole_market_daily_fingerprints
   - per_stock_daily_fingerprints
3. Deterministic document IDs and idempotent upserts.
4. Temporal similarity retrieval strictly filtering for trading_date_int < query_date_int.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chromadb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MARKET_COLLECTION_NAME = "whole_market_daily_fingerprints"
STOCK_COLLECTION_NAME = "per_stock_daily_fingerprints"
FINGERPRINT_DIM = 9


def date_to_int(date_str: str) -> int:
    """Converts 'YYYY-MM-DD' or timestamp string to integer YYYYMMDD for Chroma numeric filtering."""
    try:
        clean = str(date_str).split(" ")[0].replace("-", "")
        return int(clean)
    except Exception:
        return 20260101


def build_stock_indicator_vector(indicators: Dict[str, float]) -> List[float]:
    """Builds normalized 9-dim scale-independent indicator vector from indicator dictionary.

    Fields: price_vs_ema5, rsi, obv_norm, bollinger_position, macd/close, macd_signal/close, macd_diff/close, price_vs_vwap, price_direction_causal.
    Raw close_price and raw unscaled MACD are explicitly normalized/scaled to ensure scale independence across stocks and time.
    """
    close = float(indicators.get("close", 0.0))
    if close > 0:
        macd_norm = float(indicators.get("macd", 0.0)) / close
        macd_sig_norm = float(indicators.get("macd_signal", 0.0)) / close
        macd_diff_norm = float(indicators.get("macd_diff", 0.0)) / close
    else:
        macd_norm = float(indicators.get("macd", 0.0))
        macd_sig_norm = float(indicators.get("macd_signal", 0.0))
        macd_diff_norm = float(indicators.get("macd_diff", 0.0))

    obv_raw = indicators.get("obv", 0.0)
    obv_norm = float(np.sign(obv_raw) * np.log1p(abs(obv_raw)))

    # Pure causal price action direction (+1.0 if price >= ema5 else -1.0)
    p_ema5 = float(indicators.get("price_vs_ema5", 0.0))
    causal_direction = float(np.sign(p_ema5)) if p_ema5 != 0 else float(indicators.get("direction", 1.0))

    return [
        p_ema5,
        float(indicators.get("rsi", 50.0)),
        obv_norm,
        float(indicators.get("bollinger_position", 0.5)),
        macd_norm,
        macd_sig_norm,
        macd_diff_norm,
        float(indicators.get("price_vs_vwap", 0.0)),
        causal_direction,
    ]


class HistoricalContextStore:
    """Manages real historical whole-market and per-stock ChromaDB collections."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.market_collection = self.client.get_or_create_collection(
            name=MARKET_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.stock_collection = self.client.get_or_create_collection(
            name=STOCK_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def populate_from_historical_candles(self, df_processed: pd.DataFrame) -> Tuple[int, int]:
        """Builds daily stock and daily whole-market fingerprints from real historical 5m candles.

        Inputs: Combined processed DataFrame containing 5-minute candles with 9 technical features.
        Returns: (market_count, stock_count).
        """
        if df_processed.empty:
            return 0, 0

        df = df_processed.copy()
        if "trading_date" not in df.columns:
            df["trading_date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")

        for col in ["price_vs_ema5", "rsi", "obv", "bollinger_position", "macd", "macd_signal", "macd_diff", "price_vs_vwap", "direction", "close"]:
            if col not in df.columns:
                df[col] = 0.0 if col != "rsi" else 50.0

        # 1. Per-Stock Daily Aggregation (take final 5m candle of each trading day)
        daily_stock = (
            df.sort_values("timestamp")
            .groupby(["symbol", "trading_date"])
            .last()
            .reset_index()
        )

        stock_ids, stock_embeddings, stock_metadatas, stock_documents = [], [], [], []
        for idx, row in daily_stock.iterrows():
            sym = str(row["symbol"]).upper().strip()
            date_str = str(row["trading_date"])
            date_int = date_to_int(date_str)
            doc_id = f"stock_{sym}_{date_str}"

            vec = build_stock_indicator_vector(row.to_dict())

            stock_ids.append(doc_id)
            stock_embeddings.append(vec)
            stock_metadatas.append({
                "symbol": sym,
                "trading_date": date_str,
                "trading_date_int": date_int,
                "rsi": float(row.get("rsi", 50.0)),
                "macd": float(row.get("macd", 0.0)),
                "source_timestamp": str(row["timestamp"]),
            })
            stock_documents.append(f"Real stock state for {sym} on {date_str}")

        if stock_ids:
            self.stock_collection.upsert(
                ids=stock_ids,
                embeddings=stock_embeddings,
                metadatas=stock_metadatas,
                documents=stock_documents,
            )

        # 2. Whole-Market Daily Aggregation (cross-sectional average of features per day)
        agg_dict = {
            col: "mean"
            for col in ["rsi", "price_vs_ema5", "price_vs_vwap", "macd", "macd_signal", "macd_diff", "bollinger_position", "obv", "direction"]
            if col in daily_stock.columns
        }

        daily_market = (
            daily_stock.groupby("trading_date")
            .agg(agg_dict)
            .reset_index()
        )

        market_ids, market_embeddings, market_metadatas, market_documents = [], [], [], []
        for idx, row in daily_market.iterrows():
            date_str = str(row["trading_date"])
            date_int = date_to_int(date_str)
            doc_id = f"market_{date_str}"

            vec = build_stock_indicator_vector(row.to_dict())

            market_ids.append(doc_id)
            market_embeddings.append(vec)
            market_metadatas.append({
                "trading_date": date_str,
                "trading_date_int": date_int,
                "market_rsi_avg": float(row.get("rsi", 50.0)),
                "market_macd_avg": float(row.get("macd", 0.0)),
                "source_timestamp": date_str,
            })
            market_documents.append(f"Real whole-market state on {date_str}")

        if market_ids:
            self.market_collection.upsert(
                ids=market_ids,
                embeddings=market_embeddings,
                metadatas=market_metadatas,
                documents=market_documents,
            )

        logger.info(
            "HistoricalContextStore populated %d stock daily fingerprints and %d market daily fingerprints.",
            len(stock_ids),
            len(market_ids),
        )
        return len(market_ids), len(stock_ids)

    def query_market_similarity(self, query_vector: List[float], query_date_str: str) -> float:
        """Queries market collection strictly for trading_date_int < query_date_int.

        Returns similarity score (0.0 to 1.0) of most similar prior market day.
        """
        count = self.market_collection.count()
        if count == 0:
            return 0.0

        query_int = date_to_int(query_date_str)
        try:
            results = self.market_collection.query(
                query_embeddings=[query_vector],
                n_results=min(5, count),
                where={"trading_date_int": {"$lt": query_int}},
            )
            if results and results.get("distances") and results["distances"][0]:
                dist = results["distances"][0][0]
                return round(max(0.0, 1.0 - dist / 2.0), 4)
        except Exception as exc:
            logger.warning("Market similarity query error: %s", exc)

        return 0.0

    def query_stock_similarity(self, symbol: str, query_vector: List[float], query_date_str: str) -> float:
        """Queries stock collection strictly for matching symbol AND trading_date_int < query_date_int.

        Returns similarity score (0.0 to 1.0) of most similar prior stock day.
        """
        count = self.stock_collection.count()
        if count == 0:
            return 0.0

        clean_sym = symbol.upper().strip()
        query_int = date_to_int(query_date_str)
        try:
            results = self.stock_collection.query(
                query_embeddings=[query_vector],
                n_results=min(5, count),
                where={"$and": [{"symbol": {"$eq": clean_sym}}, {"trading_date_int": {"$lt": query_int}}]},
            )
            if results and results.get("distances") and results["distances"][0]:
                dist = results["distances"][0][0]
                return round(max(0.0, 1.0 - dist / 2.0), 4)
        except Exception as exc:
            logger.warning("Stock similarity query error: %s", exc)

        return 0.0
