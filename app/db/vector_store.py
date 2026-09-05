"""
Vector Store Manager Module

Updated for Phase 4:
Manages real whole-market and per-stock Chroma collections without synthetic seed placeholders.
Maintains backward compatibility for API queries and imports.
"""

import logging
import threading
from typing import Dict, List, Any, Optional
import numpy as np

from app.ml.context_store import (
    HistoricalContextStore,
    MARKET_COLLECTION_NAME,
    STOCK_COLLECTION_NAME,
    FINGERPRINT_DIM,
    build_stock_indicator_vector,
)

logger = logging.getLogger(__name__)

# Sector mapping for NIFTY 50 equities + common tickers (preserved for API compatibility)
SYMBOL_SECTOR_MAP: Dict[str, str] = {
    "RELIANCE.NS":   "energy",
    "TCS.NS":        "it",
    "INFY.NS":       "it",
    "HDFCBANK.NS":   "banking",
    "ICICIBANK.NS":  "banking",
    "TATAMOTORS.NS": "auto",
    "BHARTIARTL.NS": "telecom",
    "SBIN.NS":       "banking",
    "AXISBANK.NS":   "banking",
    "ITC.NS":        "fmcg",
    "WIPRO.NS":      "it",
    "BAJFINANCE.NS": "finance",
    "MARUTI.NS":     "auto",
    "LT.NS":         "infra",
    "HCLTECH.NS":    "it",
    "SUNPHARMA.NS":  "pharma",
    "TITAN.NS":      "fmcg",
    "ULTRACEMCO.NS": "infra",
    "ASIANPAINT.NS": "fmcg",
    "KOTAKBANK.NS":  "banking",
    "TATASTEEL.NS":  "metals",
    "INDUSINDBK.NS": "banking",
    "NTPC.NS":       "energy",
    "POWERGRID.NS":  "energy",
    "COALINDIA.NS":  "energy",
    "ONGC.NS":       "energy",
    "HDFCLIFE.NS":   "finance",
    "SBILIFE.NS":    "finance",
    "BAJAJ-AUTO.NS": "auto",
    "M&M.NS":        "auto",
    "HEROMOTOCO.NS": "auto",
    "EICHERMOT.NS":  "auto",
    "BPCL.NS":       "energy",
    "IOC.NS":        "energy",
    "DIVISLAB.NS":   "pharma",
    "DRREDDY.NS":    "pharma",
    "CIPLA.NS":      "pharma",
    "APOLLOHOSP.NS": "pharma",
    "BRITANNIA.NS":  "fmcg",
    "NESTLEIND.NS":  "fmcg",
    "HINDUNILVR.NS": "fmcg",
    "GRASIM.NS":     "infra",
    "JSWSTEEL.NS":   "metals",
    "HINDALCO.NS":   "metals",
    "ADANIENT.NS":   "infra",
    "ADANIPORTS.NS": "infra",
    "BEL.NS":        "defense",
    "HAL.NS":        "defense",
    "TRENT.NS":      "fmcg",
    "ZOMATO.NS":     "tech",
}


def get_sector_for_symbol(symbol: str) -> str:
    """Returns sector string for any symbol, inferring from keyword if unlisted."""
    sym = symbol.upper().strip()
    if sym in SYMBOL_SECTOR_MAP:
        return SYMBOL_SECTOR_MAP[sym]
    if "BANK" in sym or "FIN" in sym: return "banking"
    if "AUTO" in sym or "MOT" in sym: return "auto"
    if "TECH" in sym or "SOFT" in sym: return "it"
    if "PHARMA" in sym or "LAB" in sym: return "pharma"
    if "POWER" in sym or "GAS" in sym or "OIL" in sym: return "energy"
    if "STEEL" in sym or "MET" in sym: return "metals"
    return "equity"


def build_fingerprint_vector(indicators: Dict[str, Any]) -> List[float]:
    """Builds normalized 9-dim indicator vector (backward compatible wrapper)."""
    return build_stock_indicator_vector(indicators)


class VectorStoreManager:
    """Manages ChromaDB per-stock and whole-market fingerprint collections."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self._context_store: Optional[HistoricalContextStore] = None
        self._ready = False
        self._lock = threading.Lock()

        init_thread = threading.Thread(target=self._init_chroma, daemon=True)
        init_thread.start()

    def _init_chroma(self):
        try:
            store = HistoricalContextStore(persist_directory=self.persist_directory)
            with self._lock:
                self._context_store = store
                self._ready = True
            count = store.stock_collection.count()
            logger.info("ChromaDB: Collections initialized (%d stock docs).", count)
        except Exception as e:
            logger.warning("ChromaDB: Init failed (%s).", e)
            with self._lock:
                self._ready = False

    def query_similar_patterns(self, symbol: str, features: List[float], n_results: int = 3) -> Dict[str, Any]:
        with self._lock:
            ready = self._ready
            store = self._context_store

        if ready and store is not None:
            try:
                count = store.stock_collection.count()
                if count > 0:
                    vec = features[:FINGERPRINT_DIM]
                    if len(vec) < FINGERPRINT_DIM:
                        vec = vec + [0.0] * (FINGERPRINT_DIM - len(vec))

                    results = store.stock_collection.query(
                        query_embeddings=[vec],
                        n_results=min(n_results, count),
                        where={"symbol": {"$eq": symbol.upper().strip()}},
                    )
                    if results and results.get("metadatas") and results["metadatas"][0]:
                        top = results["metadatas"][0][0]
                        dist = results["distances"][0][0]
                        sim_score = round(max(0.0, 1.0 - dist / 2.0), 4)
                        return {
                            "matched_pattern": "Historical Indicator State",
                            "similarity_score": sim_score,
                            "historical_win_rate": 0.50,
                        }
            except Exception as e:
                logger.warning("ChromaDB query error: %s", e)

        return {
            "matched_pattern": "Historical Indicator State",
            "similarity_score": 0.50,
            "historical_win_rate": 0.50,
        }

    def upsert_stock_fingerprint(self, symbol: str, indicator_vector: List[float], sector: str) -> bool:
        """Stores or updates a stock's indicator fingerprint."""
        with self._lock:
            ready = self._ready
            store = self._context_store

        if not ready or store is None:
            return False

        try:
            doc_id = f"stock_{symbol.upper().strip()}_latest"
            store.stock_collection.upsert(
                ids=[doc_id],
                embeddings=[indicator_vector],
                metadatas=[{"symbol": symbol.upper().strip(), "sector": sector, "trading_date": "latest"}],
                documents=[f"Latest fingerprint for {symbol}"],
            )
            return True
        except Exception as e:
            logger.warning("ChromaDB upsert error: %s", e)
            return False

    def find_similar_peers(
        self,
        symbol: str,
        sector: str,
        indicator_vector: List[float],
        max_peers: int = 5,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            ready = self._ready
            store = self._context_store

        if not ready or store is None:
            return []

        try:
            n_query = min(max_peers + 3, store.stock_collection.count())
            if n_query < 1:
                return []

            results = store.stock_collection.query(
                query_embeddings=[indicator_vector],
                n_results=n_query,
            )

            peers = []
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for meta, dist in zip(metadatas, distances):
                peer_symbol = meta.get("symbol", "")
                if peer_symbol == symbol.upper().strip():
                    continue
                peers.append({
                    "symbol": peer_symbol,
                    "sector": sector,
                    "similarity_score": round(max(0.0, 1.0 - dist / 2.0), 4),
                    "pattern": "Historical State",
                    "win_rate": 0.50,
                })
                if len(peers) >= max_peers:
                    break

            return peers
        except Exception as e:
            logger.warning("ChromaDB find_similar_peers error: %s", e)
            return []


vector_store = VectorStoreManager()
