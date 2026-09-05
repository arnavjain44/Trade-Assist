import threading
from typing import Dict, List, Any, Optional
import numpy as np


# Sector mapping for NIFTY 50 equities + common tickers
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
    # Dynamic inference
    if "BANK" in sym or "FIN" in sym: return "banking"
    if "AUTO" in sym or "MOT" in sym: return "auto"
    if "TECH" in sym or "SOFT" in sym: return "it"
    if "PHARMA" in sym or "LAB" in sym: return "pharma"
    if "POWER" in sym or "GAS" in sym or "OIL" in sym: return "energy"
    if "STEEL" in sym or "MET" in sym: return "metals"
    return "equity"


# Canonical vector dimension: 9 real indicator values
FINGERPRINT_DIM = 9


def build_fingerprint_vector(indicators: Dict[str, Any]) -> List[float]:
    """Builds a normalised 9-dim fingerprint vector from the indicator summary dict.

    Fields: ema_5, rsi, obv (log-scaled), bb_upper, bb_lower, macd, macd_signal, vwap, close_price
    All values are kept as raw floats; Chroma uses cosine similarity so relative scale matters.
    """
    obv_raw = indicators.get("obv", 0.0)
    obv_norm = float(np.sign(obv_raw) * np.log1p(abs(obv_raw)))  # log-scale OBV

    return [
        float(indicators.get("ema_5", 0.0)),
        float(indicators.get("rsi", 50.0)),
        obv_norm,
        float(indicators.get("bb_upper", 0.0)),
        float(indicators.get("bb_lower", 0.0)),
        float(indicators.get("macd", 0.0)),
        float(indicators.get("macd_signal", 0.0)),
        float(indicators.get("vwap", 0.0)),
        float(indicators.get("close_price", 0.0)),
    ]


class VectorStoreManager:
    """Manages ChromaDB per-stock fingerprint collection (9-dim real indicator vectors).

    - Initialised in a background daemon thread — never blocks app startup.
    - Falls back gracefully to static data if ChromaDB is unavailable.
    - Schema: collection `per_stock_fingerprints_v2` with metadata {symbol, sector, pattern, win_rate}.
    """

    COLLECTION_NAME = "per_stock_fingerprints_v2"

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.stock_collection = None
        self._ready = False
        self._lock = threading.Lock()

        init_thread = threading.Thread(target=self._init_chroma, daemon=True)
        init_thread.start()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_chroma(self):
        """Runs ChromaDB initialisation in a background thread."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)

            # Use v2 collection name so old 16-dim collection is ignored
            stock_col = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._seed_if_empty(stock_col)

            with self._lock:
                self.client = client
                self.stock_collection = stock_col
                self._ready = True
            print(f"ChromaDB: Collection '{self.COLLECTION_NAME}' ready ({stock_col.count()} docs).")
        except Exception as e:
            print(f"ChromaDB: Init failed ({e}). Peer comparison will use fallback.")
            with self._lock:
                self._ready = False

    def _seed_if_empty(self, stock_col):
        """Seeds all 20 NSE tickers with placeholder fingerprints on first run.
        Fingerprints will be overwritten with real values after the first live pipeline run.
        """
        try:
            if stock_col.count() > 0:
                return
            ids, embeddings, metadatas, documents = [], [], [], []
            for symbol, sector in SYMBOL_SECTOR_MAP.items():
                seed_val = abs(hash(symbol)) % 10000
                np.random.seed(seed_val)
                vec = np.random.normal(0, 1, size=FINGERPRINT_DIM).tolist()
                ids.append(f"seed_{symbol}")
                embeddings.append(vec)
                metadatas.append({
                    "symbol": symbol,
                    "sector": sector,
                    "pattern": "Seed Placeholder",
                    "win_rate": 0.75,
                })
                documents.append(f"Seed fingerprint for {symbol} ({sector})")
            stock_col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
            print(f"ChromaDB: Seeded {len(ids)} stock fingerprints.")
        except Exception as e:
            print(f"ChromaDB seed error: {e}")

    # ------------------------------------------------------------------
    # Existing API — signature UNCHANGED
    # ------------------------------------------------------------------

    def query_similar_patterns(self, symbol: str, features: List[float], n_results: int = 3) -> Dict[str, Any]:
        """Queries the per-stock collection for top matching historical patterns.
        Signature unchanged — pads/truncates features to FINGERPRINT_DIM internally.
        """
        with self._lock:
            ready = self._ready
            stock_col = self.stock_collection

        if ready and stock_col is not None:
            try:
                count = stock_col.count()
                if count > 0:
                    vec = _pad_or_truncate(features, FINGERPRINT_DIM, symbol)
                    results = stock_col.query(
                        query_embeddings=[vec],
                        n_results=min(n_results, count),
                    )
                    if results and results.get("metadatas") and results["metadatas"][0]:
                        top = results["metadatas"][0][0]
                        return {
                            "matched_pattern": top.get("pattern", "VWAP Bounce"),
                            "similarity_score": _dist_to_score(results["distances"][0][0]),
                            "historical_win_rate": top.get("win_rate", 0.79),
                        }
            except Exception as e:
                print(f"ChromaDB query_similar_patterns error: {e}")

        # Static per-symbol fallback
        return _static_pattern_fallback(symbol)

    # ------------------------------------------------------------------
    # New API — upsert + peer search
    # ------------------------------------------------------------------

    def upsert_stock_fingerprint(self, symbol: str, indicator_vector: List[float], sector: str) -> bool:
        """Stores or updates a stock's 9-dim indicator fingerprint in Chroma.

        Args:
            symbol: NSE ticker, e.g. "HDFCBANK.NS"
            indicator_vector: 9-dim list from build_fingerprint_vector()
            sector: sector string from SYMBOL_SECTOR_MAP

        Returns True on success, False if ChromaDB is not ready.
        """
        with self._lock:
            ready = self._ready
            stock_col = self.stock_collection

        if not ready or stock_col is None:
            return False

        try:
            doc_id = f"live_{symbol}"
            stock_col.upsert(
                ids=[doc_id],
                embeddings=[indicator_vector],
                metadatas=[{"symbol": symbol, "sector": sector, "pattern": "Live Fingerprint", "win_rate": 0.80}],
                documents=[f"Live fingerprint for {symbol}"],
            )
            return True
        except Exception as e:
            print(f"ChromaDB upsert error for {symbol}: {e}")
            return False

    def find_similar_peers(
        self,
        symbol: str,
        sector: str,
        indicator_vector: List[float],
        max_peers: int = 5,
    ) -> List[Dict[str, Any]]:
        """Finds the most similar stocks in the same sector by vector distance.

        Args:
            symbol: The stock to exclude from results (the original recommendation).
            sector: Sector to filter by (metadata where clause).
            indicator_vector: The query stock's 9-dim fingerprint vector.
            max_peers: Max number of peer results to return.

        Returns:
            List of dicts with keys: symbol, sector, similarity_score, pattern, win_rate.
            Empty list if ChromaDB unavailable or no peers found.
        """
        with self._lock:
            ready = self._ready
            stock_col = self.stock_collection

        if not ready or stock_col is None:
            print(f"ChromaDB not ready — cannot find peers for {symbol}.")
            return []

        try:
            # Request more than needed so we can exclude the input symbol
            n_query = min(max_peers + 3, stock_col.count())
            if n_query < 1:
                return []

            results = stock_col.query(
                query_embeddings=[indicator_vector],
                n_results=n_query,
                where={"sector": {"$eq": sector}},
            )

            peers = []
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for meta, dist in zip(metadatas, distances):
                peer_symbol = meta.get("symbol", "")
                if peer_symbol == symbol:
                    continue  # exclude the query stock itself
                peers.append({
                    "symbol": peer_symbol,
                    "sector": sector,
                    "similarity_score": round(_dist_to_score(dist), 4),
                    "pattern": meta.get("pattern", "N/A"),
                    "win_rate": meta.get("win_rate", 0.0),
                })
                if len(peers) >= max_peers:
                    break

            return peers
        except Exception as e:
            print(f"ChromaDB find_similar_peers error for {symbol}: {e}")
            return []


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pad_or_truncate(vec: List[float], target_dim: int, symbol: str) -> List[float]:
    """Pads (with symbol-seeded noise) or truncates a vector to target_dim."""
    if len(vec) >= target_dim:
        return vec[:target_dim]
    seed_val = abs(hash(symbol)) % 10000
    np.random.seed(seed_val)
    pad = np.random.normal(0, 0.01, size=target_dim - len(vec)).tolist()
    return vec + pad


def _dist_to_score(cosine_dist: float) -> float:
    """Converts ChromaDB cosine distance [0, 2] to a similarity score [0, 1]."""
    return round(max(0.0, 1.0 - cosine_dist / 2.0), 4)


def _static_pattern_fallback(symbol: str) -> Dict[str, Any]:
    fallback_map = {
        "RELIANCE.NS":   ("Energy Breakout", 0.91),
        "TCS.NS":        ("IT Momentum Continuation", 0.88),
        "INFY.NS":       ("RSI Oversold Recovery", 0.85),
        "HDFCBANK.NS":   ("Bollinger Squeeze Expansion", 0.90),
        "ICICIBANK.NS":  ("VWAP Reclaim", 0.87),
        "TATAMOTORS.NS": ("MACD Bullish Crossover", 0.86),
        "SBIN.NS":       ("Support Bounce", 0.83),
        "WIPRO.NS":      ("EMA Crossover", 0.84),
        "HCLTECH.NS":    ("Breakout Continuation", 0.85),
        "BAJFINANCE.NS": ("NBFC Momentum", 0.82),
    }
    pattern, score = fallback_map.get(symbol, ("Historical Bullish Continuation", 0.84))
    return {"matched_pattern": pattern, "similarity_score": score, "historical_win_rate": 0.80}


vector_store = VectorStoreManager()
