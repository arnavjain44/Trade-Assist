import logging
from typing import Dict, Any, List
from app.core.data_fetcher import data_fetcher
from app.core.indicators import indicators_calculator
from app.core.sentiment import sentiment_analyzer
from app.db.vector_store import vector_store, SYMBOL_SECTOR_MAP, build_fingerprint_vector, get_sector_for_symbol
from app.db.graph_store import graph_store
from app.ml.pipeline import ml_engine
from app.core.constraints import constraint_enforcer

logger = logging.getLogger(__name__)


class AgentTools:
    """Granular toolset for the LLM agent loop as specified in Section 6.5."""



    @staticmethod
    async def fetch_and_calculate_indicators(symbols: List[str]) -> Dict[str, Any]:
        """Bundled tool: Fetches latest price data and calculates 6 technical indicators + news sentiment."""
        df_dict = await data_fetcher.fetch_stock_data_async(symbols)
        results = {}
        for symbol, df in df_dict.items():
            df_ind = indicators_calculator.calculate_all(df)
            latest_ind = indicators_calculator.extract_latest_summary(df_ind)
            sentiments = sentiment_analyzer.analyze_headlines(symbol)
            agg_sentiment = sentiment_analyzer.calculate_aggregated_sentiment(sentiments)

            results[symbol] = {
                "dataframe": df_ind,
                "latest_indicators": latest_ind,
                "sentiments": sentiments,
                "sentiment_score": agg_sentiment,
                "current_price": latest_ind["close_price"]
            }
        return results

    @staticmethod
    def query_vector_db(symbol: str, features: List[float]) -> Dict[str, Any]:
        """Tool: Queries ChromaDB for market & per-stock similarity fingerprints."""
        return vector_store.query_similar_patterns(symbol, features)

    @staticmethod
    def query_graph_db(symbol: str) -> Dict[str, Any]:
        """Tool: Queries Neo4j for stock/pattern/outcome/regime Cypher relationships."""
        return graph_store.query_pattern_relationships(symbol)

    @staticmethod
    def predict(symbol: str, current_price: float, indicators: Dict[str, Any], sentiment_score: float, similarity_score: float) -> Dict[str, Any]:
        """Tool: Runs trained ML model to output direction (BUY/SELL/HOLD), confidence %, and target price."""
        return ml_engine.predict_trade_signal(symbol, current_price, indicators, sentiment_score, similarity_score)

    @staticmethod
    def enforce_constraints(raw_recommendations: List[Dict[str, Any]], total_capital: float) -> List[Dict[str, Any]]:
        """Tool: Mandatory, deterministic post-processing layer (same-day exit hard rule + ~40% cap position sizing)."""
        return constraint_enforcer.enforce_intraday_constraints(raw_recommendations, total_capital)

    # ------------------------------------------------------------------
    # New tools for sector peer comparison
    # ------------------------------------------------------------------

    @staticmethod
    def upsert_fingerprint(symbol: str, indicators: Dict[str, Any]) -> bool:
        """Stores/updates a stock's real 9-dim indicator fingerprint in ChromaDB."""
        sector = get_sector_for_symbol(symbol)
        vec = build_fingerprint_vector(indicators)
        return vector_store.upsert_stock_fingerprint(symbol, vec, sector)

    @staticmethod
    async def get_sector_peers(
        symbol: str,
        indicator_vector: List[float],
        max_peers: int = 5,
    ) -> List[Dict[str, Any]]:
        """Finds sector peers via Chroma similarity, then runs the full pipeline on each peer."""
        sector = get_sector_for_symbol(symbol)
        raw_peers = vector_store.find_similar_peers(
            symbol=symbol,
            sector=sector,
            indicator_vector=indicator_vector,
            max_peers=max_peers,
        )


        if not raw_peers:
            logger.info("No Chroma peers found for %s in sector '%s'.", symbol, sector)
            return []

        peer_symbols = [p["symbol"] for p in raw_peers]
        similarity_map = {p["symbol"]: p["similarity_score"] for p in raw_peers}

        # Fetch all peer data in one parallel batch (skips failures per-ticker)
        try:
            peer_data = await data_fetcher.fetch_stock_data_async(peer_symbols)
        except RuntimeError as exc:
            logger.error("get_sector_peers: All peer fetches failed — %s", exc)
            return []

        enriched_peers = []
        for peer_sym, df in peer_data.items():
            try:
                df_ind = indicators_calculator.calculate_all(df)
                latest_ind = indicators_calculator.extract_latest_summary(df_ind)
                
                if latest_ind.get("close_price", 0.0) <= 0:
                    logger.warning("get_sector_peers: Skipping peer %s due to invalid price 0.00", peer_sym)
                    continue

                sentiments = sentiment_analyzer.analyze_headlines(peer_sym)
                sent_score = sentiment_analyzer.calculate_aggregated_sentiment(sentiments)

                vec_res = vector_store.query_similar_patterns(peer_sym, [latest_ind["close_price"], sent_score, latest_ind["rsi"]])
                pred = ml_engine.predict_trade_signal(
                    peer_sym,
                    latest_ind["close_price"],
                    latest_ind,
                    sent_score,
                    vec_res.get("similarity_score", 0.80),
                )
                enriched_peers.append({
                    "symbol": peer_sym,
                    "sector": sector,
                    "similarity_score": similarity_map.get(peer_sym, 0.0),
                    "confidence": pred["confidence"],
                    "action": pred["action"],
                    "current_price": latest_ind["close_price"],
                    "ema_5": latest_ind["ema_5"],
                    "rsi": latest_ind["rsi"],
                    "macd": latest_ind["macd"],
                    "vwap": latest_ind["vwap"],
                    "bb_upper": latest_ind["bb_upper"],
                    "bb_lower": latest_ind["bb_lower"],
                    "sentiment_score": round(sent_score, 3),
                    "overall_bias": latest_ind["overall_technical_bias"],
                })
            except Exception as exc:
                logger.warning("get_sector_peers: Skipping peer %s — %s", peer_sym, exc)
                continue

        return enriched_peers


agent_tools = AgentTools()
