"""
Phase 4 Neo4j Historical Graph Ingestion Engine

Populates Neo4j graph using Cypher MERGE queries strictly from real historical candle observations:
(Stock)-[:ON_DAY]->(TradingDay)-[:SHOWED_PATTERN]->(Pattern)-[:USES_INDICATOR]->(Indicator)
(Pattern)-[:RESULTED_IN]->(Outcome)
(TradingDay)-[:PART_OF]->(MarketRegime)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from app.db.graph_store import graph_store

logger = logging.getLogger(__name__)


def derive_pattern_from_row(row: pd.Series) -> Optional[Tuple[str, str, str]]:
    """Derives real technical pattern, indicator, and regime from candle features."""
    rsi = float(row.get("rsi", 50.0))
    price_vs_vwap = float(row.get("price_vs_vwap", 0.0))
    macd_diff = float(row.get("macd_diff", 0.0))
    bollinger_pos = float(row.get("bollinger_position", 0.5))

    if rsi < 35.0:
        return "RSI_Oversold_Bounce", "RSI", "Bearish_Extreme"
    elif rsi > 65.0:
        return "RSI_Overbought_Pullback", "RSI", "Bullish_Extreme"
    elif price_vs_vwap > 0.01 and macd_diff > 0.0:
        return "VWAP_Bullish_Breakout", "VWAP", "Bullish_Trend"
    elif price_vs_vwap < -0.01 and macd_diff < 0.0:
        return "VWAP_Bearish_Breakout", "VWAP", "Bearish_Trend"
    elif 0.4 <= bollinger_pos <= 0.6:
        return "Bollinger_Consolidation", "BollingerBands", "Neutral_Consolidation"

    return None


class Neo4jGraphIngestor:
    """Idempotently ingests real historical pattern relationships into Neo4j."""

    def __init__(self, driver=None):
        self.driver = driver or graph_store.driver

    def is_available(self) -> bool:
        return self.driver is not None

    def ingest_historical_patterns(self, df_processed: pd.DataFrame) -> int:
        """Ingests daily pattern observations into Neo4j via MERGE Cypher queries.

        Returns total count of created/merged trading day nodes.
        """
        if not self.is_available() or df_processed.empty:
            logger.info("Neo4j not connected or empty DataFrame. Ingestion skipped.")
            return 0

        df = df_processed.copy()
        if "trading_date" not in df.columns:
            df["trading_date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")

        # Aggregate to daily per symbol
        daily = df.groupby(["symbol", "trading_date"]).last().reset_index()

        cypher = """
        MERGE (s:Stock {symbol: $symbol})
        MERGE (td:TradingDay {date: $trading_date, symbol: $symbol})
        MERGE (s)-[:ON_DAY]->(td)
        
        MERGE (mr:MarketRegime {name: $regime_name})
        MERGE (td)-[:PART_OF]->(mr)
        
        FOREACH (ignoreMe IN CASE WHEN $pattern_name IS NOT NULL THEN [1] ELSE [] END |
            MERGE (p:Pattern {name: $pattern_name})
            MERGE (td)-[:SHOWED_PATTERN]->(p)
            
            MERGE (ind:Indicator {name: $indicator_name})
            MERGE (p)-[:USES_INDICATOR]->(ind)
            
            MERGE (o:Outcome {direction: $direction, avg_return: $realized_return})
            MERGE (p)-[:RESULTED_IN]->(o)
        )
        """

        merged_count = 0
        try:
            with self.driver.session() as session:
                for idx, row in daily.iterrows():
                    sym = str(row["symbol"]).upper().strip()
                    date_str = str(row["trading_date"])
                    pattern_tuple = derive_pattern_from_row(row)
                    
                    p_name, ind_name, regime_name = pattern_tuple if pattern_tuple else (None, None, "Neutral")
                    ret_val = float(row.get("realized_return", row.get("direction", 0.0)))
                    direction_str = "UPWARD" if ret_val >= 0 else "DOWNWARD"

                    session.run(
                        cypher,
                        symbol=sym,
                        trading_date=date_str,
                        regime_name=regime_name,
                        pattern_name=p_name,
                        indicator_name=ind_name,
                        direction=direction_str,
                        realized_return=ret_val,
                    )
                    merged_count += 1
            logger.info("Neo4jGraphIngestor merged %d historical trading day records.", merged_count)
        except Exception as exc:
            logger.warning("Neo4j ingestion error: %s", exc)

        return merged_count
