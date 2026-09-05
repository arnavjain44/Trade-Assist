from typing import Dict, Any
from app.config import settings


class GraphStoreManager:
    """Manages Neo4j graph database queries for stock/pattern/outcome relationships.
    Connection is only attempted when NEO4J_URI is explicitly set in config/env.
    Falls back to static mock data instantly when Neo4j is unavailable.
    """

    def __init__(self):
        self.driver = None
        # Only attempt connection if URI is explicitly configured
        if not settings.NEO4J_URI:
            print("Neo4j: No URI configured. Using graph DB fallback provider.")
            return
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                connection_timeout=2,  # 2-second max connection wait
                max_connection_lifetime=30,
            )
            # Verify connectivity with a quick ping
            self.driver.verify_connectivity()
            print(f"Neo4j: Connected to {settings.NEO4J_URI}")
        except Exception as e:
            print(f"Neo4j: Connection failed ({e}). Using graph DB fallback provider.")
            if self.driver:
                try:
                    self.driver.close()
                except Exception:
                    pass
            self.driver = None

    def query_pattern_relationships(self, symbol: str) -> Dict[str, Any]:
        """
        Executes Cypher query:
        (Stock)-[:ON_DAY]->(TradingDay)-[:SHOWED_PATTERN]->(Pattern)-[:RESULTED_IN]->(Outcome)
        Falls back to mock data instantly if Neo4j is unavailable.
        """
        if self.driver:
            cypher = """
            MATCH (s:Stock {symbol: $symbol})-[:ON_DAY]->(td:TradingDay)-[:SHOWED_PATTERN]->(p:Pattern)-[:RESULTED_IN]->(o:Outcome)
            RETURN p.name as pattern_name, o.direction as expected_direction, o.avg_return_pct as avg_return
            LIMIT 1
            """
            try:
                with self.driver.session() as session:
                    result = session.run(cypher, symbol=symbol)
                    record = result.single()
                    if record:
                        return {
                            "graph_pattern": record["pattern_name"],
                            "expected_direction": record["expected_direction"],
                            "historical_avg_return_pct": record["avg_return"]
                        }
            except Exception as e:
                print(f"Neo4j execution error: {e}")

        # Deterministic fallback — varies by symbol for realistic diversity
        pattern_map = {
            "RELIANCE.NS": ("Energy Breakout", "UPWARD", 2.80),
            "TCS.NS": ("IT Momentum", "UPWARD", 2.10),
            "INFY.NS": ("RSI Oversold Recovery", "UPWARD", 1.95),
            "HDFCBANK.NS": ("Bollinger Squeeze", "UPWARD", 2.45),
            "ICICIBANK.NS": ("VWAP Bounce", "UPWARD", 2.20),
            "TATAMOTORS.NS": ("MACD Crossover", "UPWARD", 3.10),
        }
        pattern, direction, avg_return = pattern_map.get(
            symbol, ("Consolidation Breakout", "UPWARD", 2.45)
        )
        return {
            "graph_pattern": pattern,
            "expected_direction": direction,
            "historical_avg_return_pct": avg_return
        }

    def close(self):
        if self.driver:
            self.driver.close()


graph_store = GraphStoreManager()
