"""
Phase 4 Feature Joiner Module

Joins Phase 3's 9 technical features with:
1. FinBERT news features (number_of_articles, mean_sentiment, pos/neg/neu_mean, has_news).
2. Chroma historical context features (market_similarity, stock_similarity).

Strictly enforces temporal alignment without forward leakage.
Includes daily context lookup caching for high performance.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import pytz

from app.ml.news_processor import (
    LocalNewsCacheProvider,
    FinBERTSentimentEngine,
    HistoricalNewsAggregator,
    NewsArticle,
    IST,
)
from app.ml.context_store import HistoricalContextStore, build_stock_indicator_vector

logger = logging.getLogger(__name__)

JOIN_COLUMNS_TO_DROP = [
    "number_of_articles",
    "mean_sentiment",
    "positive_probability_mean",
    "negative_probability_mean",
    "neutral_probability_mean",
    "latest_news_timestamp",
    "has_news",
    "market_similarity",
    "stock_similarity",
]


class Phase4FeatureJoiner:
    """Joins technical, FinBERT news, and Chroma context features into a unified dataset."""

    def __init__(
        self,
        news_provider=None,
        sentiment_engine=None,
        context_store=None,
    ):
        self.news_provider = news_provider or LocalNewsCacheProvider()
        self.sentiment_engine = sentiment_engine or FinBERTSentimentEngine()
        self.context_store = context_store or HistoricalContextStore()

    def build_joined_dataset(self, df_processed: pd.DataFrame) -> pd.DataFrame:
        """Joins 9 technical features with real FinBERT news and Chroma historical context.

        Returns new DataFrame with additional columns:
          - number_of_articles
          - mean_sentiment (NaN when has_news=False)
          - positive_probability_mean (NaN when has_news=False)
          - negative_probability_mean (NaN when has_news=False)
          - neutral_probability_mean (NaN when has_news=False)
          - latest_news_timestamp
          - has_news
          - market_similarity
          - stock_similarity
        """
        if df_processed.empty:
            return pd.DataFrame()

        df = df_processed.copy()
        # Drop any existing joined columns to prevent duplicate column names
        cols_to_drop = [c for c in JOIN_COLUMNS_TO_DROP if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        if "trading_date" not in df.columns:
            df["trading_date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")

        # 1. Fetch & score news per symbol
        symbols = df["symbol"].unique()
        symbol_news_map: Dict[str, List] = {}

        for sym in symbols:
            clean_sym = str(sym).upper().strip()
            articles = self.news_provider.fetch_news_for_symbol(clean_sym)
            articles_with_sent = []
            for art in articles:
                sent = self.sentiment_engine.analyze_text(art.headline)
                articles_with_sent.append((art, sent))
            symbol_news_map[clean_sym] = articles_with_sent

        # 2. Daily context similarity lookup cache
        daily_df = df.groupby(["symbol", "trading_date"]).last().reset_index()
        context_cache: Dict[Tuple[str, str], Dict[str, float]] = {}

        for idx, row in daily_df.iterrows():
            sym = str(row["symbol"]).upper().strip()
            date_str = str(row["trading_date"])
            vec = build_stock_indicator_vector(row.to_dict())

            mkt_sim = self.context_store.query_market_similarity(vec, date_str)
            stk_sim = self.context_store.query_stock_similarity(sym, vec, date_str)
            context_cache[(sym, date_str)] = {
                "market_similarity": mkt_sim,
                "stock_similarity": stk_sim,
            }

        # 3. Join news and cached context per candle row
        news_feat_rows = []
        context_feat_rows = []

        for idx, row in df.iterrows():
            sym = str(row["symbol"]).upper().strip()
            dt_raw = pd.to_datetime(row["timestamp"])
            if dt_raw.tzinfo is None:
                dt_ist = IST.localize(dt_raw)
            else:
                dt_ist = dt_raw.astimezone(IST)

            date_str = str(row["trading_date"])

            # A. News Aggregation (strict: news_timestamp < candle_timestamp)
            art_sent_list = symbol_news_map.get(sym, [])
            news_stats = HistoricalNewsAggregator.aggregate_news_for_candle(
                art_sent_list, dt_ist
            )
            news_feat_rows.append(news_stats)

            # B. Historical Similarity Lookup (cached per symbol-date)
            ctx_sim = context_cache.get((sym, date_str), {"market_similarity": 0.0, "stock_similarity": 0.0})
            context_feat_rows.append(ctx_sim)

        news_df = pd.DataFrame(news_feat_rows)
        context_df = pd.DataFrame(context_feat_rows)

        joined_df = pd.concat([df.reset_index(drop=True), news_df, context_df], axis=1)
        logger.info(
            "Phase4FeatureJoiner produced %d rows with %d columns.",
            len(joined_df),
            len(joined_df.columns),
        )
        return joined_df
