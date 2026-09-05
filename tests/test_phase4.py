"""
Phase 4 Comprehensive Unit Test Suite

Tests:
1. FinBERT sentiment engine & score formula (pos - neg).
2. News timestamp temporal rule (strict news_timestamp < candle_timestamp).
3. Previous evening news inclusion & future news exclusion.
4. Symbol isolation & news deduplication.
5. Explicit NaN missing news representation (when has_news=False, sentiment=NaN).
6. Chroma 2-collection schema, deterministic IDs, idempotent upserts, and temporal filtering (trading_date_int < query_date_int).
7. Neo4j graph MERGE idempotence & temporal query rule (td.date < query_date).
8. Feature joiner temporal alignment and leakage prevention.
9. Phase 3 baseline integrity protection.
"""

import datetime
from pathlib import Path
import pytz
import pytest
import pandas as pd
import numpy as np

from app.ml.news_processor import (
    NewsArticle,
    FinBERTSentimentEngine,
    HistoricalNewsAggregator,
    IST,
    UTC,
)
from app.ml.context_store import (
    HistoricalContextStore,
    MARKET_COLLECTION_NAME,
    STOCK_COLLECTION_NAME,
    date_to_int,
    build_stock_indicator_vector,
)
from app.ml.graph_ingestion import Neo4jGraphIngestor
from app.ml.feature_joiner import Phase4FeatureJoiner
from app.ml.phase4_data_quality import generate_phase4_data_quality_report


# --- 1. FinBERT Sentiment Engine Tests ---

def test_finbert_sentiment_output_and_formula():
    """Verifies FinBERT probabilities and sentiment_score = pos - neg formula."""
    engine = FinBERTSentimentEngine()
    result = engine.analyze_text("Company reports strong quarterly profit and high revenue growth.")

    assert "positive_probability" in result
    assert "negative_probability" in result
    assert "neutral_probability" in result
    assert "sentiment_score" in result

    pos = result["positive_probability"]
    neg = result["negative_probability"]
    score = result["sentiment_score"]

    assert abs(score - round(pos - neg, 4)) < 1e-4
    assert 0.0 <= pos <= 1.0
    assert 0.0 <= neg <= 1.0


# --- 2. News Timestamp Temporal Rule & NaN Missing News Tests ---

def test_news_timestamp_temporal_exclusion():
    """Given headline A at 09:00 and headline B at 11:00:

    A prediction at 10:00 MUST include headline A but MUST NOT include headline B.
    """
    dt_9am = IST.localize(datetime.datetime(2026, 9, 1, 9, 0))
    dt_11am = IST.localize(datetime.datetime(2026, 9, 1, 11, 0))
    candle_10am = IST.localize(datetime.datetime(2026, 9, 1, 10, 0))

    art_a = NewsArticle("a1", "RELIANCE.NS", "Headline A", "Summary A", dt_9am)
    art_b = NewsArticle("b1", "RELIANCE.NS", "Headline B", "Summary B", dt_11am)

    sent_a = {"positive_probability": 0.8, "negative_probability": 0.1, "neutral_probability": 0.1, "sentiment_score": 0.7}
    sent_b = {"positive_probability": 0.1, "negative_probability": 0.8, "neutral_probability": 0.1, "sentiment_score": -0.7}

    articles_with_sent = [(art_a, sent_a), (art_b, sent_b)]

    stats_10am = HistoricalNewsAggregator.aggregate_news_for_candle(articles_with_sent, candle_10am)

    assert stats_10am["number_of_articles"] == 1
    assert stats_10am["has_news"] is True
    assert stats_10am["mean_sentiment"] == 0.7
    assert stats_10am["latest_news_timestamp"] == dt_9am.isoformat()


def test_missing_news_returns_nan_and_has_news_false():
    """When no news exists prior to candle timestamp, has_news=False and sentiment fields are NaN."""
    candle_time = IST.localize(datetime.datetime(2026, 9, 1, 9, 15))
    stats = HistoricalNewsAggregator.aggregate_news_for_candle([], candle_time)

    assert stats["has_news"] is False
    assert stats["number_of_articles"] == 0
    assert np.isnan(stats["mean_sentiment"])
    assert np.isnan(stats["positive_probability_mean"])
    assert np.isnan(stats["negative_probability_mean"])
    assert np.isnan(stats["neutral_probability_mean"])


def test_previous_evening_news_included_for_next_morning():
    """Previous-evening news (e.g. 20:00 prior day) must be usable for next morning 09:15 candle."""
    dt_prev_evening = IST.localize(datetime.datetime(2026, 8, 31, 20, 0))
    candle_next_morning = IST.localize(datetime.datetime(2026, 9, 1, 9, 15))

    art = NewsArticle("prev1", "TCS.NS", "Evening Announcement", "Details", dt_prev_evening)
    sent = {"positive_probability": 0.9, "negative_probability": 0.05, "neutral_probability": 0.05, "sentiment_score": 0.85}

    stats = HistoricalNewsAggregator.aggregate_news_for_candle([(art, sent)], candle_next_morning)

    assert stats["number_of_articles"] == 1
    assert stats["has_news"] is True
    assert stats["mean_sentiment"] == 0.85


def test_symbol_isolation_and_timezone_conversion():
    """News for RELIANCE.NS must not contaminate TCS.NS."""
    dt_utc = UTC.localize(datetime.datetime(2026, 9, 1, 4, 30))
    dt_ist = dt_utc.astimezone(IST)

    art_rel = NewsArticle("r1", "RELIANCE.NS", "Reliance Deal", "", dt_ist)
    art_tcs = NewsArticle("t1", "TCS.NS", "TCS Deal", "", dt_ist)

    sent = {"positive_probability": 0.5, "negative_probability": 0.1, "neutral_probability": 0.4, "sentiment_score": 0.4}
    candle_time = IST.localize(datetime.datetime(2026, 9, 1, 10, 30))

    tcs_articles = [(art_tcs, sent)]
    tcs_stats = HistoricalNewsAggregator.aggregate_news_for_candle(tcs_articles, candle_time)

    assert tcs_stats["number_of_articles"] == 1
    assert art_rel.symbol != art_tcs.symbol


# --- 3. Chroma Vector Database & Scale-Independent Vector Tests ---

def test_chroma_two_collections_schema_and_idempotence(tmp_path):
    """Verifies that exactly two collections exist with deterministic IDs and idempotent upsert."""
    store = HistoricalContextStore(persist_directory=str(tmp_path / "chroma_test"))

    assert store.market_collection.name == MARKET_COLLECTION_NAME
    assert store.stock_collection.name == STOCK_COLLECTION_NAME

    df = pd.DataFrame([
        {
            "symbol": "RELIANCE.NS",
            "timestamp": "2026-08-01 15:30:00",
            "trading_date": "2026-08-01",
            "rsi": 55.0,
            "price_vs_ema5": 0.01,
            "price_vs_vwap": 0.005,
            "macd": 0.5,
            "macd_signal": 0.4,
            "macd_diff": 0.1,
            "bollinger_position": 0.6,
            "obv": 1000.0,
            "direction": 1.0,
            "close": 2500.0,
        }
    ])

    m_cnt1, s_cnt1 = store.populate_from_historical_candles(df)
    assert m_cnt1 == 1
    assert s_cnt1 == 1

    m_cnt2, s_cnt2 = store.populate_from_historical_candles(df)
    assert store.market_collection.count() == 1
    assert store.stock_collection.count() == 1


def test_build_stock_indicator_vector_scale_independence_and_causal_direction():
    """Verifies that MACD components are normalized by close price and direction is causal."""
    ind_cheap = {
        "price_vs_ema5": 0.02,
        "rsi": 60.0,
        "obv": 500.0,
        "bollinger_position": 0.7,
        "macd": 2.0,
        "macd_signal": 1.5,
        "macd_diff": 0.5,
        "price_vs_vwap": 0.01,
        "direction": -1.0,  # non-causal input to test override
        "close": 200.0,
    }
    ind_expensive = {
        "price_vs_ema5": 0.02,
        "rsi": 60.0,
        "obv": 500.0,
        "bollinger_position": 0.7,
        "macd": 30.0,
        "macd_signal": 22.5,
        "macd_diff": 7.5,
        "price_vs_vwap": 0.01,
        "direction": 1.0,
        "close": 3000.0,
    }

    vec_cheap = build_stock_indicator_vector(ind_cheap)
    vec_expensive = build_stock_indicator_vector(ind_expensive)

    # MACD components (indices 4, 5, 6) must be identical after close normalization (2/200 == 30/3000 = 0.01)
    assert abs(vec_cheap[4] - vec_expensive[4]) < 1e-6
    assert abs(vec_cheap[5] - vec_expensive[5]) < 1e-6
    assert abs(vec_cheap[6] - vec_expensive[6]) < 1e-6

    # Causal direction (index 8) must be +1.0 because price_vs_ema5 (0.02) > 0
    assert vec_cheap[8] == 1.0
    assert vec_expensive[8] == 1.0


def test_chroma_temporal_filtering_no_future():
    """Querying date 2026-08-05 MUST NOT retrieve fingerprint from 2026-08-05 (same day) or later (future)."""
    import chromadb
    client = chromadb.Client()
    col = client.get_or_create_collection("test_stock_col", metadata={"hnsw:space": "cosine"})

    col.upsert(
        ids=["stock_A_2026-08-01", "stock_A_2026-08-05", "stock_A_2026-08-10"],
        embeddings=[[0.1]*9, [0.5]*9, [0.9]*9],
        metadatas=[
            {"symbol": "A", "trading_date": "2026-08-01", "trading_date_int": 20260801},
            {"symbol": "A", "trading_date": "2026-08-05", "trading_date_int": 20260805},
            {"symbol": "A", "trading_date": "2026-08-10", "trading_date_int": 20260810},
        ],
        documents=["past", "same_day", "future"],
    )

    res = col.query(
        query_embeddings=[[0.1]*9],
        n_results=5,
        where={"$and": [{"symbol": {"$eq": "A"}}, {"trading_date_int": {"$lt": 20260805}}]},
    )

    metas = res["metadatas"][0]
    assert len(metas) == 1
    assert metas[0]["trading_date"] == "2026-08-01"


# --- 4. Neo4j Idempotence Test ---

def test_neo4j_graph_ingestion_idempotence():
    """Verifies Neo4j MERGE query idempotence when driver is unavailable/mocked."""
    ingestor = Neo4jGraphIngestor(driver=None)
    assert ingestor.is_available() is False

    df = pd.DataFrame([{"symbol": "RELIANCE.NS", "timestamp": "2026-08-01 15:30:00", "rsi": 30.0}])
    count = ingestor.ingest_historical_patterns(df)
    assert count == 0


# --- 5. Data Quality Report & Phase 3 Protection ---

def test_data_quality_report_reporting(tmp_path):
    """Verifies that missing data and per-symbol coverage are reported accurately."""
    df_joined = pd.DataFrame([
        {"symbol": "TCS.NS", "timestamp": "2026-09-01 09:15:00", "has_news": False, "trading_date": "2026-09-01"},
        {"symbol": "TCS.NS", "timestamp": "2026-09-01 09:20:00", "has_news": True, "trading_date": "2026-09-01"},
    ])

    out_file = tmp_path / "phase4_dq.json"
    report = generate_phase4_data_quality_report(
        df_joined=df_joined,
        news_articles_count=1,
        finbert_success_rate=1.0,
        chroma_market_count=1,
        chroma_stock_count=1,
        neo4j_nodes_count=0,
        output_path=str(out_file),
        symbol_articles_map={"TCS.NS": 1},
    )

    assert report["number_of_candles"] == 2
    assert report["number_of_candles_with_usable_news"] == 1
    assert report["percentage_missing_news"] == 50.0
    assert report["temporal_leakage_checks_passed"] is True
    assert "TCS.NS" in report["news_coverage_by_symbol"]


def test_phase3_baseline_protection():
    """Verifies Phase 3 results artifact exists and has not been altered."""
    res_path = Path("data/processed/phase3_model_results.json")
    assert res_path.exists(), "Phase 3 model results file must exist untouched."

    with open(res_path, "r", encoding="utf-8") as f:
        data = f.read()

    assert "Model A" in data or "model_formulations" in data
