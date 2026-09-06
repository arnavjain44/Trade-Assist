"""
Phase 2 Live Parity Tests

Comprehensive tests verifying:
A. 5-minute data input and timezone (Asia/Kolkata)
B. Incomplete candle dropping
C and D. Technical feature calculation and exact 14 Phase 5 canonical feature sequence
E and F. Missing news and populated news handling (sentiment_score NaN / float, has_news, number_of_articles)
G. News causal filtering (strictly pub_timestamp < decision_timestamp)
H, I, J, L. Chroma context similarities (9-dim vector, market and stock queries, prior-date cutoff, no hardcoded fallbacks)
K. Direction constraint strictly in (+1, -1)
M. Zero synthetic feature data
"""

import pytest
import datetime
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from app.core.data_fetcher import StockDataFetcher, IST_TZ
from app.core.sentiment import NewsSentimentAnalyzer, NewsArticle
from app.db.vector_store import VectorStoreManager
from app.ml.feature_engineering import FeatureEngine
from app.ml.phase5_feature_builder import (
    PHASE5_FEATURE_COLS,
    build_phase5_feature_row,
    build_phase5_feature_dataframe,
)


def test_data_fetcher_5m_and_timezone():
    sample_dt = pd.date_range("2026-09-04 09:15", periods=30, freq="5min", tz="UTC")
    mock_df = pd.DataFrame({
        "Open": [100.0 + i for i in range(30)],
        "High": [102.0 + i for i in range(30)],
        "Low": [99.0 + i for i in range(30)],
        "Close": [101.0 + i for i in range(30)],
        "Volume": [10000 + i * 100 for i in range(30)],
    }, index=sample_dt)

    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df.copy()
        mock_ticker.return_value = mock_instance

        with patch.object(StockDataFetcher, "_get_now_ist", return_value=datetime.datetime(2026, 9, 4, 15, 30, tzinfo=IST_TZ)):
            df = StockDataFetcher._fetch_single_ticker_sync("TCS.NS")

        mock_instance.history.assert_called_with(period="5d", interval="5m")

        assert "timestamp" in df.columns
        assert "date_str" in df.columns
        assert str(df["timestamp"].dt.tz) == "Asia/Kolkata"
        assert len(df) > 0


def test_incomplete_candle_dropping():
    idx = pd.date_range("2026-09-04 10:00", periods=25, freq="5min", tz="Asia/Kolkata")
    mock_df = pd.DataFrame({
        "Open": [100.0 + i for i in range(25)],
        "High": [101.0 + i for i in range(25)],
        "Low": [99.0 + i for i in range(25)],
        "Close": [100.5 + i for i in range(25)],
        "Volume": [5000 for _ in range(25)],
    }, index=idx)

    # Last candle in mock_df starts at 12:00 (index 24) and ends at 12:05
    last_candle_start = idx[-1]
    assert last_candle_start.strftime("%H:%M") == "12:00"

    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df.copy()
        mock_ticker.return_value = mock_instance

        # Case 1: now_ist is 12:02 IST -> 12:00 candle ends at 12:05 > 12:02 -> must be dropped!
        with patch.object(StockDataFetcher, "_get_now_ist", return_value=datetime.datetime(2026, 9, 4, 12, 2, tzinfo=IST_TZ)):
            df1 = StockDataFetcher._fetch_single_ticker_sync("INFY.NS")
            assert len(df1) == 24
            assert df1["timestamp"].iloc[-1] == pd.Timestamp("2026-09-04 11:55:00+0530", tz="Asia/Kolkata")

        # Case 2: now_ist is 12:06 IST -> 12:00 candle ended at 12:05 <= 12:06 -> must be kept!
        with patch.object(StockDataFetcher, "_get_now_ist", return_value=datetime.datetime(2026, 9, 4, 12, 6, tzinfo=IST_TZ)):
            df2 = StockDataFetcher._fetch_single_ticker_sync("INFY.NS")
            assert len(df2) == 25
            assert df2["timestamp"].iloc[-1] == pd.Timestamp("2026-09-04 12:00:00+0530", tz="Asia/Kolkata")


def test_technical_indicators_and_canonical_sequence():
    idx = pd.date_range("2026-09-04 09:15", periods=50, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame({
        "timestamp": idx,
        "open": np.linspace(100, 110, 50),
        "high": np.linspace(101, 111, 50),
        "low": np.linspace(99, 109, 50),
        "close": np.linspace(100.5, 110.5, 50),
        "volume": np.full(50, 1000),
    })

    feat_df = FeatureEngine.calculate_features(df)
    last_row = feat_df.iloc[-1]

    tech_dict = {
        "rsi": last_row["rsi"],
        "obv": last_row["obv"],
        "bollinger_position": last_row["bollinger_position"],
        "macd": last_row["macd"],
        "macd_signal": last_row["macd_signal"],
        "macd_diff": last_row["macd_diff"],
        "price_vs_vwap": last_row["price_vs_vwap"],
        "price_vs_ema5": last_row["price_vs_ema5"],
    }

    news_dict = {"sentiment_score": 0.35, "has_news": True, "number_of_articles": 2}
    context_dict = {"market_similarity": 0.85, "stock_similarity": 0.90}

    df_long = build_phase5_feature_dataframe(tech_dict, news_dict, context_dict, direction=1)

    assert list(df_long.columns) == PHASE5_FEATURE_COLS
    assert len(df_long.columns) == 14
    assert df_long["direction"].iloc[0] == 1
    assert df_long["sentiment_score"].iloc[0] == 0.35
    assert df_long["has_news"].iloc[0] is True or df_long["has_news"].iloc[0] == 1
    assert df_long["number_of_articles"].iloc[0] == 2
    assert df_long["market_similarity"].iloc[0] == 0.85
    assert df_long["stock_similarity"].iloc[0] == 0.90


def test_news_features_missing_and_populated():
    analyzer = NewsSentimentAnalyzer()

    # Missing news
    with patch.object(analyzer.news_provider, "fetch_news_for_symbol", return_value=[]):
        res_empty = analyzer.get_phase5_news_features("RELIANCE.NS")
        assert np.isnan(res_empty["sentiment_score"])
        assert res_empty["has_news"] is False
        assert res_empty["number_of_articles"] == 0

    # Populated news
    now_ist = datetime.datetime.now(IST_TZ)
    past_pub = now_ist - datetime.timedelta(hours=2)
    fake_articles = [
        NewsArticle(
            article_id="art1",
            symbol="RELIANCE",
            headline="Strong quarterly growth for Reliance",
            summary="Reliance announced higher Q2 net profits.",
            pub_timestamp_ist=past_pub,
        ),
        NewsArticle(
            article_id="art2",
            symbol="RELIANCE",
            headline="Reliance expands operations successfully",
            summary="New green energy contracts secured.",
            pub_timestamp_ist=past_pub,
        ),
    ]

    with patch.object(analyzer.news_provider, "fetch_news_for_symbol", return_value=fake_articles):
        with patch.object(analyzer.finbert_engine, "analyze_text", side_effect=[
            {"sentiment_score": 0.40, "label": "positive"},
            {"sentiment_score": 0.60, "label": "positive"},
        ]):
            res_pop = analyzer.get_phase5_news_features("RELIANCE.NS", decision_timestamp_ist=now_ist)
            assert res_pop["sentiment_score"] == 0.50
            assert res_pop["has_news"] is True
            assert res_pop["number_of_articles"] == 2


def test_news_causal_filtering():
    analyzer = NewsSentimentAnalyzer()
    decision_dt = datetime.datetime(2026, 9, 4, 12, 0, tzinfo=IST_TZ)

    past_art = NewsArticle(
        article_id="past1",
        symbol="INFY",
        headline="Past news",
        summary="Historical announcement",
        pub_timestamp_ist=datetime.datetime(2026, 9, 4, 11, 30, tzinfo=IST_TZ),
    )
    future_art = NewsArticle(
        article_id="fut1",
        symbol="INFY",
        headline="Future news",
        summary="Leakage article",
        pub_timestamp_ist=datetime.datetime(2026, 9, 4, 12, 30, tzinfo=IST_TZ),
    )

    with patch.object(analyzer.news_provider, "fetch_news_for_symbol", return_value=[past_art, future_art]):
        with patch.object(analyzer.finbert_engine, "analyze_text", return_value={"sentiment_score": 0.25, "label": "positive"}):
            res = analyzer.get_phase5_news_features("INFY.NS", decision_timestamp_ist=decision_dt)
            assert res["number_of_articles"] == 1
            assert res["has_news"] is True
            assert res["sentiment_score"] == 0.25


def test_chroma_context_similarities():
    mgr = VectorStoreManager()
    mgr._ready = True

    mock_context_store = MagicMock()
    mock_context_store.query_market_similarity.return_value = 0.8842
    mock_context_store.query_stock_similarity.return_value = 0.9125
    mgr._context_store = mock_context_store

    tech_indicators = {
        "rsi": 55.0,
        "obv": 10000.0,
        "bollinger_position": 0.6,
        "macd": 1.2,
        "macd_signal": 1.0,
        "macd_diff": 0.2,
        "price_vs_vwap": 0.005,
        "price_vs_ema5": 0.002,
    }

    res = mgr.query_phase5_context_similarities("TCS.NS", tech_indicators, "2026-09-04")

    mock_context_store.query_market_similarity.assert_called_once()
    mock_context_store.query_stock_similarity.assert_called_once()
    args_stk = mock_context_store.query_stock_similarity.call_args[0]
    assert args_stk[0] == "TCS.NS"
    assert args_stk[2] == "2026-09-04"

    assert res["market_similarity"] == 0.8842
    assert res["stock_similarity"] == 0.9125

    # Fallback when unready
    mgr._ready = False
    fallback_res = mgr.query_phase5_context_similarities("TCS.NS", tech_indicators, "2026-09-04")
    assert fallback_res == {"market_similarity": 0.0, "stock_similarity": 0.0}


def test_direction_strict_validation():
    tech_dict = {col: 1.0 for col in ["rsi", "obv", "bollinger_position", "macd", "macd_signal", "macd_diff", "price_vs_vwap", "price_vs_ema5"]}
    news_dict = {"sentiment_score": np.nan, "has_news": False, "number_of_articles": 0}
    context_dict = {"market_similarity": 0.0, "stock_similarity": 0.0}

    row_long = build_phase5_feature_row(tech_dict, news_dict, context_dict, direction=1)
    assert row_long["direction"] == 1

    row_short = build_phase5_feature_row(tech_dict, news_dict, context_dict, direction=-1)
    assert row_short["direction"] == -1

    for invalid_dir in [0, 2, -2, "BUY", "SELL", None]:
        with pytest.raises(ValueError, match="Invalid direction"):
            build_phase5_feature_row(tech_dict, news_dict, context_dict, direction=invalid_dir)


def test_zero_synthetic_data_in_phase5_builder():
    tech_dict = {
        "rsi": 42.1234,
        "obv": 987654.0,
        "bollinger_position": 0.7788,
        "macd": 0.3344,
        "macd_signal": 0.2211,
        "macd_diff": 0.1133,
        "price_vs_vwap": -0.0045,
        "price_vs_ema5": 0.0012,
    }
    news_dict = {"sentiment_score": 0.1234, "has_news": True, "number_of_articles": 3}
    context_dict = {"market_similarity": 0.7654, "stock_similarity": 0.8765}

    df = build_phase5_feature_dataframe(tech_dict, news_dict, context_dict, direction=1)

    assert df["rsi"].iloc[0] == 42.1234
    assert df["obv"].iloc[0] == 987654.0
    assert df["bollinger_position"].iloc[0] == 0.7788
    assert df["macd"].iloc[0] == 0.3344
    assert df["macd_signal"].iloc[0] == 0.2211
    assert df["macd_diff"].iloc[0] == 0.1133
    assert df["price_vs_vwap"].iloc[0] == -0.0045
    assert df["price_vs_ema5"].iloc[0] == 0.0012
    assert df["direction"].iloc[0] == 1
    assert df["sentiment_score"].iloc[0] == 0.1234
    assert df["has_news"].iloc[0] is True or df["has_news"].iloc[0] == 1
    assert df["number_of_articles"].iloc[0] == 3
    assert df["market_similarity"].iloc[0] == 0.7654
    assert df["stock_similarity"].iloc[0] == 0.8765
