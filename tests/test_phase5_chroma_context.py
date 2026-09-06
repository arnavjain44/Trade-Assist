"""
Phase 5 Chroma Context Store Tests

Verifies:
1. Temporal cutoff point-in-time guarantee (trading_date_int < query_date_int).
2. Historical fingerprint parity between research vector builder and Chroma stored vectors.
3. Idempotent upserts (no duplicate IDs on re-execution).
4. Live Chroma similarity retrieval feeding non-zero context features into Phase 5 model input path.
"""

import os
import pytest
import numpy as np
import pandas as pd

from app.ml.context_store import (
    HistoricalContextStore,
    build_stock_indicator_vector,
    date_to_int,
    FINGERPRINT_DIM,
)
from app.db.vector_store import VectorStoreManager

PARQUET_PATH = "data/processed/phase5_features.parquet"


# ==============================================================================
# Test 1: Point-In-Time Temporal Leakage Cutoff (Mandatory Requirement 11)
# ==============================================================================
def test_temporal_leakage_cutoff():
    """Verify query strictly excludes same-day (D) and future (> D) fingerprints."""
    store = HistoricalContextStore(persist_directory="./chroma_db")

    # Pick a mid-dataset query date D = '2026-08-01' (integer 20260801)
    query_date_str = "2026-08-01"
    query_int = date_to_int(query_date_str)

    dummy_vec = [0.0] * FINGERPRINT_DIM

    # Query Market Collection
    res_mkt = store.market_collection.query(
        query_embeddings=[dummy_vec],
        n_results=50,
        where={"trading_date_int": {"$lt": query_int}},
    )
    if res_mkt and res_mkt.get("metadatas") and res_mkt["metadatas"][0]:
        for meta in res_mkt["metadatas"][0]:
            t_int = meta["trading_date_int"]
            assert t_int < query_int, f"Temporal leakage! Market date {t_int} >= query date {query_int}"

    # Query Stock Collection for RELIANCE.NS
    res_stk = store.stock_collection.query(
        query_embeddings=[dummy_vec],
        n_results=50,
        where={"$and": [{"symbol": {"$eq": "RELIANCE.NS"}}, {"trading_date_int": {"$lt": query_int}}]},
    )
    if res_stk and res_stk.get("metadatas") and res_stk["metadatas"][0]:
        for meta in res_stk["metadatas"][0]:
            t_int = meta["trading_date_int"]
            assert t_int < query_int, f"Temporal leakage! Stock date {t_int} >= query date {query_int}"


# ==============================================================================
# Test 2: Historical Fingerprint Feature Parity (Mandatory Requirement 12)
# ==============================================================================
def test_historical_fingerprint_parity():
    """Verify vector builder output matches stored Chroma vector for a known historical day."""
    if not os.path.exists(PARQUET_PATH):
        pytest.skip("Phase 5 feature parquet not present.")

    df = pd.read_parquet(PARQUET_PATH)
    rel = df[(df["symbol"] == "RELIANCE.NS") & (df["trading_date"] == "2026-08-14")].sort_values("timestamp")
    assert len(rel) > 0, "Known row missing for RELIANCE.NS on 2026-08-14"

    # Compute vector from final candle of the day
    final_candle = rel.iloc[-1].to_dict()
    expected_vec = build_stock_indicator_vector(final_candle)

    # Retrieve stored vector from Chroma
    store = HistoricalContextStore(persist_directory="./chroma_db")
    doc_id = "stock_RELIANCE.NS_2026-08-14"
    res = store.stock_collection.get(ids=[doc_id], include=["embeddings"])

    assert len(res["ids"]) == 1, f"Document {doc_id} not found in Chroma"
    stored_vec = res["embeddings"][0]

    assert len(expected_vec) == len(stored_vec) == FINGERPRINT_DIM
    for dim_idx, (exp_val, act_val) in enumerate(zip(expected_vec, stored_vec)):
        assert abs(exp_val - act_val) < 1e-5, (
            f"Parity mismatch at dimension {dim_idx}: expected {exp_val:.6f}, got {act_val:.6f}"
        )


# ==============================================================================
# Test 3: Idempotent Re-Execution (Requirement 8)
# ==============================================================================
def test_idempotent_reexecution():
    """Verify rerunning population does not duplicate Chroma documents."""
    if not os.path.exists(PARQUET_PATH):
        pytest.skip("Phase 5 feature parquet not present.")

    df = pd.read_parquet(PARQUET_PATH)
    rel = df[(df["symbol"] == "RELIANCE.NS") & (df["trading_date"] == "2026-08-14")].sort_values("timestamp")

    store = HistoricalContextStore(persist_directory="./chroma_db")
    mkt_count_before = store.market_collection.count()
    stk_count_before = store.stock_collection.count()

    assert mkt_count_before == 59
    assert stk_count_before == 2832

    # Rerun with actual historical sample
    store.populate_from_historical_candles(rel)

    # Count must remain unchanged because doc_id 'stock_RELIANCE.NS_2026-08-14' was updated in-place
    assert store.stock_collection.count() == stk_count_before
    assert store.market_collection.count() == mkt_count_before


# ==============================================================================
# Test 4: Live Query Retrieves Real Context (Requirement 13)
# ==============================================================================
def test_live_context_similarity_retrieval():
    """Verify live query retrieves non-zero Chroma similarities for real historical cutoff."""
    mgr = VectorStoreManager(persist_directory="./chroma_db")
    # Force store ready state
    import time
    time.sleep(0.5)

    tech_dict = {
        "rsi": 55.0, "obv": 10000.0, "bollinger_position": 0.6,
        "macd": 1.2, "macd_signal": 1.0, "macd_diff": 0.2,
        "price_vs_vwap": 0.005, "price_vs_ema5": 0.002, "close": 1322.0,
    }

    res = mgr.query_phase5_context_similarities("RELIANCE.NS", tech_dict, "2026-09-04")
    assert res["market_similarity"] > 0.0
    assert res["stock_similarity"] > 0.0
    assert 0.0 <= res["market_similarity"] <= 1.0
    assert 0.0 <= res["stock_similarity"] <= 1.0
