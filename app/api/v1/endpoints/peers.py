import asyncio
from fastapi import APIRouter, HTTPException
from app.schemas.requests import PeersRequest
from app.agent.tools import agent_tools
from app.db.vector_store import SYMBOL_SECTOR_MAP, build_fingerprint_vector, vector_store
from typing import List, Dict, Any

router = APIRouter()


@router.post("/peers", response_model=List[Dict[str, Any]])
async def get_sector_peers(request: PeersRequest):
    """
    Finds sector peers for a given stock using ChromaDB vector similarity.

    1. Looks up the stock's stored fingerprint in Chroma (upserted during last recommendation run).
    2. Filters peers to the same sector.
    3. Runs the full pipeline (fetch → indicators → sentiment → ML predict) on each peer.
    4. Returns a comparison table payload — one entry per stock including the original.

    Falls back gracefully:
    - If symbol has no sector mapping → 404 with clear message.
    - If Chroma is not ready → returns empty peers list with a message.
    - Per-peer failures are skipped (logged server-side).
    """
    symbol = request.symbol.upper().strip()

    # Validate sector mapping
    sector = SYMBOL_SECTOR_MAP.get(symbol)
    if not sector:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No sector mapping found for '{symbol}'. "
                f"Peer comparison is available for: {', '.join(sorted(SYMBOL_SECTOR_MAP.keys()))}"
            )
        )

    # Fetch the original stock's current live data to build its fingerprint vector for querying
    try:
        original_data = await asyncio.wait_for(
            agent_tools.fetch_and_calculate_indicators([symbol]),
            timeout=15.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Timed out fetching live data for {symbol}.")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch live data for {symbol}: {exc}")

    if symbol not in original_data:
        raise HTTPException(status_code=502, detail=f"No data returned for {symbol}.")

    orig_info = original_data[symbol]
    orig_ind = orig_info["latest_indicators"]
    orig_sent = orig_info["sentiment_score"]
    orig_vec = build_fingerprint_vector(orig_ind)

    # Upsert original stock's fingerprint (keeps Chroma fresh)
    agent_tools.upsert_fingerprint(symbol, orig_ind)

    # Get original stock's prediction
    orig_vec_res = vector_store.query_similar_patterns(symbol, [orig_ind["close_price"], orig_sent, orig_ind["rsi"]])
    orig_pred = agent_tools.predict(symbol, orig_ind["close_price"], orig_ind, orig_sent, orig_vec_res.get("similarity_score", 0.85))

    original_row = {
        "symbol": symbol,
        "sector": sector,
        "similarity_score": 1.0,  # The query stock itself is 100% similar to itself
        "confidence": orig_pred["confidence"],
        "action": orig_pred["action"],
        "current_price": orig_ind["close_price"],
        "ema_5": orig_ind["ema_5"],
        "rsi": orig_ind["rsi"],
        "macd": orig_ind["macd"],
        "vwap": orig_ind["vwap"],
        "bb_upper": orig_ind["bb_upper"],
        "bb_lower": orig_ind["bb_lower"],
        "sentiment_score": round(orig_sent, 3),
        "overall_bias": orig_ind["overall_technical_bias"],
        "is_original": True,
    }

    # Find sector peers via Chroma similarity
    peers = await agent_tools.get_sector_peers(
        symbol=symbol,
        indicator_vector=orig_vec,
        max_peers=request.max_peers,
    )

    # Mark peers as not original
    for p in peers:
        p["is_original"] = False

    # Original stock first, then peers sorted by similarity
    return [original_row] + sorted(peers, key=lambda x: x["similarity_score"], reverse=True)
