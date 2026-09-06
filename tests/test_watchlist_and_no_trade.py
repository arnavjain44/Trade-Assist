import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.agent.loop import agent_loop

client = TestClient(app)

def test_zero_qualified_trades_and_non_empty_watchlist():
    """Requirements A, B, C, D, E, F, H: Verify zero qualified trades produces non-empty watchlist sorted by model_probability capped at 5."""
    payload = {
        "investment_amount": 50000.0,
        "tickers": ["HDFCBANK.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS", "AXISBANK.NS"],
        "provider": "auto"
    }
    response = client.post("/api/v1/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "watchlist" in data
    
    # Recommendations should only contain P >= 0.80 candidates
    for rec in data["recommendations"]:
        assert rec["qualified"] is True
        assert rec["action"] in ["BUY", "SELL"]
        assert rec["allocated_capital"] > 0
    
    # Watchlist requirements
    watchlist = data["watchlist"]
    assert len(watchlist) <= 5
    
    if len(data["recommendations"]) == 0:
        assert data["total_allocated"] == 0.0
        assert data["unallocated_cash"] == 50000.0
        assert len(watchlist) > 0

    # Requirement E & H: Watchlist allocation always 0 and shares 0, no BUY/SELL action
    # Requirement D: Watchlist sorted descending by model_probability
    probs = []
    for item in watchlist:
        assert item["qualified"] is False
        assert item["allocated_capital"] == 0.0
        assert item["shares_to_trade"] == 0
        assert item["direction"] in ["LONG", "SHORT"]
        assert "action" not in item or item.get("action") not in ["BUY", "SELL"]
        assert item["model_threshold"] == 0.8000
        assert item["distance_to_threshold"] == round((0.8000 - item["model_probability"]) * 100.0, 2)
        probs.append(item["model_probability"])

    # Requirement D: Sorted descending
    assert probs == sorted(probs, reverse=True)

    # Requirement F: Qualified candidates not duplicated in watchlist
    rec_symbols = {r["symbol"] for r in data["recommendations"]}
    watch_symbols = {w["symbol"] for w in watchlist}
    assert rec_symbols.isdisjoint(watch_symbols)

def test_threshold_remains_eighty_percent():
    """Requirement G: Model threshold strictly remains 0.8000."""
    from app.ml.pipeline import ml_engine
    assert ml_engine.inference.threshold == 0.8000

def test_existing_allocation_constraints_unchanged():
    """Requirement I: Constraints layer capped at 40% per stock, same-day exit enforced."""
    from app.core.constraints import constraint_enforcer
    sample_picks = [
        {"symbol": "AAA.NS", "action": "BUY", "confidence": 90.0, "current_price": 100.0},
        {"symbol": "BBB.NS", "action": "BUY", "confidence": 85.0, "current_price": 200.0}
    ]
    res = constraint_enforcer.enforce_intraday_constraints(sample_picks, 10000.0)
    assert len(res) == 2
    for r in res:
        assert r["hold_until"] == "same_day"
        assert r["allocation_pct"] <= 0.40
        assert r["allocated_capital"] <= 4000.00
