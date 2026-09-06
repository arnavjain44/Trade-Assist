from app.core.constraints import constraint_enforcer

def test_intraday_constraint_enforcement():
    total_capital = 10000.0
    raw_picks = [
        {"symbol": "RELIANCE.NS", "action": "BUY", "confidence": 90.0, "current_price": 2500.0, "hold_until": "next_day"},
        {"symbol": "TCS.NS", "action": "BUY", "confidence": 70.0, "current_price": 3800.0, "hold_until": "next_day"}
    ]

    enforced = constraint_enforcer.enforce_intraday_constraints(raw_picks, total_capital)

    assert len(enforced) == 2
    for pick in enforced:
        # Mandatory same day exit
        assert pick["hold_until"] == "same_day"
        # Position cap max ~40% (4000.0 max allocated out of 10000.0)
        assert pick["allocated_capital"] <= (total_capital * 0.40) + 1.0
        assert pick["shares_to_trade"] >= 0
