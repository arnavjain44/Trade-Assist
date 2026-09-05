from typing import List, Dict, Any
from app.config import settings

class ConstraintEnforcer:
    """
    Deterministic constraint enforcement layer (Section 5 & 6.6 in Spec).
    Runs as a mandatory post-processing step AFTER model/agent recommendations.
    """

    @staticmethod
    def enforce_intraday_constraints(raw_recommendations: List[Dict[str, Any]], total_capital: float) -> List[Dict[str, Any]]:
        """
        Applies non-bypassable intraday rules:
        1. Force hold_until = "same_day" on all positions.
        2. Filter only actionable trades (BUY / SELL recommendations).
        3. Confidence-weighted position allocation capped at MAX_POSITION_ALLOCATION_PCT (40%).
        4. Calculates exact share count based on allocated capital and current stock price.
        5. Prevents float allocation overflow so unallocated cash is never negative.
        """
        processed = []
        buy_sell_picks = [r for r in raw_recommendations if r.get("action") in ["BUY", "SELL"]]
        
        if not buy_sell_picks:
            return []

        # Calculate initial confidence weights
        total_confidence = sum(p.get("confidence", 50.0) for p in buy_sell_picks)
        if total_confidence == 0:
            total_confidence = len(buy_sell_picks) * 50.0

        raw_allocations = {}
        for pick in buy_sell_picks:
            weight = pick.get("confidence", 50.0) / total_confidence
            raw_allocations[pick["symbol"]] = weight

        # Apply maximum allocation cap per stock (e.g. 40% cap)
        cap_pct = settings.MAX_POSITION_ALLOCATION_PCT
        capped_allocations = {}
        excess_capital_pct = 0.0

        for symbol, pct in raw_allocations.items():
            if pct > cap_pct:
                excess_capital_pct += (pct - cap_pct)
                capped_allocations[symbol] = cap_pct
            else:
                capped_allocations[symbol] = pct

        # Redistribute excess proportionally to remaining uncapped stocks
        uncapped_symbols = [s for s, p in capped_allocations.items() if p < cap_pct]
        if uncapped_symbols and excess_capital_pct > 0:
            uncapped_weight_sum = sum(capped_allocations[s] for s in uncapped_symbols)
            if uncapped_weight_sum > 0:
                for s in uncapped_symbols:
                    additional = (capped_allocations[s] / uncapped_weight_sum) * excess_capital_pct
                    capped_allocations[s] = min(cap_pct, capped_allocations[s] + additional)

        # Normalize allocations so total sum never exceeds 1.0 (100%)
        total_alloc_sum = sum(capped_allocations.values())
        if total_alloc_sum > 1.0:
            for s in capped_allocations:
                capped_allocations[s] = capped_allocations[s] / total_alloc_sum

        # Build final trade recommendation payloads
        accumulated_allocated = 0.0
        for i, pick in enumerate(buy_sell_picks):
            item = pick.copy()
            
            if item.get("hold_until") != "same_day":
                item["hold_until"] = "same_day"
                item["forced_override"] = True
            
            symbol = item["symbol"]
            alloc_pct = round(capped_allocations.get(symbol, 0.0), 4)
            
            # Ensure last item absorbs any tiny floating point rounding differences
            if i == len(buy_sell_picks) - 1:
                allocated_capital = round(min(total_capital - accumulated_allocated, total_capital * alloc_pct), 2)
            else:
                allocated_capital = round(total_capital * alloc_pct, 2)

            accumulated_allocated += allocated_capital
            current_price = item.get("current_price", 100.0)
            shares = int(allocated_capital // current_price) if current_price > 0 else 0
            
            item["allocation_pct"] = alloc_pct
            item["allocated_capital"] = allocated_capital
            item["shares_to_trade"] = shares
            
            if item["action"] == "BUY":
                if item.get("target_price", 0) <= current_price:
                    item["target_price"] = round(current_price * 1.022, 2)
                if item.get("stop_loss", 0) >= current_price:
                    item["stop_loss"] = round(current_price * 0.991, 2)
            elif item["action"] == "SELL":
                if item.get("target_price", 0) >= current_price:
                    item["target_price"] = round(current_price * 0.978, 2)
                if item.get("stop_loss", 0) <= current_price:
                    item["stop_loss"] = round(current_price * 1.009, 2)

            processed.append(item)

        return processed

constraint_enforcer = ConstraintEnforcer()
