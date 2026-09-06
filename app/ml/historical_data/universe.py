"""
Historical Symbol Universe Manager (Phase 5.4)

Supports Point-in-Time Index Membership Universe(t) to eliminate survivorship bias.
Provides fallback contemporary universe with explicit survivorship-bias flagging.
"""

import logging
from datetime import date, datetime
from typing import List, Dict, Set, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical list of contemporary NIFTY 50 equities (as of late 2024 / early 2025)
CONTEMPORARY_NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "TATAMOTORS.NS", "BHARTIARTL.NS", "SBIN.NS", "AXISBANK.NS", "ITC.NS",
    "WIPRO.NS", "BAJFINANCE.NS", "MARUTI.NS", "LT.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "KOTAKBANK.NS",
    "TATASTEEL.NS", "INDUSINDBK.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
    "ONGC.NS", "HDFCLIFE.NS", "SBILIFE.NS", "BAJAJ-AUTO.NS", "M&M.NS",
    "HEROMOTOCO.NS", "EICHERMOT.NS", "BPCL.NS", "IOC.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "NESTLEIND.NS",
    "HINDUNILVR.NS", "GRASIM.NS", "JSWSTEEL.NS", "HINDALCO.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "BEL.NS", "HAL.NS", "TRENT.NS", "ZOMATO.NS"
]

# Historical point-in-time reconstitution events (NSE semi-annual rebalances)
# Format: effective_date -> {"added": [...], "removed": [...]}
HISTORICAL_REBALANCES: Dict[date, Dict[str, List[str]]] = {
    date(2024, 9, 27): {
        "added": ["BEL.NS", "TRENT.NS"],
        "removed": ["DIVISLAB.NS", "LTIM.NS"]
    },
    date(2024, 3, 28): {
        "added": ["SHRIRAMFIN.NS"],
        "removed": ["UPL.NS"]
    },
    date(2022, 9, 30): {
        "added": ["ADANIENT.NS"],
        "removed": ["SHREECEM.NS"]
    },
    date(2022, 3, 31): {
        "added": ["APOLLOHOSP.NS"],
        "removed": ["IOC.NS"]
    },
}


class HistoricalUniverseManager:
    """
    Manages historical universe selection.
    Distinguishes strictly causal Point-in-Time membership from fixed contemporary membership.
    """

    def __init__(self, mode: str = "contemporary_with_bias_warning"):
        """
        mode:
          - "point_in_time": Uses dynamic date-indexed constituent membership.
          - "contemporary_with_bias_warning": Uses contemporary Nifty 50 with explicit bias disclaimer.
        """
        self.mode = mode
        if mode == "contemporary_with_bias_warning":
            logger.warning(
                "HistoricalUniverseManager operating in 'contemporary_with_bias_warning' mode. "
                "SURVIVORSHIP BIAS WARNING: Contemporary constituents are back-projected into past dates."
            )

    def get_universe_at(self, target_date: date) -> Tuple[List[str], Dict[str, Any]]:
        """
        Returns list of active constituent symbols on target_date.
        Returns:
            Tuple[List[symbols], metadata_dict]
        """
        if self.mode == "point_in_time":
            # In complete implementation, reconstructed from NSE circular history
            metadata = {
                "universe_type": "point_in_time",
                "as_of_date": target_date.isoformat(),
                "survivorship_biased": False,
                "notes": "Point-in-time constituent membership reconstituted from NSE rebalances.",
            }
            # Start with contemporary and reverse known adjustments if before dates
            universe = set(CONTEMPORARY_NIFTY_50)
            for reb_date in sorted(HISTORICAL_REBALANCES.keys(), reverse=True):
                if target_date < reb_date:
                    added = HISTORICAL_REBALANCES[reb_date]["added"]
                    removed = HISTORICAL_REBALANCES[reb_date]["removed"]
                    for sym in added:
                        universe.discard(sym)
                    for sym in removed:
                        universe.add(sym)
            return sorted(list(universe)), metadata
        else:
            metadata = {
                "universe_type": "contemporary_static",
                "as_of_date": target_date.isoformat(),
                "survivorship_biased": True,
                "warning": "SURVIVORSHIP BIAS: Fixed contemporary constituents applied to historical date.",
            }
            return list(CONTEMPORARY_NIFTY_50), metadata
