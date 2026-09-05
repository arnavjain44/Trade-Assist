"""
Phase 5 Evaluation & Hierarchical Selection Module

Provides:
1. Hierarchical economic selection logic:
   - Primary: Validation Net Average Return (after 0.05% friction)
   - Secondary: Validation Net Profit Factor
   - Tertiary: Validation PR-AUC
   - Guard: Minimum 30 selected trades on Validation set
2. Detailed economic breakdown for Out-of-Sample Test set:
   - Overall statistical and economic metrics
   - Segmented LONG vs SHORT performance
   - Day-by-day trade distribution and return stability
   - Average holding period and max winning/losing trades
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.ml.evaluation import ChronologicalEvaluator

logger = logging.getLogger(__name__)


class Phase5Evaluator(ChronologicalEvaluator):
    """Extended Chronological Evaluator tailored for Phase 5 real-context experiments."""

    @staticmethod
    def select_best_candidate(candidates: List[Dict[str, Any]], min_trades: int = 30) -> Dict[str, Any]:
        """
        Hierarchically selects the winning configuration based strictly on Validation metrics.

        Hierarchy:
        1. Eligibility: selected_trade_count >= min_trades (30)
        2. Primary: Validation Net Average Return (%)
        3. Secondary: Validation Net Profit Factor
        4. Tertiary: Validation PR-AUC
        """
        if not candidates:
            raise ValueError("No candidates provided for selection.")

        # Filter candidates meeting the trade count guard
        eligible = [
            c for c in candidates
            if c.get("val_econ", {}).get("selected_trade_count", 0) >= min_trades
        ]

        pool = eligible if eligible else candidates

        def sort_key(item: Dict[str, Any]):
            econ = item.get("val_econ", {})
            stat = item.get("val_stat", {})
            net_avg = econ.get("net_avg_return_pct", -999.0)
            pf = econ.get("net_profit_factor", 0.0)
            # Cap extreme profit factors for stable sorting
            pf_capped = min(pf, 100.0)
            pr_auc = stat.get("pr_auc", 0.0)
            trades = econ.get("selected_trade_count", 0)
            return (net_avg, pf_capped, pr_auc, trades)

        best = max(pool, key=sort_key)
        return best

    def evaluate_test_detailed(
        self,
        df_test: pd.DataFrame,
        probabilities: np.ndarray,
        threshold: float,
    ) -> Dict[str, Any]:
        """
        Computes detailed statistical and economic performance on the locked Test set.
        Includes LONG vs SHORT breakdown, day-by-day stability, holding periods, and trade extremes.
        """
        # Base economic backtesting
        econ_base = self.backtest_economic_performance(df_test, probabilities, threshold=threshold)
        stat_base = self.calculate_statistical_metrics(df_test["label"].values, probabilities, threshold=threshold)

        # Filter executed trades
        mask = (probabilities >= threshold) & (df_test["label_status"] == "VALID")
        df_trades = df_test[mask].copy().sort_values("timestamp").reset_index(drop=True)

        if df_trades.empty:
            return {
                "statistical": stat_base,
                "economic": econ_base,
                "long_breakdown": {},
                "short_breakdown": {},
                "daily_breakdown": [],
                "holding_period_stats": {},
                "extremes": {},
            }

        # Apply single-position overlap constraint (same as backtest_economic_performance)
        active_trades = []
        symbol_exit_times: Dict[str, Any] = {}

        for idx, row in df_trades.iterrows():
            sym = row["symbol"]
            entry_time = pd.to_datetime(row["timestamp"])
            exit_time = pd.to_datetime(row["exit_timestamp"]) if pd.notnull(row["exit_timestamp"]) else None

            if sym in symbol_exit_times and symbol_exit_times[sym] is not None:
                active_exit = symbol_exit_times[sym]
                if active_exit.tzinfo is None and entry_time.tzinfo is not None:
                    active_exit = active_exit.tz_localize(entry_time.tzinfo)
                elif active_exit.tzinfo is not None and entry_time.tzinfo is None:
                    entry_time = entry_time.tz_localize(active_exit.tzinfo)

                if entry_time < active_exit:
                    continue

            active_trades.append(row)
            if exit_time is not None:
                symbol_exit_times[sym] = exit_time

        df_exec = pd.DataFrame(active_trades) if active_trades else pd.DataFrame()
        if df_exec.empty:
            return {
                "statistical": stat_base,
                "economic": econ_base,
                "long_breakdown": {},
                "short_breakdown": {},
                "daily_breakdown": [],
                "holding_period_stats": {},
                "extremes": {},
            }

        # Net returns
        net_ret = df_exec["realized_return"].values - self.cost_pct

        # 1. LONG vs SHORT Breakdown
        long_sub = df_exec[df_exec["direction"] == 1]
        short_sub = df_exec[df_exec["direction"] == -1]

        def _sub_stats(sub: pd.DataFrame) -> Dict[str, Any]:
            if sub.empty:
                return {"trades": 0, "win_rate": 0.0, "net_avg_return_pct": 0.0, "net_profit_factor": 0.0}
            sub_net = sub["realized_return"].values - self.cost_pct
            wins = np.sum(sub_net > 0)
            gains = np.sum(sub_net[sub_net > 0])
            losses = np.abs(np.sum(sub_net[sub_net < 0]))
            pf = float(gains / losses) if losses > 0 else (999.0 if gains > 0 else 0.0)
            return {
                "trades": len(sub),
                "win_rate": round(float(wins / len(sub)) * 100.0, 2),
                "net_avg_return_pct": round(float(np.mean(sub_net)) * 100.0, 4),
                "net_profit_factor": round(pf, 4),
            }

        long_stats = _sub_stats(long_sub)
        short_stats = _sub_stats(short_sub)

        # 2. Day-by-Day Stability
        df_exec["trade_date"] = pd.to_datetime(df_exec["timestamp"]).dt.strftime("%Y-%m-%d")
        daily_breakdown = []
        for date_str, group in df_exec.groupby("trade_date"):
            g_net = group["realized_return"].values - self.cost_pct
            wins = int(np.sum(g_net > 0))
            daily_breakdown.append({
                "date": date_str,
                "trades": len(group),
                "wins": wins,
                "win_rate": round((wins / len(group)) * 100.0, 1),
                "net_avg_return_pct": round(float(np.mean(g_net)) * 100.0, 4),
                "net_total_return_pct": round(float(np.sum(g_net)) * 100.0, 4),
            })

        # 3. Holding Period & Extremes
        hold_periods = df_exec["holding_period_minutes"].values
        extremes = {
            "largest_winning_trade_pct": round(float(np.max(net_ret)) * 100.0, 4) if len(net_ret) > 0 else 0.0,
            "largest_losing_trade_pct": round(float(np.min(net_ret)) * 100.0, 4) if len(net_ret) > 0 else 0.0,
            "mean_holding_period_minutes": round(float(np.mean(hold_periods)), 2) if len(hold_periods) > 0 else 0.0,
            "median_holding_period_minutes": round(float(np.median(hold_periods)), 2) if len(hold_periods) > 0 else 0.0,
        }

        return {
            "statistical": stat_base,
            "economic": econ_base,
            "long_breakdown": long_stats,
            "short_breakdown": short_stats,
            "daily_breakdown": daily_breakdown,
            "holding_period_stats": {
                "mean_minutes": extremes["mean_holding_period_minutes"],
                "median_minutes": extremes["median_holding_period_minutes"],
            },
            "extremes": extremes,
        }
