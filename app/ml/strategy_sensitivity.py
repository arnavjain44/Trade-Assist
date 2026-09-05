import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from app.config import settings
from app.ml.labeling import HistoricalTradeLabeler

logger = logging.getLogger(__name__)


class StrategySensitivityAnalyzer:
    """
    Strategy / Label Sensitivity Analysis Pipeline (Phase 2.5).

    Evaluates historical trade outcomes and trading economics across varying
    TARGET_PCT, STOP_LOSS_PCT, and MAX_HOLD_MINUTES configurations.

    Crucial Principles:
    - Current Phase 2 configuration (2.2% target, 0.9% stop, 240m hold) is the BASELINE.
    - Does NOT modify, overwrite, or delete data/processed/labeled_dataset.parquet.
    - Exact Phase 2 labeling methodology is preserved (entry candle excluded, same-day IST restricted, same-candle ambiguity handled).
    - Evaluates before-cost economics and a clearly separated hypothetical transaction cost model (e.g. 0.05% / 5 bps round-trip STT/brokerage/slippage).
    """

    TARGET_STOP_COMBINATIONS = [
        {"name": "A", "target_pct": 0.010, "stop_loss_pct": 0.005},
        {"name": "B", "target_pct": 0.010, "stop_loss_pct": 0.009},
        {"name": "C", "target_pct": 0.015, "stop_loss_pct": 0.009},
        {"name": "D", "target_pct": 0.020, "stop_loss_pct": 0.009},
        {"name": "E", "target_pct": 0.022, "stop_loss_pct": 0.009},  # BASELINE
        {"name": "F", "target_pct": 0.025, "stop_loss_pct": 0.009},
        {"name": "G", "target_pct": 0.030, "stop_loss_pct": 0.009},
    ]

    HOLD_PERIODS = [60, 120, 240, 375]  # 375m = remaining same-day session

    def __init__(self, cost_pct: float = 0.0005):
        """
        cost_pct: Hypothetical round-trip transaction cost (default 0.05% / 5 bps).
        """
        self.cost_pct = cost_pct

    def run_sensitivity_analysis(
        self,
        df: pd.DataFrame,
        output_json_path: str = "data/processed/strategy_sensitivity_results.json",
        output_csv_path: str = "data/processed/strategy_sensitivity_results.csv"
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Runs sensitivity analysis across all target/stop/horizon combinations.

        Returns (results_list, results_df).
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        logger.info("Starting Strategy Sensitivity Analysis on %d rows...", len(df))
        results = []

        # Evaluate target/stop variations at baseline hold (240m) + holding period variations at baseline target/stop (2.2%/0.9%)
        configs_to_run = []

        # 1. All Target/Stop combinations at 240m hold
        for combo in self.TARGET_STOP_COMBINATIONS:
            configs_to_run.append({
                "combo_name": combo["name"],
                "target_pct": combo["target_pct"],
                "stop_loss_pct": combo["stop_loss_pct"],
                "max_hold_minutes": 240,
                "is_baseline": (combo["name"] == "E")
            })

        # 2. Hold period variations for Baseline (2.2% / 0.9%)
        for hold_min in [60, 120, 375]:
            configs_to_run.append({
                "combo_name": f"E_{hold_min}m",
                "target_pct": 0.022,
                "stop_loss_pct": 0.009,
                "max_hold_minutes": hold_min,
                "is_baseline": False
            })

        # 3. Hold period variations for 1.0% / 0.5% (Combo A) and 1.5% / 0.9% (Combo C)
        for hold_min in [60, 120]:
            configs_to_run.append({
                "combo_name": f"A_{hold_min}m",
                "target_pct": 0.010,
                "stop_loss_pct": 0.005,
                "max_hold_minutes": hold_min,
                "is_baseline": False
            })
            configs_to_run.append({
                "combo_name": f"C_{hold_min}m",
                "target_pct": 0.015,
                "stop_loss_pct": 0.009,
                "max_hold_minutes": hold_min,
                "is_baseline": False
            })

        for cfg in configs_to_run:
            logger.info(
                "Running Sensitivity Config %s (Target: %.1f%%, Stop: %.1f%%, Hold: %dm)...",
                cfg["combo_name"], cfg["target_pct"] * 100.0, cfg["stop_loss_pct"] * 100.0, cfg["max_hold_minutes"]
            )
            labeler = HistoricalTradeLabeler(
                target_pct=cfg["target_pct"],
                stop_loss_pct=cfg["stop_loss_pct"],
                max_hold_minutes=cfg["max_hold_minutes"]
            )
            
            # Label single symbol or combined dataset
            symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else ["UNKNOWN"]
            sub_dfs = []
            for sym in symbols:
                sym_df = df[df["symbol"] == sym].copy() if "symbol" in df.columns else df.copy()
                sub_dfs.append(labeler._label_single_symbol(sym_df))
            
            labeled_cfg = pd.concat(sub_dfs, ignore_index=True)
            stats = self._calculate_config_metrics(labeled_cfg, cfg)
            results.append(stats)

        results_df = pd.DataFrame(results)

        # Export JSON and CSV
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(os.path.abspath(output_json_path), "w", encoding="utf-8") as f:
            json.dump({
                "analysis_status": "SUCCESS",
                "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                "hypothetical_round_trip_cost_pct": self.cost_pct,
                "total_configurations_evaluated": len(results),
                "results": results
            }, f, indent=2)

        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        results_df.to_csv(os.path.abspath(output_csv_path), index=False)
        logger.info("Sensitivity analysis completed. Exported to %s and %s", output_json_path, output_csv_path)

        return results, results_df

    def _calculate_config_metrics(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates comprehensive trading economics & dataset statistics for a configuration."""
        total_candidates = len(df)
        long_df = df[df["direction"] == 1]
        short_df = df[df["direction"] == -1]

        valid_df = df[df["label_status"] == "VALID"]
        valid_count = len(valid_df)

        target_hits = df[df["exit_reason"] == "TARGET"]
        stop_hits = df[df["exit_reason"] == "STOP"]
        timeouts = df[df["exit_reason"] == "TIMEOUT"]
        ambiguous = df[df["label_status"] == "AMBIGUOUS"]
        insufficient = df[df["label_status"] == "INSUFFICIENT_FUTURE_DATA"]

        target_hit_count = len(target_hits)
        stop_hit_count = len(stop_hits)
        timeout_count = len(timeouts)
        ambiguous_count = len(ambiguous)
        insufficient_count = len(insufficient)

        pos_rate = round((target_hit_count / valid_count) * 100.0, 2) if valid_count > 0 else 0.0
        timeout_pct = round((timeout_count / valid_count) * 100.0, 2) if valid_count > 0 else 0.0

        # Realized Returns (Before-Cost)
        all_realized = valid_df["realized_return"].values if not valid_df.empty else np.array([0.0])
        avg_realized = float(np.mean(all_realized)) * 100.0
        median_realized = float(np.median(all_realized)) * 100.0

        avg_target_ret = float(np.mean(target_hits["realized_return"])) * 100.0 if not target_hits.empty else 0.0
        avg_stop_ret = float(np.mean(stop_hits["realized_return"])) * 100.0 if not stop_hits.empty else 0.0
        avg_timeout_ret = float(np.mean(timeouts["realized_return"])) * 100.0 if not timeouts.empty else 0.0

        win_loss_ratio = round(target_hit_count / stop_hit_count, 3) if stop_hit_count > 0 else 0.0

        # Profit Factor: Sum(positive returns) / |Sum(negative returns)|
        pos_returns = all_realized[all_realized > 0]
        neg_returns = all_realized[all_realized < 0]
        sum_pos = float(np.sum(pos_returns)) if len(pos_returns) > 0 else 0.0
        sum_neg = abs(float(np.sum(neg_returns))) if len(neg_returns) > 0 else 0.0
        profit_factor = round(sum_pos / sum_neg, 3) if sum_neg > 0 else 0.0

        expected_ret_per_valid = float(np.sum(all_realized) / valid_count) * 100.0 if valid_count > 0 else 0.0
        expected_ret_per_candidate = float(np.sum(all_realized) / total_candidates) * 100.0 if total_candidates > 0 else 0.0

        # Hypothetical After-Cost Expectations (Net Return = Realized Return - cost_pct)
        net_realized = all_realized - self.cost_pct
        net_avg_realized = float(np.mean(net_realized)) * 100.0
        net_sum_pos = float(np.sum(net_realized[net_realized > 0])) if (net_realized > 0).any() else 0.0
        net_sum_neg = abs(float(np.sum(net_realized[net_realized < 0]))) if (net_realized < 0).any() else 0.0
        net_profit_factor = round(net_sum_pos / net_sum_neg, 3) if net_sum_neg > 0 else 0.0

        # Opportunities per symbol per day
        symbols_count = df["symbol"].nunique() if "symbol" in df.columns else 1
        n_days = len(df["timestamp"].dt.date.unique()) if "timestamp" in df.columns else 1
        opps_per_sym_day = round(total_candidates / (symbols_count * n_days * 2), 2)  # 2 candidates per candle

        # LONG vs SHORT Performance Breakdown
        long_valid = long_df[long_df["label_status"] == "VALID"]
        short_valid = short_df[short_df["label_status"] == "VALID"]

        long_pos_rate = round((len(long_df[long_df["exit_reason"] == "TARGET"]) / len(long_valid)) * 100.0, 2) if not long_valid.empty else 0.0
        short_pos_rate = round((len(short_df[short_df["exit_reason"] == "TARGET"]) / len(short_valid)) * 100.0, 2) if not short_valid.empty else 0.0

        long_avg_ret = float(np.mean(long_valid["realized_return"])) * 100.0 if not long_valid.empty else 0.0
        short_avg_ret = float(np.mean(short_valid["realized_return"])) * 100.0 if not short_valid.empty else 0.0

        return {
            "combo_name": cfg["combo_name"],
            "target_pct": cfg["target_pct"],
            "stop_loss_pct": cfg["stop_loss_pct"],
            "max_hold_minutes": cfg["max_hold_minutes"],
            "is_baseline": cfg["is_baseline"],
            "total_candidates": total_candidates,
            "valid_candidates": valid_count,
            "long_candidates": len(long_df),
            "short_candidates": len(short_df),
            "target_hit_count": target_hit_count,
            "stop_hit_count": stop_hit_count,
            "timeout_count": timeout_count,
            "ambiguous_count": ambiguous_count,
            "insufficient_future_data_count": insufficient_count,
            "positive_rate_pct": pos_rate,
            "timeout_rate_pct": timeout_pct,
            "avg_realized_return_pct": round(avg_realized, 4),
            "median_realized_return_pct": round(median_realized, 4),
            "avg_target_hit_return_pct": round(avg_target_ret, 4),
            "avg_stop_hit_return_pct": round(avg_stop_ret, 4),
            "avg_timeout_return_pct": round(avg_timeout_ret, 4),
            "win_loss_ratio": win_loss_ratio,
            "profit_factor": profit_factor,
            "expected_return_per_valid_pct": round(expected_ret_per_valid, 4),
            "expected_return_per_candidate_pct": round(expected_ret_per_candidate, 4),
            "net_avg_realized_return_pct": round(net_avg_realized, 4),
            "net_profit_factor": net_profit_factor,
            "opps_per_sym_day": opps_per_sym_day,
            "long_positive_rate_pct": long_pos_rate,
            "short_positive_rate_pct": short_pos_rate,
            "long_avg_return_pct": round(long_avg_ret, 4),
            "short_avg_return_pct": round(short_avg_ret, 4),
        }


strategy_analyzer = StrategySensitivityAnalyzer()
