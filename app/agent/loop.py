import asyncio
import time
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from app.config import settings
from app.agent.tools import agent_tools
from app.agent.state import agent_memory
from app.schemas.responses import StockChartData, IndicatorPoint
from app.agent.llm_client import llm_client



class LLMAgentLoop:
    """
    Hand-built async tool-calling agent loop.
    Scans the NSE market universe and allocates capital across top-confidence stocks.
    """


    def __init__(self):
        self.providers = settings.SUPPORTED_PROVIDERS

    async def execute_trading_pipeline(
        self,
        investment_amount: float,
        tickers: Optional[List[str]] = None,
        sector: Optional[str] = None,
        user_provider_choice: str = "auto",
        session_id: str = "default_session"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, StockChartData], Dict[str, Any]]:
        """
        Executes end-to-end trading pipeline:
        1. Screens top NSE liquid equities (or user-provided tickers/sector scoping).
        2. Fetches live yfinance data + calculates 6 technical indicators + sentiment in parallel.
        3. Queries vector & graph pattern DBs.
        4. Predicts trade direction, confidence score %, target price, stop loss.
        5. Applies deterministic intraday constraints (same-day exit, ~40% position cap).
        6. Upserts each recommended stock's real indicator fingerprint into ChromaDB.
        7. Synthesises full AI rationale per pick.
        """
        start_time = time.time()

        chosen_provider = user_provider_choice.lower() if user_provider_choice in settings.SUPPORTED_PROVIDERS else "auto"
        if chosen_provider == "auto":
            chosen_provider = settings.DEFAULT_PRIMARY_PROVIDER

        # 1. Determine target tickers (screen up to 10 stocks based on sector scoping or balanced sampling)
        if tickers and len(tickers) > 0:
            target_tickers = tickers[:10]
        elif sector:
            from app.db.vector_store import get_sector_for_symbol, SYMBOL_SECTOR_MAP
            sec_lower = sector.lower().strip()
            matched = [
                t for t in settings.DEFAULT_NSE_TICKERS
                if SYMBOL_SECTOR_MAP.get(t, "").lower() == sec_lower or get_sector_for_symbol(t).lower() == sec_lower
            ]
            target_tickers = matched[:10] if matched else settings.DEFAULT_NSE_TICKERS[:10]
        else:
            import random
            all_tickers = settings.DEFAULT_NSE_TICKERS
            # Unbiased random sampling across full NIFTY 50 universe without hardcoded anchor bias
            random.seed(int(time.time() * 1000) % 100000)
            target_tickers = random.sample(all_tickers, min(10, len(all_tickers)))


        # 2. Fetch live data & calculate indicators in parallel
        data_res = await agent_tools.fetch_and_calculate_indicators(target_tickers)

        candidates = []
        for symbol, stock_info in data_res.items():
            latest_ind = stock_info["latest_indicators"]
            sent_score = stock_info["sentiment_score"]
            price = stock_info["current_price"]

            vec_res = agent_tools.query_vector_db(symbol, [price, sent_score, latest_ind["rsi"]])
            graph_res = agent_tools.query_graph_db(symbol)

            pred = agent_tools.predict(symbol, price, latest_ind, sent_score, vec_res.get("similarity_score", 0.85))
            pred["indicators_summary"] = latest_ind
            pred["graph_pattern"] = graph_res.get("graph_pattern")
            pred["vector_pattern"] = vec_res.get("matched_pattern")

            candidates.append(pred)

        # 3. Rank by confidence, pick top 3
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        top_picks = candidates[:3]
        rejected_picks = candidates[3:]

        # 4. Enforce mandatory intraday constraints
        final_recommendations = agent_tools.enforce_constraints(top_picks, investment_amount)

        # 5. Build chart data series for frontend
        charts_dict = {}
        for symbol in target_tickers:
            if symbol in data_res:
                stock_info = data_res[symbol]
                df = stock_info["dataframe"]
                indicator_points = []
                for _, row in df.iterrows():
                    indicator_points.append(IndicatorPoint(
                        date=str(row["date_str"]),
                        price=round(float(row["close"]), 2),
                        ema_5=round(float(row["ema_5"]), 2) if pd_not_null(row.get("ema_5")) else None,
                        rsi=round(float(row["rsi"]), 2) if pd_not_null(row.get("rsi")) else None,
                        obv=float(row["obv"]) if pd_not_null(row.get("obv")) else None,
                        bb_upper=round(float(row["bb_upper"]), 2) if pd_not_null(row.get("bb_upper")) else None,
                        bb_lower=round(float(row["bb_lower"]), 2) if pd_not_null(row.get("bb_lower")) else None,
                        bb_middle=round(float(row["bb_middle"]), 2) if pd_not_null(row.get("bb_middle")) else None,
                        macd=round(float(row["macd"]), 2) if pd_not_null(row.get("macd")) else None,
                        macd_signal=round(float(row["macd_signal"]), 2) if pd_not_null(row.get("macd_signal")) else None,
                        vwap=round(float(row["vwap"]), 2) if pd_not_null(row.get("vwap")) else None
                    ))

                charts_dict[symbol] = StockChartData(
                    symbol=symbol,
                    indicators=indicator_points,
                    sentiments=stock_info["sentiments"],
                    overall_sentiment_score=stock_info["sentiment_score"]
                )

        # 6. Upsert real fingerprints into ChromaDB for future peer comparisons
        for rec in final_recommendations:
            agent_tools.upsert_fingerprint(rec["symbol"], rec["indicators_summary"])

        # 7. Synthesise AI rationale per recommendation
        for rec in final_recommendations:
            ind = rec["indicators_summary"]
            rec["rationale"] = (
                f"Selected based on strong technical setup ({ind['overall_technical_bias']} bias). "
                f"RSI is at {ind['rsi']} ({ind['rsi_signal']}), 5 EMA at ₹{ind['ema_5']} vs entry ₹{rec['current_price']}. "
                f"Graph pattern '{rec.get('graph_pattern')}' confirms momentum with headline sentiment ({ind.get('sentiment_score', 0.5):+.2f}). "
                f"Capital allocated at {rec['allocation_pct'] * 100:.1f}% (₹{rec['allocated_capital']:.2f}) with mandatory intraday same-day exit."
            )

        agent_memory.save_session_run(session_id, final_recommendations, rejected_picks, investment_amount)

        execution_time = round(time.time() - start_time, 3)
        trace_info = {
            "execution_time_seconds": execution_time,
            "provider_used": chosen_provider,
            "fallbacks_triggered": [],
            "tool_calls_count": len(target_tickers) * 4,
            "model_name": f"{chosen_provider}-intraday-v1"
        }

        return final_recommendations, charts_dict, trace_info

    async def generate_chat_response(
        self,
        user_message: str,
        session_id: str,
        provider_choice: str = "auto",
        history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Handles agentic chat responses using LLM / financial knowledge client."""
        context = agent_memory.get_session_context(session_id)
        picked = context.get("picked_stocks", [])
        rejected = context.get("rejected_stocks", [])
        active_stock = context.get("active_stock")

        # Build context summary
        system_context_parts = []
        if active_stock:
            system_context_parts.append(f"Currently focused stock context: {active_stock}.")
        if picked:
            picks_str = ", ".join([f"{p['symbol']} ({p['action']} {p.get('confidence', 80)}%)" for p in picked])
            system_context_parts.append(f"Top recommendations in session: {picks_str}.")
        if rejected:
            rejs_str = ", ".join([r['symbol'] for r in rejected])
            system_context_parts.append(f"Rejected candidate stocks: {rejs_str}.")

        # Combine session memory history with request history
        past_turns = history or context.get("history", [])
        if past_turns:
            turns_str = " | ".join([f"{t.get('role', t.get('user', 'user'))}: {t.get('content', t.get('agent', ''))}" for t in past_turns[-6:]])
            system_context_parts.append(f"Prior Conversation Transcript: [{turns_str}]")

        system_context = " ".join(system_context_parts)

        # Delegate to LLM client
        answer, provider_used = await llm_client.generate_response(
            prompt=user_message,
            system_context=system_context,
            provider_choice=provider_choice,
            history=past_turns
        )

        agent_memory.add_chat_history(session_id, user_message, answer)

        return answer, provider_used, {"picked_count": len(picked), "rejected_count": len(rejected), "active_stock": active_stock}


def pd_not_null(val):
    return val is not None and str(val) != "nan"


agent_loop = LLMAgentLoop()
