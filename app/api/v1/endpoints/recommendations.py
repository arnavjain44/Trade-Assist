import asyncio
from fastapi import APIRouter, HTTPException
from app.schemas.requests import InvestmentRequest
from app.schemas.responses import RecommendationResponse, TraceInfo
from app.agent.loop import agent_loop

router = APIRouter()

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_trading_recommendations(request: InvestmentRequest):
    """
    Main Entry Point: Accepts investment amount, natural language agent prompt (preferred sectors/domains),
    optional tickers, and LLM provider choice.
    Returns trade picks, capital splits, 6 indicator charts data, and AI rationale.
    Hard 15-second timeout ensures the endpoint never hangs indefinitely.
    """
    try:
        recommendations, charts, trace_data = await asyncio.wait_for(
            agent_loop.execute_trading_pipeline(
                investment_amount=request.investment_amount,
                tickers=request.tickers,
                sector=request.sector,
                user_provider_choice=request.provider or "auto"
            ),
            timeout=15.0
        )

        total_allocated = sum(r["allocated_capital"] for r in recommendations)
        unallocated_cash = round(max(0.0, request.investment_amount - total_allocated), 2)

        return RecommendationResponse(
            investment_amount=request.investment_amount,
            total_allocated=round(total_allocated, 2),
            unallocated_cash=unallocated_cash,
            recommendations=recommendations,
            charts=charts,
            trace=TraceInfo(**trace_data)
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Market scan timed out (>15s). Please retry — using cached fallback data."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")
