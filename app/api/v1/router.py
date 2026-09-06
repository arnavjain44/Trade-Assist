from fastapi import APIRouter
from app.api.v1.endpoints import recommendations, indicators, chat, backtest, trace, peers

api_router = APIRouter()

api_router.include_router(recommendations.router, tags=["Recommendations"])
api_router.include_router(indicators.router, tags=["Indicators & Charts"])
api_router.include_router(chat.router, tags=["Agent Chat"])
api_router.include_router(backtest.router, tags=["Backtesting"])
api_router.include_router(trace.router, tags=["Observability & Telemetry"])
api_router.include_router(peers.router, tags=["Sector Peer Comparison"])

