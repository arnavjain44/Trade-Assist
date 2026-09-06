from fastapi import APIRouter
from app.config import settings

router = APIRouter()

@router.get("/trace")
async def get_system_trace():
    """Returns AI operational telemetry (token usage, latency, provider status)."""
    return {
        "status": "online",
        "primary_provider": settings.DEFAULT_PRIMARY_PROVIDER,
        "supported_providers": settings.SUPPORTED_PROVIDERS,
        "intraday_constraint_rule": "Enforced (Same-day exit, ~40% cap)",
        "stock_universe": "NSE / BSE Indian Equities"
    }
