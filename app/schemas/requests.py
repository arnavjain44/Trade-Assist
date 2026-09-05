from pydantic import BaseModel, Field
from typing import List, Optional


class InvestmentRequest(BaseModel):
    investment_amount: float = Field(..., gt=0, description="Total capital to invest (e.g. Rs. 5000)")
    tickers: Optional[List[str]] = Field(
        default=None,
        description="Optional list of explicit NSE stock symbols to screen"
    )
    sector: Optional[str] = Field(
        default=None,
        description="Optional sector string to scope recommendations, e.g. 'banking' or 'it'"
    )
    provider: Optional[str] = Field(
        default="auto",
        description="Preferred LLM provider choice: 'auto', 'gemini', 'groq', or 'openrouter'"
    )
    max_stock_count: Optional[int] = Field(default=5, ge=1, le=10)
    risk_tolerance: Optional[str] = Field(default="medium")


class PeersRequest(BaseModel):
    symbol: str = Field(..., description="NSE ticker to find sector peers for, e.g. 'HDFCBANK.NS'")
    max_peers: int = Field(default=5, ge=1, le=10, description="Max number of sector peers to return")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Follow-up question for the trading agent")
    session_id: Optional[str] = Field(default="default_session")
    provider: Optional[str] = Field(default="auto")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Prior conversation message turns")



class BacktestRequest(BaseModel):
    tickers: Optional[List[str]] = None
    days: int = Field(default=30, ge=5, le=365)
