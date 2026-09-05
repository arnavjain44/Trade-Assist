from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class IndicatorPoint(BaseModel):
    date: str
    price: float
    ema_5: Optional[float] = None
    rsi: Optional[float] = None
    obv: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_middle: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    vwap: Optional[float] = None

class SentimentItem(BaseModel):
    headline: str
    sentiment: str  # POSITIVE, NEGATIVE, NEUTRAL
    score: float   # -1.0 to +1.0
    date: str

class StockChartData(BaseModel):
    symbol: str
    indicators: List[IndicatorPoint]
    sentiments: List[SentimentItem]
    overall_sentiment_score: float

class TradeRecommendation(BaseModel):
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 100.0%
    current_price: float
    target_price: float
    stop_loss: float
    hold_until: str = "same_day"  # Strictly enforced intraday
    forced_override: bool = False
    allocation_pct: float  # e.g. 0.35 (35%)
    allocated_capital: float  # e.g. 1750.00
    shares_to_trade: int
    rationale: str
    indicators_summary: Dict[str, Any]

class TraceInfo(BaseModel):
    execution_time_seconds: float
    provider_used: str
    fallbacks_triggered: List[str]
    tool_calls_count: int
    model_name: str

class RecommendationResponse(BaseModel):
    investment_amount: float
    total_allocated: float
    unallocated_cash: float
    recommendations: List[TradeRecommendation]
    charts: Dict[str, StockChartData]
    trace: TraceInfo

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    provider_used: str
    reasoning_context: Dict[str, Any]

class BacktestResult(BaseModel):
    overall_accuracy: float
    logistic_regression_accuracy: float
    random_forest_accuracy: float
    xgboost_accuracy: float
    total_trades_analyzed: int
    recommended_model: str
