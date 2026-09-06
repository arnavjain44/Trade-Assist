from fastapi import APIRouter, HTTPException
from app.core.data_fetcher import data_fetcher
from app.core.indicators import indicators_calculator
from app.core.sentiment import sentiment_analyzer
from app.schemas.responses import StockChartData, IndicatorPoint, SentimentItem

router = APIRouter()

@router.get("/indicators/{symbol}", response_model=StockChartData)
async def get_stock_indicators(symbol: str):
    """
    Returns time series data for all 6 technical indicators (5 EMA, RSI, OBV, Bollinger Bands, MACD, VWAP)
    + news sentiment timeline for a specific NSE/BSE stock symbol.
    """
    clean_symbol = symbol.strip().upper()
    if not (clean_symbol.endswith('.NS') or clean_symbol.endswith('.BO')):
        clean_symbol = f"{clean_symbol}.NS"

    try:
        data_dict = await data_fetcher.fetch_stock_data_async([clean_symbol])
        df = data_dict[clean_symbol]
        df_ind = indicators_calculator.calculate_all(df)
        sentiments = sentiment_analyzer.analyze_headlines(clean_symbol)
        agg_sent = sentiment_analyzer.calculate_aggregated_sentiment(sentiments)

        indicator_points = []
        for _, row in df_ind.iterrows():
            indicator_points.append(IndicatorPoint(
                date=str(row['date_str']),
                price=round(float(row['close']), 2),
                ema_5=round(float(row['ema_5']), 2) if row.get('ema_5') is not None else None,
                rsi=round(float(row['rsi']), 2) if row.get('rsi') is not None else None,
                obv=float(row['obv']) if row.get('obv') is not None else None,
                bb_upper=round(float(row['bb_upper']), 2) if row.get('bb_upper') is not None else None,
                bb_lower=round(float(row['bb_lower']), 2) if row.get('bb_lower') is not None else None,
                bb_middle=round(float(row['bb_middle']), 2) if row.get('bb_middle') is not None else None,
                macd=round(float(row['macd']), 2) if row.get('macd') is not None else None,
                macd_signal=round(float(row['macd_signal']), 2) if row.get('macd_signal') is not None else None,
                vwap=round(float(row['vwap']), 2) if row.get('vwap') is not None else None
            ))

        return StockChartData(
            symbol=clean_symbol,
            indicators=indicator_points,
            sentiments=sentiments,
            overall_sentiment_score=agg_sent
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating indicators for {symbol}: {str(e)}")
