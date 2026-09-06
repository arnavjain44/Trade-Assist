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

        # Evaluate Phase 5 model for latest candle
        latest_ind = indicators_calculator.extract_latest_summary(df_ind)
        decision_dt = latest_ind.get("timestamp")
        query_date_str = str(df_ind["timestamp"].iloc[-1].strftime("%Y-%m-%d")) if len(df_ind) > 0 else None

        from app.db.vector_store import vector_store
        from app.ml.pipeline import ml_engine

        news_feats = sentiment_analyzer.get_phase5_news_features(clean_symbol, decision_timestamp_ist=decision_dt)
        context_feats = vector_store.query_phase5_context_similarities(clean_symbol, latest_ind, query_date_str) if query_date_str else {"market_similarity": 0.0, "stock_similarity": 0.0}

        pred = ml_engine.predict_trade_signal(
            clean_symbol,
            latest_ind["close_price"],
            latest_ind,
            news_feats,
            context_feats,
            timeframe=latest_ind.get("timeframe", "5m"),
        )
        pred["news_features"] = news_feats
        pred["context_features"] = context_feats
        pred["latest_indicators"] = latest_ind

        return StockChartData(
            symbol=clean_symbol,
            indicators=indicator_points,
            sentiments=sentiments,
            overall_sentiment_score=agg_sent,
            latest_prediction=pred,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating indicators for {symbol}: {str(e)}")
