import pandas as pd
import numpy as np
from app.core.indicators import indicators_calculator

def test_technical_indicators_calculation():
    # Create sample DataFrame
    dates = pd.date_range(end=pd.Timestamp.now(), periods=25, freq='D')
    prices = [100 + i + (i % 3) for i in range(25)]
    df = pd.DataFrame({
        'date_str': dates.strftime('%Y-%m-%d'),
        'open': prices,
        'high': [p + 2 for p in prices],
        'low': [p - 2 for p in prices],
        'close': prices,
        'volume': [100000 + i * 1000 for i in range(25)]
    })

    result_df = indicators_calculator.calculate_all(df)

    assert 'ema_5' in result_df.columns
    assert 'rsi' in result_df.columns
    assert 'obv' in result_df.columns
    assert 'bb_upper' in result_df.columns
    assert 'bb_lower' in result_df.columns
    assert 'macd' in result_df.columns
    assert 'vwap' in result_df.columns

    summary = indicators_calculator.extract_latest_summary(result_df)
    assert 'close_price' in summary
    assert 'rsi_signal' in summary
    assert 'overall_technical_bias' in summary
