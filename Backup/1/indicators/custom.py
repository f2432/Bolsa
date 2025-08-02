import pandas as pd
import numpy as np

def atr(high, low, close, period=14):
    """Average True Range (ATR). Uses simple moving average of True Range for period."""
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # Use Wilder's smoothing for ATR (exponential moving average)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    return atr

def momentum(series, window=10):
    """Momentum indicator: difference between current price and price 'window' periods ago."""
    return series - series.shift(window)
