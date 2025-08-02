import pandas as pd

def sma(series, window):
    """Simple Moving Average."""
    return series.rolling(window=window).mean()

def ema(series, window):
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()

def rsi(series, period=14):
    """Relative Strength Index (RSI)."""
    # Calculate price differences
    delta = series.diff(1)
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Exponential moving average of gains and losses (Wilder's smoothing)
    avg_gain = up.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence (MACD). Returns MACD line and signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def bollinger_bands(series, window=20, num_std=2):
    """Bollinger Bands. Returns a DataFrame with columns: 'middle', 'upper', 'lower'."""
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bands = pd.DataFrame({'middle': mid, 'upper': upper, 'lower': lower})
    return bands
