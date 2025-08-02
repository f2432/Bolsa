import pandas as pd
import numpy as np

def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-8)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def bollinger_bands(series, window=20, num_std=2):
    sma_ = sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper = sma_ + num_std * std
    lower = sma_ - num_std * std
    return sma_, upper, lower

def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_ = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).sum() / atr_)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).sum() / atr_)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)) * 100
    return dx.rolling(window=period, min_periods=1).mean()

def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period, min_periods=1).mean()
    mad = tp.rolling(window=period, min_periods=1).apply(lambda x: np.fabs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def stochastic_k(close, low, high, k_period=14):
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    return 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-8))

def obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def mfi(close, high, low, volume, period=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    pos_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(period).sum()
    neg_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(period).sum()
    mfi = 100 - (100 / (1 + (pos_flow / (neg_flow + 1e-8))))
    return mfi

def average_volume(volume, window):
    return volume.rolling(window=window, min_periods=1).mean()

def is_bullish_engulfing(open_, close):
    return ((close.shift(1) < open_.shift(1)) & (close > open_) & (close > open_.shift(1)) & (open_ < close.shift(1))).astype(int)

def is_bearish_engulfing(open_, close):
    return ((close.shift(1) > open_.shift(1)) & (close < open_) & (close < open_.shift(1)) & (open_ > close.shift(1))).astype(int)

def compute_all_indicators(data):
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    open_ = data['Open']
    features = {
        "sma10": sma(close, 10),
        "sma20": sma(close, 20),
        "sma50": sma(close, 50),
        "sma200": sma(close, 200),
        "ema10": ema(close, 10),
        "ema20": ema(close, 20),
        "ema50": ema(close, 50),
        "ema200": ema(close, 200),
        "rsi14": rsi(close, 14),
        "macd": macd(close)[0],
        "macd_signal": macd(close)[1],
        "bb_upper": bollinger_bands(close)[1],
        "bb_lower": bollinger_bands(close)[2],
        "adx": adx(high, low, close, 14),
        "cci": cci(high, low, close, 20),
        "atr": atr(high, low, close, 14),
        "stoch_k": stochastic_k(close, low, high, 14),
        "obv": obv(close, volume),
        "mfi": mfi(close, high, low, volume, 14),
        "avg_vol20": average_volume(volume, 20),
        "avg_vol50": average_volume(volume, 50),
        "avg_vol200": average_volume(volume, 200),
        "bullish_engulfing": is_bullish_engulfing(open_, close),
        "bearish_engulfing": is_bearish_engulfing(open_, close),
    }
    return features

