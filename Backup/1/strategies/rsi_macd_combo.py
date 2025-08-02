import pandas as pd
from strategies.base_strategy import BaseStrategy
from indicators.ta import rsi, macd

class RSIMACDStrategy(BaseStrategy):
    """Strategy that uses RSI and MACD combination: 
       Buy when RSI leaves oversold and MACD is bullish, sell when RSI leaves overbought and MACD is bearish."""
    def __init__(self, rsi_buy_threshold=30, rsi_sell_threshold=70):
        name = f"RSI+MACD Strategy"
        super().__init__(name=name)
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold

    def generate_signals(self, data):
        close = data['Close']
        # Compute indicators
        rsi_series = rsi(close, period=14)
        macd_line, signal_line = macd(close)
        signals = pd.Series(0, index=data.index)
        # Buy when RSI crosses above buy_threshold (leaving oversold) and MACD indicates bullish momentum
        buy_signals = (rsi_series.shift(1) < self.rsi_buy_threshold) & (rsi_series >= self.rsi_buy_threshold) \
                      & (macd_line > signal_line)
        # Sell when RSI crosses below sell_threshold (leaving overbought) and MACD indicates bearish momentum
        sell_signals = (rsi_series.shift(1) > self.rsi_sell_threshold) & (rsi_series <= self.rsi_sell_threshold) \
                       & (macd_line < signal_line)
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        return signals
