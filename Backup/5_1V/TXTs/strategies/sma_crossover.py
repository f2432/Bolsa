import pandas as pd
from indicators import ta
from strategies.base_strategy import BaseStrategy

class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, short_window=50, long_window=200):
        super().__init__(name=f"SMA Crossover ({short_window}/{long_window})")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        if data is None or data.empty:
            raise ValueError("Dados históricos vazios ou inválidos.")
        if 'Close' not in data.columns:
            raise KeyError("O DataFrame precisa de uma coluna 'Close'.")

        close = data['Close']
        short_sma = ta.sma(close, self.short_window)
        long_sma = ta.sma(close, self.long_window)
        position = (short_sma > long_sma).astype(int)
        signals = position.diff().fillna(0).astype(int)
        return signals.reindex(close.index, fill_value=0)