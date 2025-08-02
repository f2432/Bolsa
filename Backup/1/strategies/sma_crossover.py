import pandas as pd
from strategies.base_strategy import BaseStrategy
from indicators.ta import sma

class SMACrossoverStrategy(BaseStrategy):
    """Strategy that buys when a short-term SMA crosses above a long-term SMA, and sells on the opposite cross."""
    def __init__(self, short_window=50, long_window=200):
        name = f"SMA Crossover ({short_window}/{long_window})"
        super().__init__(name=name)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        # --- CORREÇÃO: Garante que o índice é só datas ---
        if isinstance(data.index, pd.MultiIndex):
            # Se tiver MultiIndex (ex: Ticker, Date), converte para apenas Date
            if 'Date' in data.index.names:
                data = data.reset_index()
                data = data.set_index('Date')
            else:
                data = data.reset_index(drop=True)
        if 'Ticker' in data.columns:
            # Se houver coluna Ticker, elimina (deve ser só um ticker nesta função)
            data = data.drop(columns=['Ticker'])
        # Agora o índice é apenas datas, e as colunas têm 'Close'
        close = data['Close']

        # Cálculo dos SMAs
        sma_short = sma(close, self.short_window)
        sma_long = sma(close, self.long_window)
        signals = pd.Series(0, index=close.index)

        # Buy signal: short SMA crosses above long SMA
        buy_signals = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
        # Sell signal: short SMA crosses below long SMA
        sell_signals = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))
        signals.loc[buy_signals] = 1
        signals.loc[sell_signals] = -1
        return signals
