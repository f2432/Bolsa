import pandas as pd
from strategies.base_strategy import BaseStrategy
from indicators.ta import rsi, macd

class RSIMACDStrategy(BaseStrategy):
    """
    Estratégia que combina sinais do RSI e MACD.
    Compra quando RSI cruza para cima do limiar de sobrevenda e MACD está acima do sinal.
    Vende quando RSI cruza para baixo do limiar de sobrecompra e MACD está abaixo do sinal.
    """
    def __init__(self, rsi_buy_threshold=30, rsi_sell_threshold=70):
        super().__init__(name=f"RSI+MACD ({rsi_buy_threshold}/{rsi_sell_threshold})")
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold

    def generate_signals(self, data):
        if data is None or data.empty:
            raise ValueError("Dados históricos vazios ou inválidos.")
        if 'Close' not in data.columns:
            raise KeyError("O DataFrame precisa de uma coluna 'Close'.")

        close = data['Close']
        rsi_series = rsi(close, period=14)
        macd_line, signal_line = macd(close)
        signals = pd.Series(0, index=data.index)

        # Compra: RSI cruza limiar de baixo para cima e MACD positivo
        buy = (
            (rsi_series.shift(1) < self.rsi_buy_threshold) &
            (rsi_series >= self.rsi_buy_threshold) &
            (macd_line > signal_line)
        )

        # Venda: RSI cruza limiar de cima para baixo e MACD negativo
        sell = (
            (rsi_series.shift(1) > self.rsi_sell_threshold) &
            (rsi_series <= self.rsi_sell_threshold) &
            (macd_line < signal_line)
        )

        signals[buy] = 1
        signals[sell] = -1
        return signals.reindex(close.index, fill_value=0)
