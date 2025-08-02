import pandas as pd

class Backtester:
    """
    Classe para backtesting de estratégias de trading.
    Gera a curva de capital (equity curve) e regista cada trade executada.
    """
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def run(self, data, signals):
        """
        Executa o backtest de uma estratégia.
        Parâmetros:
            data: DataFrame com preços históricos (tem de conter 'Close')
            signals: pd.Series com sinais (1=compra, -1=venda, 0=nada)
        Retorna:
            - equity_curve: pd.Series (evolução do capital)
            - trades: pd.DataFrame (detalhes das compras/vendas)
        """
        if not signals.index.equals(data.index):
            signals = signals.reindex(data.index, fill_value=0)

        cash = self.initial_capital
        position = 0
        equity_curve = []
        trades = []
        entry_price = None

        for date, signal in signals.items():
            price = data['Close'].loc[date]

            # Compra: só se não estamos comprados
            if signal == 1 and position == 0:
                qty = int(cash // price)
                if qty > 0:
                    position = qty
                    cash -= qty * price
                    entry_price = price
                    trades.append({
                        "Data": str(date.date()) if hasattr(date, "date") else str(date),
                        "Tipo": "Compra",
                        "Quantidade": qty,
                        "Preço": round(price, 2),
                        "Resultado (€)": ""
                    })
            # Venda: só se estamos comprados
            elif signal == -1 and position > 0:
                resultado = (price - entry_price) * position
                cash += position * price
                trades.append({
                    "Data": str(date.date()) if hasattr(date, "date") else str(date),
                    "Tipo": "Venda",
                    "Quantidade": position,
                    "Preço": round(price, 2),
                    "Resultado (€)": round(resultado, 2)
                })
                position = 0
                entry_price = None

            # Atualiza equity curve: saldo = cash + valor da posição aberta (se existir)
            equity_curve.append(cash + position * price)

        equity_curve = pd.Series(equity_curve, index=data.index)
        trades_df = pd.DataFrame(trades)
        return {
            "equity_curve": equity_curve,
            "trades": trades_df
        }
