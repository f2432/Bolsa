import pandas as pd

class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def run(self, data, signals):
        """
        Executa o backtest de uma estratégia.
        Retorna:
            - equity_curve: pd.Series com evolução do capital
            - trades: pd.DataFrame com detalhes de cada compra/venda
        """
        # Garante alinhamento dos sinais com o histórico
        if not signals.index.equals(data.index):
            signals = signals.reindex(data.index, fill_value=0)

        cash = self.initial_capital
        position = 0
        equity_curve = []
        trades = []

        for i, (date, signal) in enumerate(signals.items()):
            price = data['Close'].loc[date]
            # Compra
            if signal == 1 and position == 0:
                qty = int(cash // price)
                if qty > 0:
                    position = qty
                    cash -= qty * price
                    trades.append({
                        "Data": str(date.date()) if hasattr(date, "date") else str(date),
                        "Tipo": "Compra",
                        "Quantidade": qty,
                        "Preço": round(price, 2),
                        "Resultado (€)": ""
                    })
            # Venda
            elif signal == -1 and position > 0:
                resultado = (price - trades[-1]["Preço"]) * position
                cash += position * price
                trades.append({
                    "Data": str(date.date()) if hasattr(date, "date") else str(date),
                    "Tipo": "Venda",
                    "Quantidade": position,
                    "Preço": round(price, 2),
                    "Resultado (€)": round(resultado, 2)
                })
                position = 0

            equity_curve.append(cash + position * price)

        equity_curve = pd.Series(equity_curve, index=data.index)
        trades_df = pd.DataFrame(trades)
        return {
            "equity_curve": equity_curve,
            "trades": trades_df
        }
