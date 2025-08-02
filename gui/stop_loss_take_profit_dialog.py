"""
StopLossTakeProfitDialog
------------------------

Diálogo para optimizar parâmetros de stop-loss e take-profit (profit-taking)
para uma estratégia de trading. Realiza um conjunto de backtests
variando percentuais de stop-loss e take-profit, calcula o Sharpe ratio
e exibe um mapa de calor com os resultados.

Uso típico:
    dlg = StopLossTakeProfitDialog(data, signals, initial_capital)
    dlg.exec_()

Onde `data` é um DataFrame com coluna 'Close', `signals` é uma série de
sinais (1=compra, -1=venda, 0=neutro) gerada por uma estratégia, e
`initial_capital` é o capital inicial do backtest.
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from backtest.metrics import calculate_metrics


def simulate_with_sl_tp(data: pd.DataFrame, signals: pd.Series, stop_loss: float, take_profit: float, initial_capital: float = 10000):
    """
    Simula uma estratégia com stop-loss e take-profit percentuais
    relativamente ao preço de entrada.

    Parâmetros:
        data: DataFrame com coluna 'Close'
        signals: Série com sinais (1=compra, -1=venda, 0=nada)
        stop_loss: percentagem negativa, ex: -0.02 para -2%
        take_profit: percentagem positiva, ex: 0.03 para +3%
        initial_capital: capital inicial

    Retorna:
        equity_curve: série de capital ao longo do tempo
        trades: DataFrame de trades registadas
    """
    # Assegura alinhamento de índice
    if not signals.index.equals(data.index):
        signals = signals.reindex(data.index, fill_value=0)
    cash = initial_capital
    position = 0
    equity_curve = []
    trades = []
    entry_price = None
    for date, sig in signals.items():
        price = data['Close'].loc[date]
        if sig == 1 and position == 0:
            qty = int(cash // price)
            if qty > 0:
                position = qty
                cash -= qty * price
                entry_price = price
                trades.append({"Data": str(date.date()) if hasattr(date, 'date') else str(date),
                               "Tipo": "Compra", "Quantidade": qty, "Preço": round(price, 2), "Resultado (€)": ""})
        # Se existe posição aberta, verifica stop-loss / take-profit
        if position > 0 and entry_price is not None:
            ret = (price - entry_price) / entry_price
            exit_reason = None
            if ret <= stop_loss:
                exit_reason = 'stop'
            elif ret >= take_profit:
                exit_reason = 'take'
            # Ou se sinal de venda
            if sig == -1:
                exit_reason = 'signal'
            if exit_reason:
                resultado = (price - entry_price) * position
                cash += position * price
                trades.append({"Data": str(date.date()) if hasattr(date, 'date') else str(date),
                               "Tipo": "Venda", "Quantidade": position, "Preço": round(price, 2),
                               "Resultado (€)": round(resultado, 2)})
                position = 0
                entry_price = None
        # Actualiza curva de capital
        equity_curve.append(cash + position * price)
    equity_curve = pd.Series(equity_curve, index=data.index)
    trades_df = pd.DataFrame(trades)
    return equity_curve, trades_df


class StopLossTakeProfitDialog(QDialog):
    def __init__(self, data: pd.DataFrame, signals: pd.Series, initial_capital=10000, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mapa de Otimização Stop-Loss / Take-Profit")
        layout = QVBoxLayout(self)
        if data is None or data.empty or 'Close' not in data.columns or signals is None:
            layout.addWidget(QLabel("Dados insuficientes para otimização."))
            return
        # Define grelha de percentagens (negativas para stop-loss, positivas para take-profit)
        stop_vals = [-0.02, -0.03, -0.05, -0.1]
        take_vals = [0.02, 0.04, 0.06, 0.1]
        heatmap = np.zeros((len(stop_vals), len(take_vals)))
        # Calcula métrica (Sharpe) para cada par
        for i, sl in enumerate(stop_vals):
            for j, tp in enumerate(take_vals):
                try:
                    equity_curve, trades = simulate_with_sl_tp(data, signals, sl, tp, initial_capital)
                    metrics = calculate_metrics(data, signals, equity_curve, trades)
                    sharpe_str = metrics.get('sharpe', '0')
                    try:
                        sharpe_val = float(sharpe_str)
                    except Exception:
                        sharpe_val = 0.0
                    heatmap[i, j] = sharpe_val
                except Exception:
                    heatmap[i, j] = np.nan
        # Desenha heatmap
        fig = Figure(figsize=(7, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        im = ax.imshow(heatmap, cmap='viridis', aspect='auto', origin='lower')
        ax.set_xticks(range(len(take_vals)))
        ax.set_yticks(range(len(stop_vals)))
        # Etiquetas formatadas em percentagem
        ax.set_xticklabels([f"{v*100:.0f}%" for v in take_vals])
        ax.set_yticklabels([f"{v*100:.0f}%" for v in stop_vals])
        ax.set_xlabel('Take-Profit (%)')
        ax.set_ylabel('Stop-Loss (%)')
        ax.set_title('Mapa de Sharpe Ratio (Stop-Loss vs Take-Profit)')
        # Anota valores
        vmax = np.nanmax(heatmap) if not np.isnan(heatmap).all() else 0
        for i in range(len(stop_vals)):
            for j in range(len(take_vals)):
                val = heatmap[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                            color='white' if val < vmax/2 else 'black', fontsize=8)
        fig.colorbar(im, ax=ax, label='Sharpe Ratio')
        layout.addWidget(canvas)