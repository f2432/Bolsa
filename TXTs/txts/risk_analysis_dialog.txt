"""
RiskAnalysisDialog
------------------

Este diálogo apresenta uma análise de risco da curva de saldo de um backtest.
Inclui a visualização do drawdown ao longo do tempo, a distribuição de
drawdowns e estatísticas relevantes como o máximo drawdown e o tempo
máximo em território negativo (time under water).

Uso:
    dlg = RiskAnalysisDialog(equity_curve)
    dlg.exec_()

Requer matplotlib para os gráficos.
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np


class RiskAnalysisDialog(QDialog):
    def __init__(self, equity_curve: pd.Series, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Análise de Risco do Backtest")
        layout = QVBoxLayout(self)
        # Verifica se a curva é válida
        if equity_curve is None or len(equity_curve) == 0:
            layout.addWidget(QLabel("Curva de saldo vazia."))
            return
        # Garante que é uma Series com índice temporal
        if not isinstance(equity_curve, pd.Series):
            equity_curve = pd.Series(equity_curve)
        # Calcula drawdown: (saldo actual / máximo acumulado) - 1
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1
        drawdown = drawdown.fillna(0)
        # Calcula tempo sob água (dias consecutivos de drawdown < 0)
        underwater = drawdown < 0
        durations = []
        count = 0
        for flag in underwater:
            if flag:
                count += 1
            else:
                if count > 0:
                    durations.append(count)
                count = 0
        if count > 0:
            durations.append(count)
        max_duration = max(durations) if durations else 0
        max_dd = drawdown.min()  # valor negativo máximo
        avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        # Cria figura com subplots
        fig = Figure(figsize=(8, 5))
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(211)
        ax1.plot(drawdown.index, drawdown.values, color='blue')
        ax1.set_title("Evolução do Drawdown")
        ax1.set_ylabel("Drawdown")
        ax1.grid(True, linestyle='--', alpha=0.5)
        # Histograma de drawdowns negativos
        ax2 = fig.add_subplot(212)
        negative_dd = drawdown[drawdown < 0]
        if len(negative_dd) > 0:
            ax2.hist(negative_dd.values, bins=30, color='orange', edgecolor='black')
            ax2.set_title("Distribuição dos Drawdowns")
            ax2.set_xlabel("Drawdown")
            ax2.set_ylabel("Frequência")
        else:
            ax2.text(0.5, 0.5, "Sem drawdowns negativos", ha='center', va='center')
        fig.tight_layout()
        # Estatísticas
        stats_text = (f"<b>Máx Drawdown:</b> {max_dd:.2%}<br>"
                      f"<b>Média Drawdown:</b> {avg_dd:.2%}<br>"
                      f"<b>Maior tempo sob água:</b> {max_duration} períodos")
        stats_label = QLabel(stats_text)
        stats_label.setTextFormat(1)
        layout.addWidget(canvas)
        layout.addWidget(stats_label)