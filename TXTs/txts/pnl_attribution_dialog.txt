"""
PnLAttributionDialog
--------------------

Este diálogo mostra a contribuição de lucro/prejuízo de cada posição
actualmente no portefólio, com um gráfico de barras e um resumo numérico.

Uso:
    dlg = PnLAttributionDialog(metrics)
    dlg.exec_()

Onde `metrics` é o resultado de `portfolio_manager.calculate_metrics`,
contendo a lista de posições e o lucro total.
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd


class PnLAttributionDialog(QDialog):
    def __init__(self, metrics: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Atribuição de PnL por Ativo")
        layout = QVBoxLayout(self)
        positions = metrics.get('positions', []) if metrics else []
        if not positions:
            layout.addWidget(QLabel("Nenhuma posição para atribuição."))
            return
        # Extrai PnL por ticker
        tickers = [p['ticker'] for p in positions]
        pnls = [p.get('profit', 0) for p in positions]
        # Cria figura
        fig = Figure(figsize=(8, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        bars = ax.bar(tickers, pnls, color=['green' if v >= 0 else 'red' for v in pnls])
        ax.set_ylabel('Lucro / Prejuízo (€)')
        ax.set_title('Contribuição de PnL por Ativo')
        ax.set_xticklabels(tickers, rotation=45, ha='right')
        # Adiciona valores nas barras
        for bar, val in zip(bars, pnls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.2f}",
                    ha='center', va='bottom' if val >= 0 else 'top')
        fig.tight_layout()
        layout.addWidget(canvas)
        # Resumo
        total_profit = metrics.get('total_profit', 0)
        total_value = metrics.get('total_value', 0)
        summary = (f"<b>Total PnL:</b> {total_profit:.2f}€<br>"
                   f"<b>Valor Total do Portefólio:</b> {total_value:.2f}€")
        lbl_summary = QLabel(summary)
        lbl_summary.setTextFormat(1)
        layout.addWidget(lbl_summary)