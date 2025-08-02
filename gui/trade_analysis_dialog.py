"""
TradeAnalysisDialog
-------------------

Diálogo para analisar os resultados das trades individuais de um backtest.
Mostra um histograma dos lucros e perdas (PnL) de cada trade e algumas
estatísticas de desempenho como média, mediana e taxa de sucesso.
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd

# Importa visualizations opcionalmente
import importlib.util
import os

plot_histogram = None

def _load_visualizations():
    global plot_histogram
    try:
        from visualizations import plot_histogram as ph  # type: ignore
        plot_histogram = ph
        return
    except Exception:
        pass
    this_dir = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(this_dir, 'visualizations.py'), os.path.join(os.path.dirname(this_dir), 'visualizations.py')]:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location('visualizations', path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore
                plot_histogram = getattr(module, 'plot_histogram', None)
                return
            except Exception:
                continue

_load_visualizations()


class TradeAnalysisDialog(QDialog):
    def __init__(self, trades: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Análise de Trades")
        layout = QVBoxLayout(self)
        if trades is None or trades.empty or 'Resultado (€)' not in trades.columns:
            layout.addWidget(QLabel("Não há resultados de trades para analisar."))
            return
        # Extrai apenas trades de venda com resultado
        # Converte valores vazios para NaN antes de transformar em float
        pnl = trades['Resultado (€)'].replace('', float('nan')).dropna().astype(float)
        # Histograma de PnL
        fig = Figure(figsize=(7, 3))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        if plot_histogram is not None:
            try:
                plot_histogram(pnl, ax=ax)
            except Exception:
                ax.hist(pnl, bins=30, alpha=0.7)
                ax.set_title("Distribuição de Lucros/Perdas das Trades")
        else:
            ax.hist(pnl, bins=30, alpha=0.7)
            ax.set_title("Distribuição de Lucros/Perdas das Trades")
        ax.set_xlabel("Lucro/Perda (€)")
        ax.set_ylabel("Frequência")
        layout.addWidget(canvas)
        # Estatísticas
        media = pnl.mean()
        mediana = pnl.median()
        taxa_sucesso = (pnl > 0).mean() * 100
        stats_lines = [
            f"Média de PnL: {media:.2f} €",
            f"Mediana de PnL: {mediana:.2f} €",
            f"Taxa de Trades Vencedoras: {taxa_sucesso:.1f}%"
        ]
        layout.addWidget(QLabel("<br>".join(stats_lines)))