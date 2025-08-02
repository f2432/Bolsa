"""
BacktestAnalysisDialog
---------------------

Diálogo para analisar resultados de um backtest. Mostra um histograma dos
retornos periódicos da estratégia e a curva de saldo (equity curve).
Usa funções de visualização do módulo visualizations.py quando disponíveis.
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

# Importa dinamicamente as funções de visualização
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
    # tenta carregar a partir do directório local ou pai
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


class BacktestAnalysisDialog(QDialog):
    def __init__(self, equity_curve: pd.Series, trades: pd.DataFrame = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Análise do Backtest")
        layout = QVBoxLayout(self)
        # Mostra a curva de saldo
        if equity_curve is not None and isinstance(equity_curve, pd.Series):
            fig1 = Figure(figsize=(7, 2.5))
            canvas1 = FigureCanvas(fig1)
            ax1 = fig1.add_subplot(111)
            equity_curve.plot(ax=ax1)
            ax1.set_title("Evolução do saldo (Equity Curve)")
            ax1.set_xlabel("Data")
            ax1.set_ylabel("Saldo (€)")
            ax1.grid(True)
            layout.addWidget(canvas1)
        # Histograma dos retornos
        if equity_curve is not None and isinstance(equity_curve, pd.Series):
            # calcula retornos percentuais
            returns = equity_curve.pct_change().dropna()
            fig2 = Figure(figsize=(7, 2.5))
            canvas2 = FigureCanvas(fig2)
            ax2 = fig2.add_subplot(111)
            if plot_histogram is not None:
                try:
                    plot_histogram(returns, ax=ax2)
                except Exception:
                    ax2.hist(returns, bins=30, alpha=0.7)
                    ax2.set_title("Distribuição de Retornos do Backtest")
                    ax2.set_xlabel("Retorno")
                    ax2.set_ylabel("Frequência")
            else:
                # fallback: usa matplotlib directo
                ax2.hist(returns, bins=30, alpha=0.7)
                ax2.set_title("Distribuição de Retornos do Backtest")
                ax2.set_xlabel("Retorno")
                ax2.set_ylabel("Frequência")
            layout.addWidget(canvas2)
        # Informação adicional
        if trades is not None and not trades.empty:
            num_trades = trades.shape[0]
            win_rate = (trades['Resultado'] > 0).mean() * 100 if 'Resultado' in trades.columns else None
            info_lines = []
            info_lines.append(f"Nº Trades: {num_trades}")
            if win_rate is not None:
                info_lines.append(f"% Trades Vencedoras: {win_rate:.1f}%")
            info_label = QLabel("<br>".join(info_lines))
            layout.addWidget(info_label)