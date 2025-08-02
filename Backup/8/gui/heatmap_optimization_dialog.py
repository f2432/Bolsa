"""
HeatmapOptimizationDialog
------------------------

Diálogo que executa um conjunto de backtests variando dois parâmetros de
estratégias simples e mostra um mapa de calor (heatmap) de uma métrica
de desempenho (Sharpe ratio ou retorno total).  Actualmente implementado
para a estratégia SMA Crossover (short_window, long_window).
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from strategies.sma_crossover import SMACrossoverStrategy
from backtest.backtester import Backtester
from backtest.metrics import calculate_metrics


class HeatmapOptimizationDialog(QDialog):
    def __init__(self, data, initial_capital=10000, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mapa de Otimização de Estratégia")
        layout = QVBoxLayout(self)
        if data is None or data.empty or 'Close' not in data.columns:
            layout.addWidget(QLabel("Dados insuficientes para otimização."))
            return
        # Define grelha de parâmetros (short e long window) para SMA Crossover
        short_vals = [10, 20, 30, 40, 50]
        long_vals = [50, 100, 150, 200]
        heatmap = np.zeros((len(short_vals), len(long_vals)))
        # Prepara backtester
        backtester = Backtester(initial_capital=initial_capital)
        # Calcula métrica (Sharpe) para cada combinação short/long
        for i, sw in enumerate(short_vals):
            for j, lw in enumerate(long_vals):
                if sw >= lw:
                    heatmap[i, j] = np.nan
                    continue
                strategy = SMACrossoverStrategy(short_window=sw, long_window=lw)
                # Gera sinais e corre backtest
                try:
                    signals = strategy.generate_signals(data)
                    results = backtester.run(data, signals)
                    metrics = calculate_metrics(data, signals, results['equity_curve'], results.get('trades'))
                    # Extrai Sharpe ratio como float (removendo %) ou usa retorno total
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
        ax.set_xticks(range(len(long_vals)))
        ax.set_yticks(range(len(short_vals)))
        ax.set_xticklabels(long_vals)
        ax.set_yticklabels(short_vals)
        ax.set_xlabel('Long Window')
        ax.set_ylabel('Short Window')
        ax.set_title('Mapa de Sharpe Ratio para SMA Crossover')
        # Adiciona anotações de valores
        for i in range(len(short_vals)):
            for j in range(len(long_vals)):
                val = heatmap[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if val < np.nanmax(heatmap)/2 else 'black', fontsize=8)
        fig.colorbar(im, ax=ax, label='Sharpe Ratio')
        layout.addWidget(canvas)