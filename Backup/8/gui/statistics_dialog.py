"""
Dialogo para visualizacoes estatisticas adicionais (histograma, correlacao e ACF/PACF).

Este modulo define um QDialog com tres tabs para explorar distribuicoes, correlacoes e
autocorrelacoes de uma serie temporal financeira. Requer PyQt5 e utiliza as funcoes
definidas em `visualizations.py`.
"""

import pandas as pd
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# Importa dinamicamente as funções de visualização tentando primeiro o módulo no mesmo
# diretório que este ficheiro. Se falhar, tenta importações convencionais.
import importlib.util
import os

def _load_visualizations():
    """Carrega o módulo visualizations a partir do mesmo directório ou via import normal."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Caminho para visualizations.py no mesmo directório
    local_path = os.path.join(current_dir, 'visualizations.py')
    if os.path.isfile(local_path):
        spec = importlib.util.spec_from_file_location('visualizations', local_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        return module
    # Tenta import normal
    try:
        return importlib.import_module('visualizations')
    except ImportError:
        # Tenta import a partir de gui.visualizations
        try:
            return importlib.import_module('gui.visualizations')
        except ImportError:
            return None

_vis_module = _load_visualizations()
if _vis_module is None:
    raise ImportError("Não foi possível carregar o módulo visualizations.")

plot_histogram = _vis_module.plot_histogram
plot_correlation_matrix = _vis_module.plot_correlation_matrix
plot_acf_pacf = _vis_module.plot_acf_pacf


class StatisticsDialog(QDialog):
    def __init__(self, data: pd.DataFrame, parent=None):
        """
        Inicializa o dialogo de estatisticas.

        :param data: DataFrame contendo pelo menos a coluna 'Close' e, opcionalmente,
                     outras features para correlacao.
        :param parent: widget pai (opcional).
        """
        super().__init__(parent)
        self.setWindowTitle("Análise Estatística")
        self.resize(800, 600)
        self.data = data.copy()

        layout = QVBoxLayout()
        tabs = QTabWidget()

        # Tab 1: Histograma de retornos
        hist_tab = QWidget()
        hist_layout = QVBoxLayout()
        # Calcula retornos percentuais para o histograma
        returns = self.data['Close'].pct_change().dropna()
        fig1 = plot_histogram(returns, bins=50)
        canvas1 = FigureCanvas(fig1)
        hist_layout.addWidget(canvas1)
        hist_tab.setLayout(hist_layout)
        tabs.addTab(hist_tab, "Histograma")

        # Tab 2: Matriz de correlação (usando features numericas se existirem)
        corr_tab = QWidget()
        corr_layout = QVBoxLayout()
        # Seleciona apenas colunas numericas para a correlacao
        num_df = self.data.select_dtypes(include=['float', 'int'])
        if not num_df.empty:
            fig2 = plot_correlation_matrix(num_df)
            canvas2 = FigureCanvas(fig2)
            corr_layout.addWidget(canvas2)
        else:
            corr_layout.addWidget(QLabel("Sem colunas numericas para correlação."))
        corr_tab.setLayout(corr_layout)
        tabs.addTab(corr_tab, "Correlação")

        # Tab 3: ACF e PACF dos retornos
        acf_tab = QWidget()
        acf_layout = QVBoxLayout()
        try:
            fig_acf, fig_pacf = plot_acf_pacf(returns, lags=40)
            canvas_acf = FigureCanvas(fig_acf)
            canvas_pacf = FigureCanvas(fig_pacf)
            acf_layout.addWidget(canvas_acf)
            acf_layout.addWidget(canvas_pacf)
        except ImportError:
            acf_layout.addWidget(QLabel("statsmodels nao disponivel para ACF/PACF."))
        acf_tab.setLayout(acf_layout)
        tabs.addTab(acf_tab, "ACF/PACF")

        layout.addWidget(tabs)
        self.setLayout(layout)