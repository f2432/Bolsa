"""
FeatureImportanceDialog
-----------------------

Este módulo define um diálogo para visualizar as importâncias das features
num gráfico de barras. É utilizado para mostrar as importâncias sem
substituir o gráfico principal da aplicação.
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Importa dinamicamente as funções de visualização de visualizations.py.
# Procura o módulo no mesmo directório deste ficheiro e no directório pai,
# para acomodar diferentes estruturas de projecto.
import importlib.util
import os

# Tenta carregar visualizations
plot_feature_importance = None

def _load_visualizations():
    global plot_feature_importance
    # Primeiro tenta importar a partir do sys.path
    try:
        from visualizations import plot_feature_importance as pfi  # type: ignore
        plot_feature_importance = pfi
        return
    except Exception:
        pass
    # Depois procura por visualizations.py próximo a este ficheiro
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [os.path.join(this_dir, "visualizations.py"),
                       os.path.join(os.path.dirname(this_dir), "visualizations.py")]
    for vis_path in candidate_paths:
        if os.path.exists(vis_path):
            spec = importlib.util.spec_from_file_location("visualizations", vis_path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore
                plot_feature_importance = getattr(module, "plot_feature_importance", None)
                if plot_feature_importance is not None:
                    return
            except Exception:
                continue


_load_visualizations()


class FeatureImportanceDialog(QDialog):
    """
    Diálogo para mostrar a importância das features num gráfico de barras.
    Aceita uma lista de tuplos (feature, valor) e desenha as top_n.
    """

    def __init__(self, importances, parent=None, top_n: int = 10):
        super().__init__(parent)
        self.setWindowTitle("Importância das Features")
        layout = QVBoxLayout(self)
        # Prepara figura e canvas do matplotlib
        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # Desenha o gráfico se a função estiver disponível
        if plot_feature_importance is not None:
            try:
                plot_feature_importance(importances, top_n=top_n, ax=ax)
            except Exception:
                ax.text(0.5, 0.5, "Erro ao desenhar importâncias", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "Módulo de visualização não encontrado", ha='center', va='center')
        layout.addWidget(canvas)