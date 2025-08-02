from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Importa funções de visualização genéricas
# Tenta primeiro um import directo. Se falhar, procura o ficheiro visualizations.py
# no mesmo directório (útil quando a aplicação é distribuída sem tornar visualizations
# um pacote instalável).
plot_price_line = None
plot_volume = None
plot_indicators = None
plot_feature_importance = None
try:
    from visualizations import plot_price_line, plot_volume, plot_indicators, plot_feature_importance
except Exception:
    # fallback: tenta importar visualizations a partir do mesmo diretório deste ficheiro
    import importlib.util
    import os
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # Primeiro procura visualizations.py no mesmo directório
    candidate_paths = [os.path.join(this_dir, "visualizations.py")]
    # Depois procura no directório pai (por ex. widgets/.. para encontrar gui/visualizations.py)
    parent_dir = os.path.dirname(this_dir)
    candidate_paths.append(os.path.join(parent_dir, "visualizations.py"))
    loaded = False
    for vis_path in candidate_paths:
        if os.path.exists(vis_path):
            spec = importlib.util.spec_from_file_location("visualizations", vis_path)
            vis_module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(vis_module)
                plot_price_line = getattr(vis_module, "plot_price_line", None)
                plot_volume = getattr(vis_module, "plot_volume", None)
                plot_indicators = getattr(vis_module, "plot_indicators", None)
                plot_feature_importance = getattr(vis_module, "plot_feature_importance", None)
                loaded = True
                break
            except Exception:
                # Continua a tentar outras paths
                plot_price_line = plot_volume = plot_indicators = plot_feature_importance = None
    if not loaded:
        # Se não encontrou em lado nenhum, mantém as funções a None
        plot_price_line = plot_volume = plot_indicators = plot_feature_importance = None

class ChartWidget(QWidget):
    """
    Widget de gráfico Matplotlib para PyQt5, pronto para overlay de sinais,
    datas de compra e previsões IA.
    """
    def __init__(self, parent=None):
        super(ChartWidget, self).__init__(parent)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, series, title="Gráfico", xlabel="Data", ylabel="Preço",
             buy_dates=None, buy_prices=None,
             pred_dates=None, pred_prices=None, pred_directions=None):
        # Este método original plota uma única série com marcações de compra/venda e previsões.
        # Mantemos compatibilidade, criando um eixo único sempre que necessário.
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # Guarda referência ao último eixo criado para manter compatibilidade com métodos externos
        self.ax = ax
        # Preço histórico
        if hasattr(series, 'index'):
            ax.plot(series.index, series.values, label=ylabel)
        else:
            ax.plot(series, label=ylabel)
        # Linhas verticais para datas de compra (uma legenda só)
        buy_date_label_used = False
        if buy_dates:
            for d in buy_dates:
                ax.axvline(
                    x=d, color='blue', linestyle=':', alpha=0.5, linewidth=2,
                    label='Data de Compra' if not buy_date_label_used else ""
                )
                buy_date_label_used = True
        # Linhas horizontais para preços de compra (uma legenda só)
        buy_price_label_used = False
        if buy_prices:
            for p in buy_prices:
                ax.axhline(
                    y=p, color='orange', linestyle='--', alpha=0.6, linewidth=1.3,
                    label='Preço de Compra' if not buy_price_label_used else ""
                )
                buy_price_label_used = True
        # Overlay de previsões
        label_used = {"Previsão Subida": False, "Previsão Queda": False, "Previsão": False}
        if pred_dates is not None and pred_prices is not None:
            for i, (d, p) in enumerate(zip(pred_dates, pred_prices)):
                if pred_directions is not None:
                    cor = 'green' if pred_directions[i] in [1, 2] else 'red'
                    label = 'Previsão Subida' if pred_directions[i] in [1, 2] else 'Previsão Queda'
                else:
                    cor = 'purple'
                    label = 'Previsão'
                show_label = label if not label_used[label] else ""
                ax.scatter([d], [p], color=cor, marker='^', s=90, label=show_label, zorder=5)
                label_used[label] = True
                # Linha pontilhada ao preço real (se existir)
                if hasattr(series, 'index') and d in series.index:
                    preco_real = series.loc[d]
                    ax.plot([d, d], [preco_real, p], color=cor, linestyle='dotted', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        self.figure.autofmt_xdate()
        self.canvas.draw()

    def plot_price_and_volume(self, data):
        """
        Plota o preço de fecho e o volume em dois painéis. Utiliza as funções
        definidas em visualizations.py se disponíveis.
        """
        if plot_price_line is None or plot_volume is None:
            return
        if plot_price_line is None or plot_volume is None:
            return
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        plot_price_line(data, ax=ax1)
        ax2 = self.figure.add_subplot(212, sharex=ax1)
        plot_volume(data, ax=ax2)
        # Guarda o primeiro eixo como referência
        self.ax = ax1
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_price_with_indicators(self, data, indicators):
        """
        Plota o preço de fecho sobreposto com indicadores técnicos seleccionados e o volume abaixo.
        `indicators` deve ser uma lista de nomes de colunas já presentes em `data`.
        """
        if plot_price_line is None or plot_indicators is None or plot_volume is None:
            return
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        # Preço e indicadores no mesmo eixo
        plot_indicators(data, indicators, ax=ax1)
        ax1.set_xlabel('')
        # Volume no segundo eixo
        ax2 = self.figure.add_subplot(212, sharex=ax1)
        plot_volume(data, ax=ax2)
        # Guarda o primeiro eixo como referência para compatibilidade
        self.ax = ax1
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_feature_importances(self, importances, top_n=10):
        """Plota as importâncias das features no widget."""
        if plot_feature_importance is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        plot_feature_importance(importances, top_n=top_n, ax=ax)
        self.ax = ax
        self.canvas.draw()

