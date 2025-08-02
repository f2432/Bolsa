from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class ChartWidget(QWidget):
    """
    Widget de gráfico Matplotlib para PyQt5, pronto para overlay de sinais,
    datas de compra e previsões IA.
    """
    def __init__(self, parent=None):
        super(ChartWidget, self).__init__(parent)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, series, title="Gráfico", xlabel="Data", ylabel="Preço",
             buy_dates=None, buy_prices=None,
             pred_dates=None, pred_prices=None, pred_directions=None):
        self.ax.clear()
        # Preço histórico
        if hasattr(series, 'index'):
            self.ax.plot(series.index, series.values, label=ylabel)
        else:
            self.ax.plot(series, label=ylabel)
        # Linhas verticais para datas de compra (uma legenda só)
        buy_date_label_used = False
        if buy_dates:
            for d in buy_dates:
                self.ax.axvline(
                    x=d, color='blue', linestyle=':', alpha=0.5, linewidth=2,
                    label='Data de Compra' if not buy_date_label_used else ""
                )
                buy_date_label_used = True
        # Linhas horizontais para preços de compra (uma legenda só)
        buy_price_label_used = False
        if buy_prices:
            for p in buy_prices:
                self.ax.axhline(
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
                self.ax.scatter([d], [p], color=cor, marker='^', s=90, label=show_label, zorder=5)
                label_used[label] = True
                # Linha pontilhada ao preço real (se existir)
                if hasattr(series, 'index') and d in series.index:
                    preco_real = series.loc[d]
                    self.ax.plot([d, d], [preco_real, p], color=cor, linestyle='dotted', alpha=0.7)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend(loc='best')
        self.figure.autofmt_xdate()
        self.canvas.draw()

