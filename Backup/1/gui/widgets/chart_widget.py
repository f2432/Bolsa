from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class ChartWidget(QWidget):
    """A reusable chart widget that embeds a Matplotlib plot in a PyQt5 application."""
    def __init__(self, parent=None):
        super(ChartWidget, self).__init__(parent)
        # Create a matplotlib figure and canvas
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        # Matplotlib navigation toolbar for interactivity (zoom, pan, etc.)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, series, title="Chart", xlabel="Date", ylabel="Price"):
        """Plot a pandas Series (or list of values) on the chart."""
        self.ax.clear()
        if hasattr(series, 'index'):
            # If series has an index (like pandas Series), plot using index for x-axis
            self.ax.plot(series.index, series.values, label=ylabel)
        else:
            self.ax.plot(series, label=ylabel)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()
        self.figure.autofmt_xdate()  # Auto-format date labels if present
        self.canvas.draw()
