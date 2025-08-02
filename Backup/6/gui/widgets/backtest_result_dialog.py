from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QDialogButtonBox, QPushButton, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd

class BacktestResultDialog(QDialog):
    def __init__(self, info, metricas, equity_curve, trades, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resultados do Backtest")
        self.trades = trades
        layout = QVBoxLayout()

        head_txt = (f"<b>Estratégia:</b> {info.get('estrategia', '-')}&nbsp;&nbsp;"
                    f"<b>Período:</b> {info.get('inicio', '-')} a {info.get('fim', '-')}")
        layout.addWidget(QLabel(head_txt))
        metrics_html = "<br>".join([
            f"<b>Retorno Total:</b> {metricas.get('retorno', 'N/A')}",
            f"<b>Drawdown Máximo:</b> {metricas.get('drawdown', 'N/A')}",
            f"<b>Sharpe Ratio:</b> {metricas.get('sharpe', 'N/A')}",
            f"<b>Nº de Trades:</b> {metricas.get('num_trades', 'N/A')}",
            f"<b>% Trades Vencedores:</b> {metricas.get('win_rate', 'N/A')}",
        ])
        layout.addWidget(QLabel(metrics_html))

        # Gráfico do equity curve
        if equity_curve is not None and isinstance(equity_curve, pd.Series):
            fig, ax = plt.subplots(figsize=(7, 2.8), dpi=100)
            equity_curve.plot(ax=ax)
            ax.set_title("Evolução do saldo (Equity Curve)")
            ax.set_xlabel("Data")
            ax.set_ylabel("Saldo (€)")
            ax.grid(True)
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            plt.close(fig)

        # Tabela de trades
        if trades is not None and not trades.empty:
            table = QTableWidget(trades.shape[0], trades.shape[1])
            table.setHorizontalHeaderLabels([str(c) for c in trades.columns])
            for row in range(trades.shape[0]):
                for col in range(trades.shape[1]):
                    table.setItem(row, col, QTableWidgetItem(str(trades.iloc[row, col])))
            table.resizeColumnsToContents()
            layout.addWidget(QLabel("<b>Trades executadas:</b>"))
            layout.addWidget(table)
        elif trades is not None:
            layout.addWidget(QLabel("Nenhuma trade foi executada."))

        # Botões: exportar trades e fechar
        button_layout = QVBoxLayout()
        export_btn = QPushButton("Exportar Trades para CSV")
        export_btn.clicked.connect(self.export_trades)
        button_layout.addWidget(export_btn)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        button_layout.addWidget(buttons)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def export_trades(self):
        if self.trades is None or self.trades.empty:
            QMessageBox.information(self, "Exportar Trades", "Nenhuma trade para exportar.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Exportar Trades", "", "CSV (*.csv)")
        if file_path:
            self.trades.to_csv(file_path, index=False)
            QMessageBox.information(self, "Exportação", f"Trades exportadas para:\n{file_path}")
