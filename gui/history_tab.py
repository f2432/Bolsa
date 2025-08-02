"""
HistoryTab
----------

Um separador (QWidget) que mostra o histórico de previsões registadas
pelo PredictionLogger. Permite visualizar todas as previsões
armazenadas no ficheiro prediction_log.csv, com colunas como data,
ticker, modelo, horizonte, direcção, probabilidades ajustadas, preço
real e preço previsto. Pode ser expandido no futuro com filtros
interactivos.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel
import pandas as pd


class HistoryTab(QWidget):
    def __init__(self, pred_logger, parent=None):
        super().__init__(parent)
        self.pred_logger = pred_logger
        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        layout.addWidget(QLabel("Histórico de Previsões"))
        layout.addWidget(self.table)
        self.refresh()

    def refresh(self):
        """Carrega o CSV de log e preenche a tabela."""
        try:
            df = self.pred_logger.get_log()
        except Exception:
            df = pd.DataFrame()
        if df is None or df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        # Ajusta colunas
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(list(df.columns))
        self.table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(df.columns):
                val = row[col]
                self.table.setItem(i, j, QTableWidgetItem(str(val)))
        self.table.resizeColumnsToContents()