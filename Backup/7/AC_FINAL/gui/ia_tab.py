
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QMessageBox
from gui.plot_utils import plot_signals_with_predictions
from gui.dialogs import PredictionDialog

class IATab(QWidget):
    def __init__(self, data_provider, predictor, parent=None):
        super().__init__(parent)
        self.data_provider = data_provider
        self.predictor = predictor

        layout = QVBoxLayout()
        self.setLayout(layout)

        input_layout = QHBoxLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ex: AAPL")
        input_layout.addWidget(QLabel("Ticker:"))
        input_layout.addWidget(self.ticker_input)

        self.btn_prever = QPushButton("Prever")
        self.btn_prever.clicked.connect(self.executar_previsao)
        input_layout.addWidget(self.btn_prever)

        layout.addLayout(input_layout)

    def executar_previsao(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Aviso", "Introduz um ticker válido.")
            return

        try:
            df = self.data_provider.get_historical_data(ticker)
            if df is None or df.empty:
                QMessageBox.warning(self, "Erro", f"Não foi possível obter dados para {ticker}")
                return

            # Previsões
            direction = self.predictor.predict_direction(df)
            proba = self.predictor.predict_proba(df)
            price_pred = self.predictor.predict_price(df)

            # Mostrar janela com info detalhada
            dlg = PredictionDialog(direction, proba, price_pred, features=None)
            dlg.exec_()

            # Mostrar gráfico
            signals = self.predictor.predict(df)
            plot_signals_with_predictions(df, signals, price_pred)

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Ocorreu um erro: {e}")
