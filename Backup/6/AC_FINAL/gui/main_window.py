
    import yfinance as yf
    import joblib
    from PyQt5.QtWidgets import QTabWidget, QPushButton, QMessageBox, QVBoxLayout
    from ai.train_utils import train_direction_model, train_price_regressor
    from backtest.backtester import Backtester
from gui.plot_utils import plot_signals_with_predictions

    class MainWindow(QMainWindow):
        def __init__(self, data_provider, portfolio_manager, predictor, initial_capital, tickers, strategy):
            super().__init__()
            self.data_provider = data_provider
            self.portfolio_manager = portfolio_manager
            self.predictor = predictor
            self.strategy = strategy
            self.initial_capital = initial_capital
            self.tickers = tickers
            self.setWindowTitle("Trading App")

            # Criar layout principal
            layout = QVBoxLayout()
            self.train_button = QPushButton("Treinar Modelos")
            self.train_button.clicked.connect(self.train_models_and_backtest)
            layout.addWidget(self.train_button)

            # Widget central
            central_widget = QWidget()
            central_widget.setLayout(layout)
            tabs = QTabWidget()
        tabs.addTab(central_widget, 'Painel Principal')
        from gui.explore_tab import ExploreTab
        explore_tab = ExploreTab(self.data_provider, self.predictor, self.portfolio_manager, '', self)
        tabs.addTab(explore_tab, 'Explorar')
        ia_tab = IATab(self.data_provider, self.predictor)
        tabs.addTab(ia_tab, 'Previsão IA')
        self.setCentralWidget(tabs)

        def train_models_and_backtest(self):
            try:
                ticker = self.tickers[0]
                df = yf.download(ticker, period="1y", interval="1d")
                clf, _ = train_direction_model(df)
                reg, _ = train_price_regressor(df)
                joblib.dump(clf, "ai/logistic_model.pkl")
                joblib.dump(reg, "ai/rf_model.pkl")

                signals = self.strategy.generate_signals(df)
                predictions = None
                if hasattr(self.strategy, 'predictor'):
                    try:
                        predictions = self.strategy.predictor.predict_price(df)
                    except Exception as e:
                        print(f'Erro ao obter previsões de preço: {e}')
                bt = Backtester(df, signals, self.initial_capital)
                results = bt.run()

                msg = QMessageBox()
                msg.setWindowTitle("Modelos treinados e Backtest")
                summary = f"💰 Lucro: ${results['profit']:.2f}\n📊 Nº trades: {results['num_trades']}"
                msg.setText("✅ Treino concluído com sucesso!
" + summary)
                msg.exec_()
            except Exception as e:
                msg = QMessageBox()
                msg.setWindowTitle("Erro")
                msg.setText(f"❌ Erro ao treinar ou fazer backtest:\n{e}")
                msg.exec_()
